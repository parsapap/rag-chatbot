import argparse
import json
import os
import sys
import time
from pathlib import Path

from app.db.store import SearchResult, VectorStore
from app.services.embeddings import Embedder
from app.services.loaders import normalize
from app.services.rag import CANDIDATE_FACTOR, RAG, fuse

KS = (1, 3, 5, 10)
REWRITE_PAUSE_SECONDS = 5


def rank_of(results: list[SearchResult], filename: str, page: int) -> int | None:
    # Ground truth is a page, not a chunk id, so the eval set stays valid
    # when chunk size or overlap change.
    for i, r in enumerate(results, 1):
        if r.filename == filename and r.page == page:
            return i
    return None


def best_rank(results: list[SearchResult], filename: str, pages: list[int]) -> int | None:
    # Some facts appear on several pages (a total in a table and in a note); any of them counts.
    ranks = [r for p in pages if (r := rank_of(results, filename, p)) is not None]
    return min(ranks, default=None)


def evaluate_follow_ups(store, embedder, rag, items: list[dict]) -> tuple[list[int | None], list[int | None]]:
    """Searches each follow-up as typed and after the rewrite; returns both rank lists."""
    rewrites = []
    for it in items:
        rewrites.append(rag.rewrite_question(it["question"], [tuple(t) for t in it["history"]]))
        # A burst of rewrites hits the free-tier quota, and a failed rewrite would be scored as "not rewritten".
        time.sleep(REWRITE_PAUSE_SECONDS)
    typed_vectors = embedder.embed_queries([it["question"] for it in items])
    rewritten_vectors = embedder.embed_queries([r or it["question"] for it, r in zip(items, rewrites)])
    typed, rewritten = [], []
    for it, r, tv, rv in zip(items, rewrites, typed_vectors, rewritten_vectors):
        a = best_rank(hybrid(store, it["user_id"], it["question"], tv), it["filename"], it["pages"])
        b = best_rank(hybrid(store, it["user_id"], r or it["question"], rv), it["filename"], it["pages"])
        typed.append(a)
        rewritten.append(b)
        print(f"  {str(a or '-'):>2} -> {str(b or '-'):<2}  {it['question']}  =>  {r or '(not rewritten)'}")
    return typed, rewritten


def hybrid(store, user_id: str, question: str, vector: list[float]) -> list[SearchResult]:
    # Same fusion as RAG.retrieve, without spread_across_files, so ranks past 5 are still measured.
    if os.environ.get("EVAL_VECTOR_ONLY"):
        return store.search(user_id, vector, max(KS))
    n = max(KS) * CANDIDATE_FACTOR
    return fuse(store.search(user_id, vector, n), store.keyword_search(user_id, normalize(question), n))[: max(KS)]


def summarize(ranks: list[int | None], ks: tuple[int, ...] = KS) -> dict[str, float]:
    n = len(ranks)
    out = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    # Reciprocal rank: 1 at rank 1, 0.5 at rank 2, ..., 0 if not found within max(ks).
    out["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure retrieval quality on an eval set")
    parser.add_argument("--questions", default="data/eval/questions.jsonl")
    parser.add_argument("--show-misses", action="store_true", help="print questions not found in the top 5")
    args = parser.parse_args()

    path = Path(args.questions)
    if not path.is_file():
        print(f"error: {path} not found, run app.make_eval first", file=sys.stderr)
        return 1
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not items:
        print(f"error: {path} is empty", file=sys.stderr)
        return 1

    for it in items:
        # Generated items carry "page"; hand-written ones carry "pages", empty when the documents have no answer.
        it.setdefault("pages", [it["page"]] if it.get("page") is not None else [])
    # Items with a "history" of [question, answer] turns are follow-ups, scored in their own section.
    follow_ups = [it for it in items if it.get("history")]
    items = [it for it in items if not it.get("history")]
    answerable = [it for it in items if it["pages"]]

    embedder = Embedder()
    vectors = embedder.embed_queries([it["question"] for it in items])
    ranks: list[int | None] = []
    top_scores: dict[bool, list[float]] = {True: [], False: []}
    with VectorStore.connect() as store:
        for it, v in zip(items, vectors):
            results = hybrid(store, it["user_id"], it["question"], v)
            top_scores[bool(it["pages"])].append(results[0].score if results else 0.0)
            if not it["pages"]:
                continue
            rank = best_rank(results, it["filename"], it["pages"])
            ranks.append(rank)
            if args.show_misses and (rank is None or rank > 5):
                print(f"\nMISS (rank={rank}) [{it['user_id']}] {it['question']}")
                print(f"  expected: {it['filename']} p{it['pages']}")
                for r in results[:3]:
                    print(f"  got:      {r.filename} p{r.page}  score={r.score:.3f}")

    print(f"\nanswerable questions: {len(answerable)}")
    if ranks:
        for name, value in summarize(ranks).items():
            print(f"  {name:7} {value:.3f}")

    # If every answerable question scores above every unanswerable one, a score cutoff could skip the LLM.
    print("\ntop-1 score:")
    for label, key in (("answerable", True), ("no answer", False)):
        s = top_scores[key]
        if s:
            print(f"  {label:10} min={min(s):.3f} max={max(s):.3f}  ({len(s)} q)")

    print("\nhit@5 per file:")
    per_file: dict[str, list[int | None]] = {}
    for it, r in zip(answerable, ranks):
        per_file.setdefault(it["filename"], []).append(r)
    for name, rs in sorted(per_file.items()):
        print(f"  {summarize(rs, (5,))['hit@5']:.2f}  ({len(rs):2} q)  {name}")

    if follow_ups:
        print(f"\nfollow-up questions: {len(follow_ups)}  (rank as typed -> rank after rewrite)")
        with VectorStore.connect() as store:
            typed, rewritten = evaluate_follow_ups(store, embedder, RAG(store, embedder), follow_ups)
        print("\n  metric   typed  rewritten")
        for (name, a), b in zip(summarize(typed).items(), summarize(rewritten).values()):
            print(f"  {name:7} {a:6.3f}  {b:9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
