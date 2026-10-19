import argparse
import json
import sys
from pathlib import Path

from app.embeddings import Embedder
from app.store import SearchResult, VectorStore

KS = (1, 3, 5, 10)


def rank_of(results: list[SearchResult], filename: str, page: int) -> int | None:
    # Ground truth is a page, not a chunk id, so the eval set stays valid
    # when chunk size or overlap change.
    for i, r in enumerate(results, 1):
        if r.filename == filename and r.page == page:
            return i
    return None


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

    vectors = Embedder().embed_queries([it["question"] for it in items])
    ranks: list[int | None] = []
    with VectorStore.connect() as store:
        for it, v in zip(items, vectors):
            results = store.search(it["user_id"], v, max(KS))
            rank = rank_of(results, it["filename"], it["page"])
            ranks.append(rank)
            if args.show_misses and (rank is None or rank > 5):
                print(f"\nMISS (rank={rank}) [{it['user_id']}] {it['question']}")
                print(f"  expected: {it['filename']} p{it['page']}")
                for r in results[:3]:
                    print(f"  got:      {r.filename} p{r.page}  score={r.score:.3f}")

    print(f"\nquestions: {len(items)}")
    for name, value in summarize(ranks).items():
        print(f"  {name:7} {value:.3f}")

    print("\nhit@5 per file:")
    per_file: dict[str, list[int | None]] = {}
    for it, r in zip(items, ranks):
        per_file.setdefault(it["filename"], []).append(r)
    for name, rs in sorted(per_file.items()):
        print(f"  {summarize(rs, (5,))['hit@5']:.2f}  ({len(rs):2} q)  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
