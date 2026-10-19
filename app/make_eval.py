import argparse
import json
import random
import sys
import time
from pathlib import Path

from google import genai
from google.genai import errors, types

from app.rag import CHAT_MODEL
from app.store import VectorStore

MIN_CHARS = 200
MAX_RETRIES = 5

PROMPT = """Write one question in Persian that a user could ask and that the passage below answers.
Rules:
- The answer must be in the passage, and the question must make sense without seeing it.
- Paraphrase: do not copy long phrases from the passage.
- Name the subject (company, product, report) so the question is not ambiguous.
- If the passage has no clear fact to ask about (only unlabeled numbers, only a title), reply with exactly: SKIP
Reply with the question only.

<passage>
{text}
</passage>"""


def _generate(client: genai.Client, prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            r = client.models.generate_content(
                model=CHAT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7),
            )
            return (r.text or "").strip()
        except errors.APIError as e:
            if e.code != 429 or attempt == MAX_RETRIES - 1:
                raise
            print("quota exceeded, waiting 30s…", file=sys.stderr, flush=True)
            time.sleep(30)
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate retrieval eval questions from stored chunks")
    parser.add_argument("--per-user", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/eval/questions.jsonl")
    args = parser.parse_args()

    with VectorStore.connect() as store:
        rows = store.conn.execute(
            """
            SELECT c.user_id, d.filename, c.page, c.content
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE length(c.content) >= %s
            ORDER BY c.id
            """,
            (MIN_CHARS,),
        ).fetchall()

    by_user: dict[str, list[tuple]] = {}
    for row in rows:
        by_user.setdefault(row[0], []).append(row)

    # Fixed seed: the same chunks are picked every run, so results are comparable.
    rng = random.Random(args.seed)
    client = genai.Client()
    items = []
    for user_id, pool in sorted(by_user.items()):
        rng.shuffle(pool)
        made = 0
        for _, filename, page, content in pool:
            if made >= args.per_user:
                break
            q = _generate(client, PROMPT.format(text=content.replace("</passage>", "")))
            if not q or q.upper().startswith("SKIP"):
                print(f"[{user_id}] skip {filename} p{page}", file=sys.stderr)
                continue
            made += 1
            items.append(
                {"user_id": user_id, "question": q, "filename": filename, "page": page, "source_text": content}
            )
            print(f"[{user_id}] {made}/{args.per_user} {filename} p{page}: {q}", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} questions to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
