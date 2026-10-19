import argparse
import sys

from google.genai import errors

from app.services.embeddings import Embedder
from app.services.rag import RAG, SYSTEM_PROMPT, TOP_K, build_prompt
from app.db.store import VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a question over one user's documents (RAG)")
    parser.add_argument("--user", required=True)
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=TOP_K, help="number of chunks to retrieve")
    parser.add_argument("--show-prompt", action="store_true", help="print the full prompt sent to the model")
    args = parser.parse_args()

    with VectorStore.connect() as store:
        rag = RAG(store, Embedder(), k=args.k)
        try:
            if args.show_prompt:
                results = rag.retrieve(args.user, args.question)
                print("===== system =====\n" + SYSTEM_PROMPT)
                print("\n===== user =====\n" + build_prompt(args.question, results))
                print("\n==================\n")
            answer = rag.ask(args.user, args.question)
        except errors.APIError as e:
            print(f"Gemini error ({e.code}): {e.status}", file=sys.stderr)
            return 1

    print(f"Question: {args.question}\n")
    print(answer.text)
    if answer.sources:
        print("\nSources:")
        for i, s in enumerate(answer.sources, 1):
            mark = "*" if i in answer.cited else " "
            print(f" {mark}[{i}] {s.filename} page {s.page}  score={s.score:.3f}")
        print("(* = cited in the answer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
