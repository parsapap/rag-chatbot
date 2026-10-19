import argparse
import sys

from app.embeddings import Embedder
from app.store import VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic search over one user's documents")
    parser.add_argument("--user", required=True)
    parser.add_argument("query")
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    q = Embedder().embed_query(args.query)
    with VectorStore.connect() as store:
        results = store.search(args.user, q, args.top)

    if not results:
        print(f"User {args.user} has no documents.")
        return 0
    print(f"Query: {args.query}")
    for r in results:
        print(f"\n--- {r.filename} page {r.page} (chunk {r.chunk_index})  score={r.score:.3f} ---")
        print(r.content[:300])
    return 0


if __name__ == "__main__":
    sys.exit(main())
