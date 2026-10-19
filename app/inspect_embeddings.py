import argparse
import sys

from app.chunking import chunk_pages
from app.embeddings import Embedder, cosine
from app.loaders import LoaderError, load

# Persian on purpose: the demo shows similarity across Persian paraphrases and English.
DEMO_SENTENCES = [
    "فیدیبو کتاب صوتی می‌فروشد.",
    "در فیدیبو می‌توان به کتاب گوش داد.",
    "Fidibo sells audiobooks.",
    "قیمت بنزین در تهران افزایش یافت.",
]


def similarity_demo(embedder: Embedder) -> None:
    vectors = embedder.embed_documents(DEMO_SENTENCES)
    print(f"model: {embedder.model}  |  dimensions: {len(vectors[0])}")
    print(f"first 5 values of sentence 1: {[round(v, 4) for v in vectors[0][:5]]}\n")
    for i, s in enumerate(DEMO_SENTENCES, 1):
        print(f"[{i}] {s}")
    print("\ncosine similarity (1 = same meaning, near 0 = unrelated):")
    print("      " + "".join(f"[{j}]    " for j in range(1, len(vectors) + 1)))
    for i, a in enumerate(vectors, 1):
        print(f"[{i}]  " + "".join(f"{cosine(a, b):.3f}  " for b in vectors))


def search_file(embedder: Embedder, path: str, query: str, top: int) -> int:
    try:
        chunks = chunk_pages(load(path))
    except LoaderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"embedding {len(chunks)} chunks…")
    doc_vectors = embedder.embed_documents([c.text for c in chunks])
    q = embedder.embed_query(query)

    # Brute force: compare the query to every chunk. A vector DB does the same
    # job with an index so it stays fast for millions of chunks.
    scored = sorted(zip(chunks, doc_vectors), key=lambda cv: cosine(q, cv[1]), reverse=True)
    print(f"\nquery: {query}")
    for c, v in scored[:top]:
        print(f"\n--- {c.id}  score={cosine(q, v):.3f} ---")
        print(c.text[:300])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Explore embeddings and semantic similarity")
    parser.add_argument("--file", help="search the chunks of this file")
    parser.add_argument("--query", help="query to search for (with --file)")
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    embedder = Embedder()
    if args.file:
        if not args.query:
            parser.error("--query is required with --file")
        return search_file(embedder, args.file, args.query, args.top)
    similarity_demo(embedder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
