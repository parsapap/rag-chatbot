import argparse
import hashlib
import sys
from pathlib import Path

from google.genai import errors

from app.services.chunking import chunk_pages
from app.services.embeddings import Embedder
from app.services.loaders import LoaderError, load
from app.db.store import VectorStore


def ingest_file(store: VectorStore, embedder: Embedder, user_id: str, path: Path) -> str:
    pages = load(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    # Same file for the same user: skip it, so the embedding quota isn't spent twice.
    if (doc_id := store.find_document(user_id, sha)) is not None:
        return f"duplicate, skipped (id={doc_id})"

    chunks = chunk_pages(pages)
    if not chunks:
        return "no text found, skipped"

    vectors = embedder.embed_documents([c.text for c in chunks])
    doc_id = store.add_document(user_id, path.name, sha, len(pages), chunks, vectors)
    return f"saved (id={doc_id}, {len(chunks)} chunks)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Add, list and delete one user's documents")
    parser.add_argument("--user", required=True)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--list", action="store_true", help="list the user's documents")
    parser.add_argument("--delete", type=int, metavar="ID", help="delete the document with this id")
    args = parser.parse_args()

    with VectorStore.connect() as store:
        if args.delete is not None:
            ok = store.delete_document(args.user, args.delete)
            print("deleted" if ok else "this user has no document with that id")
            return 0 if ok else 1

        if args.paths:
            embedder = Embedder()
            for p in args.paths:
                name = Path(p).name
                try:
                    print(f"{name}: {ingest_file(store, embedder, args.user, Path(p))}")
                except LoaderError as e:
                    print(f"{name}: error: {e}", file=sys.stderr)
                except errors.APIError as e:
                    # Nothing was saved for this file (add_document never ran).
                    print(f"{name}: Gemini error ({e.code}): {e.status}. Run again later.", file=sys.stderr)

        if args.list or not args.paths:
            docs = store.list_documents(args.user)
            print(f"\nDocuments of {args.user}: {len(docs)}")
            for d in docs:
                print(f"  id={d.id}  {d.filename}  pages={d.pages}  chunks={d.chunks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
