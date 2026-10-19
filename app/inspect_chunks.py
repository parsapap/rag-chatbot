import argparse
import statistics
import sys

from app.chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_pages
from app.loaders import LoaderError, load


def main() -> int:
    parser = argparse.ArgumentParser(description="Show the chunks of one or more files")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--show", type=int, default=3, help="chunks to print per file")
    args = parser.parse_args()

    for path in args.paths:
        try:
            pages = load(path)
        except LoaderError as e:
            print(f"error: {e}", file=sys.stderr)
            continue

        chunks = chunk_pages(pages, args.size, args.overlap)
        if not chunks:
            print(f"\n=== {path}: no chunks ===")
            continue

        lens = [len(c.text) for c in chunks]
        print(f"\n=== {chunks[0].source} ===")
        print(
            f"pages: {len(pages)}  |  chunks: {len(chunks)}  |  "
            f"length: min={min(lens)} median={int(statistics.median(lens))} max={max(lens)}"
        )
        for c in chunks[: args.show]:
            print(f"\n--- {c.id} ({len(c.text)} chars) ---")
            print(c.text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
