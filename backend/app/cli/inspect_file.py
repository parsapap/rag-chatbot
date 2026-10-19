import argparse
import sys

from app.services.loaders import LoaderError, load


def main() -> int:
    parser = argparse.ArgumentParser(description="Show the text extracted from a file")
    parser.add_argument("path")
    parser.add_argument("--chars", type=int, default=300, help="preview length per page")
    args = parser.parse_args()

    try:
        pages = load(args.path)
    except LoaderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    empty = [p.page for p in pages if not p.text]
    total_chars = sum(len(p.text) for p in pages)

    print(f"file: {pages[0].source if pages else args.path}")
    print(f"pages: {len(pages)}  |  chars: {total_chars}  |  empty pages: {len(empty)}")
    if pages and len(empty) == len(pages):
        print("warning: no text found. The PDF is probably scanned (images only).")
    elif empty:
        print(f"warning: pages without text: {empty}")

    for p in pages:
        if not p.text:
            continue
        more = "…" if len(p.text) > args.chars else ""
        print(f"\n--- page {p.page} ({len(p.text)} chars) ---")
        print(p.text[: args.chars] + more)

    return 0


if __name__ == "__main__":
    sys.exit(main())
