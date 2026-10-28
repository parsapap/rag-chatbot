import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

ZWNJ = "\u200c"


class LoaderError(Exception):
    pass


@dataclass(frozen=True)
class Page:
    text: str
    source: str
    page: int


_CHAR_MAP = str.maketrans(
    {
        "\u064a": "\u06cc",
        "\u0649": "\u06cc",
        "\u0643": "\u06a9",
        **{chr(0x0660 + d): str(d) for d in range(10)},  # Arabic-Indic digits
        **{chr(0x06F0 + d): str(d) for d in range(10)},  # Persian digits
    }
)

# Tatweel, Arabic diacritics, zero-width space, BOM, and bidi control marks.
# ZWNJ (nim-fasele) is intentionally kept: it is part of Persian spelling.
_INVISIBLE = re.compile("[\u0640\u064b-\u0652\u200b\u200e\u200f\u202a-\u202e\ufeff]")
_SPACES = re.compile("[ \t\u00a0]+")
# Persian layouts write 778,910 as 778\u060c910 (Arabic comma) and 1.65 as 1\u066b65.
_THOUSANDS_SEP = re.compile("(?<=\\d)[\u060c\u066c](?=\\d{3}(?!\\d))")
_DECIMAL_SEP = re.compile("(?<=\\d)\u066b(?=\\d)")
_ZWNJ_NEAR_SPACE = re.compile(f" ?{ZWNJ}+ ?")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_CHAR_MAP)
    text = _INVISIBLE.sub("", text)
    text = _THOUSANDS_SEP.sub(",", text)
    text = _DECIMAL_SEP.sub(".", text)
    text = _SPACES.sub(" ", text)
    text = _ZWNJ_NEAR_SPACE.sub(lambda m: " " if " " in m.group() else ZWNJ, text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _EXTRA_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def load(path: str | Path) -> list[Page]:
    path = Path(path)
    if not path.is_file():
        raise LoaderError(f"file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise LoaderError(f"unsupported format: {ext or '(no extension)'}")

    if ext == ".pdf":
        return _load_pdf(path)
    return _load_text(path)


def _load_text(path: Path) -> list[Page]:
    try:
        # utf-8-sig also strips the BOM that Windows editors often add.
        raw = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise LoaderError(f"not UTF-8: {path.name} (byte {e.start})") from e
    return [Page(text=normalize(raw), source=path.name, page=1)]


def _load_pdf(path: Path) -> list[Page]:
    try:
        doc = pymupdf.open(path)
    except RuntimeError as e:  # pymupdf.FileDataError is a RuntimeError
        raise LoaderError(f"corrupt or unreadable PDF: {path.name}") from e

    with doc:
        if doc.needs_pass:
            raise LoaderError(f"encrypted PDF: {path.name}")
        # Empty pages are kept so page numbers stay aligned with the real PDF.
        texts = strip_page_furniture([normalize(_page_text(page)) for page in doc])
        return [
            Page(text=text, source=path.name, page=i)
            for i, text in enumerate(texts, start=1)
        ]


# A running header/footer: a text line near the top or bottom of a page that
# repeats (digits ignored) on at least this share of pages.
_REPEAT_SHARE = 0.3
_MIN_REPEAT_PAGES = 3
_EDGE_LINES = 2
_MIN_HEADER_LETTERS = 10
_DIGITS = re.compile(r"\d+")


def _header_key(line: str) -> str | None:
    # Digits are dropped so "header 350" and "header 352" match. A header must
    # carry a page number: repeated lines without one, like the unit line
    # "(مبالغ به میلیون ریال)", are context the numbers need. Lines with few
    # letters (numbers, "درصد", brackets) are data, never headers.
    if not _DIGITS.search(line):
        return None
    key = _DIGITS.sub("", line).strip()
    return key if sum(c.isalpha() for c in key) >= _MIN_HEADER_LETTERS else None


def _edge_indexes(lines: list[str]) -> set[int]:
    n = len(lines)
    return set(range(min(_EDGE_LINES, n))) | set(range(max(0, n - _EDGE_LINES), n))


def strip_page_furniture(texts: list[str]) -> list[str]:
    if len(texts) < _MIN_REPEAT_PAGES:
        return texts
    pages = [t.split("\n") for t in texts]

    counts: dict[str, int] = {}
    for lines in pages:
        keys = {_header_key(lines[j]) for j in _edge_indexes(lines)}
        for key in keys - {None}:
            counts[key] = counts.get(key, 0) + 1
    threshold = max(_MIN_REPEAT_PAGES, int(len(pages) * _REPEAT_SHARE))
    repeated = {k for k, n in counts.items() if n >= threshold}

    drop = _page_number_lines(pages)
    for i, lines in enumerate(pages):
        for j in _edge_indexes(lines):
            if _header_key(lines[j]) in repeated:
                drop.add((i, j))

    return [
        "\n".join(line for j, line in enumerate(lines) if (i, j) not in drop).strip()
        for i, lines in enumerate(pages)
    ]


def _page_number_lines(pages: list[list[str]]) -> set[tuple[int, int]]:
    # A bare number on a page's first/last line is a printed page number only if
    # (number - page index) is the same for most pages; otherwise it's data.
    candidates: dict[int, list[tuple[int, int]]] = {}
    for i, lines in enumerate(pages):
        for j in {0, len(lines) - 1}:
            if lines and lines[j].isdigit():
                candidates.setdefault(int(lines[j]) - i, []).append((i, j))
    if not candidates:
        return set()
    best = max(candidates.values(), key=len)
    return set(best) if len(best) >= max(_MIN_REPEAT_PAGES, len(pages) // 3) else set()


def _page_text(page: pymupdf.Page) -> str:
    lines: list[tuple[tuple[float, float, float, float], str]] = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            chars = [c for span in line["spans"] for c in span["chars"]]
            lines.append((line["bbox"], _fix_lam_alef(chars)))
    return "\n".join(_merge_table_rows(lines))


# Excel-made PDFs put every table cell on its own line, so "label / 1404 value / 1403 value"
# loses which number belongs to which row. Lines whose vertical middles are this close
# (in points) sit on one visual row.
_ROW_TOLERANCE = 3.0
# A horizontal gap wider than this (in points) between two lines of a row is a cell border.
_CELL_GAP = 8.0
_NUMERIC_CELL = re.compile(r"^[()\s\d,./%\-]*\d[()\s\d,./%\-]*$")
_LETTER = re.compile(r"[^\W\d_]")


def _is_table_row(texts: list[str]) -> bool:
    # A table row: two or more cells, at least one number, and at most one text cell (the
    # row label). Two side-by-side text cells are prose columns, which must stay apart.
    # Cells without letters, like the "(" and ")" around a negative amount, are not text.
    cells = [t.strip() for t in texts if t.strip()]
    numbers = [t for t in cells if _NUMERIC_CELL.match(t)]
    words = [t for t in cells if _LETTER.search(t) and not _NUMERIC_CELL.match(t)]
    return len(cells) >= 2 and len(numbers) >= 1 and len(words) <= 1


def _join_row(cells: list[tuple[tuple[float, float, float, float], str]]) -> str:
    # Right to left: the rightmost cell is read first, as on a Persian page.
    cells = sorted(cells, key=lambda c: -c[0][2])
    out = cells[0][1].strip()
    for prev, cur in zip(cells, cells[1:]):
        gap = prev[0][0] - cur[0][2]
        out += (" | " if gap > _CELL_GAP else " ") + cur[1].strip()
    # A negative amount is drawn as "(", number, ")" in three cells; keep it as "( 365,327 )".
    return re.sub(r"\(\s*\|\s*", "( ", re.sub(r"\s*\|\s*\)", " )", out))


def _merge_table_rows(lines: list[tuple[tuple[float, float, float, float], str]]) -> list[str]:
    middle = [(bbox[1] + bbox[3]) / 2 for bbox, _ in lines]
    rows: list[list[int]] = []
    for i in sorted(range(len(lines)), key=lambda i: middle[i]):
        if rows and abs(middle[rows[-1][0]] - middle[i]) <= _ROW_TOLERANCE:
            rows[-1].append(i)
        else:
            rows.append([i])

    merged: dict[int, str] = {}
    dropped: set[int] = set()
    for row in rows:
        if len(row) > 1 and _is_table_row([lines[i][1] for i in row]):
            first = min(row)
            merged[first] = _join_row([lines[i] for i in row])
            dropped.update(i for i in row if i != first)
    # The merged row takes the place of its first line, so everything else keeps its order.
    return [merged.get(i, text) for i, (_, text) in enumerate(lines) if i not in dropped]


# Word/Excel PDFs draw the lam-alef ligature as one glyph that pymupdf
# extracts as a zero-width alef followed by lam ("طوالنی" instead of "طولانی").
# The zero width tells it apart from a real alef+lam such as "الزامی".
_ZERO_WIDTH = 0.5


def _fix_lam_alef(chars: list[dict]) -> str:
    out = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if (
            c["c"] == "ا"
            and i + 1 < len(chars)
            and chars[i + 1]["c"] == "ل"
            and c["bbox"][2] - c["bbox"][0] < _ZERO_WIDTH
        ):
            out.append("لا")
            i += 2
            continue
        out.append(c["c"])
        i += 1
    return "".join(out)
