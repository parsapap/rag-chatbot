import re
from dataclasses import dataclass

from app.services.loaders import Page

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 30

# Tried in order: paragraph, line, sentence, word. Zero-width splits keep the
# separator on the piece, so "".join(pieces) == text and nothing is lost.
_SEPARATORS = [
    re.compile(r"(?<=\n\n)"),
    re.compile(r"(?<=\n)"),
    re.compile(r"(?<=[.!?؟] )"),
    re.compile(r"(?<= )"),
]
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int
    index: int

    @property
    def id(self) -> str:
        return f"{self.source}:p{self.page}:c{self.index}"


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    pieces = _split(text, size, _SEPARATORS)
    return [c for c in (c.strip() for c in _merge(pieces, size, overlap)) if c]


def chunk_pages(
    pages: list[Page], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    # Chunks never cross a page boundary, so every chunk has one page to cite.
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for page in pages:
        for text in split_text(page.text, size, overlap):
            if len(text) < MIN_CHUNK_CHARS:
                continue
            i = counters.get(page.source, 0)
            counters[page.source] = i + 1
            chunks.append(Chunk(text=text, source=page.source, page=page.page, index=i))
    return chunks


def _split(text: str, size: int, seps: list[re.Pattern]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]
    parts = [p for p in seps[0].split(text) if p]
    if len(parts) == 1:
        return _split(text, size, seps[1:])
    out: list[str] = []
    for part in parts:
        out.extend(_split(part, size, seps[1:]))
    return out


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for piece in pieces:
        if cur and cur_len + len(piece) > size:
            chunks.append("".join(cur))
            cur = _overlap_tail(cur, overlap)
            cur_len = sum(len(p) for p in cur)
            while cur and cur_len + len(piece) > size:
                cur_len -= len(cur.pop(0))
        cur.append(piece)
        cur_len += len(piece)
    if cur:
        chunks.append("".join(cur))
    return chunks


def _overlap_tail(cur: list[str], overlap: int) -> list[str]:
    # Whole pieces first; if the next piece is longer than the room left,
    # take only its last whole words so the overlap isn't silently empty.
    tail: list[str] = []
    room = overlap
    for prev in reversed(cur):
        if len(prev) <= room:
            tail.insert(0, prev)
            room -= len(prev)
            continue
        if room > 0:
            cut = prev[-room:]
            boundary = _WHITESPACE.search(cut)
            if boundary and cut[boundary.end() :]:
                tail.insert(0, cut[boundary.end() :])
        break
    return tail
