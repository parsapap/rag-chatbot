import pytest

from app.chunking import MIN_CHUNK_CHARS, chunk_pages, split_text
from app.loaders import Page


def _paragraphs(n: int) -> str:
    return "\n\n".join(f"پاراگراف شماره {i} " + "متن نمونه " * 15 for i in range(n))


def test_short_text_is_one_chunk():
    assert split_text("یک متن کوتاه.", size=100, overlap=10) == ["یک متن کوتاه."]


def test_chunks_respect_size():
    chunks = split_text(_paragraphs(20), size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_no_content_lost():
    text = _paragraphs(20)
    chunks = split_text(text, size=300, overlap=50)
    for i in range(20):
        assert any(f"پاراگراف شماره {i} " in c for c in chunks)


def test_consecutive_chunks_overlap():
    words = " ".join(f"w{i}" for i in range(400))
    chunks = split_text(words, size=200, overlap=50)
    for a, b in zip(chunks, chunks[1:]):
        assert b.split()[0] in a.split()


def test_words_not_cut_when_spaces_exist():
    words = " ".join(f"کلمه{i}" for i in range(300))
    valid = set(words.split())
    for c in split_text(words, size=120, overlap=20):
        assert set(c.split()) <= valid


def test_hard_cut_when_no_separator():
    chunks = split_text("x" * 1000, size=300, overlap=0)
    assert [len(c) for c in chunks] == [300, 300, 300, 100]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("abc", size=100, overlap=100)


def test_chunks_keep_page_and_never_cross_pages():
    pages = [
        Page(text="صفحه اول " * 50, source="a.pdf", page=1),
        Page(text="صفحه دوم " * 50, source="a.pdf", page=2),
    ]
    chunks = chunk_pages(pages, size=200, overlap=30)
    assert {c.page for c in chunks} == {1, 2}
    assert all("دوم" not in c.text for c in chunks if c.page == 1)
    assert all("اول" not in c.text for c in chunks if c.page == 2)


def test_index_runs_across_pages_and_ids_unique():
    pages = [Page(text="متن " * 100, source="a.pdf", page=p) for p in (1, 2)]
    chunks = chunk_pages(pages, size=150, overlap=20)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks[0].id == "a.pdf:p1:c0"


def test_tiny_and_empty_pages_skipped():
    pages = [
        Page(text="351", source="a.pdf", page=1),
        Page(text="", source="a.pdf", page=2),
        Page(text="متن واقعی " * 10, source="a.pdf", page=3),
    ]
    chunks = chunk_pages(pages)
    assert [c.page for c in chunks] == [3]
    assert all(len(c.text) >= MIN_CHUNK_CHARS for c in chunks)


def test_overlap_not_empty_when_last_piece_is_long():
    # Each line is longer than the overlap; the tail must still carry words.
    line = " ".join(f"k{i}" for i in range(40))
    text = "\n".join(f"{line} L{n}" for n in range(6))
    chunks = split_text(text, size=300, overlap=60)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        assert b.split()[0] in a.split()
