import pytest

from app.cli.evaluate import best_rank, rank_of, summarize
from app.db.store import SearchResult


def _r(name: str, page: int) -> SearchResult:
    return SearchResult(filename=name, page=page, chunk_index=0, content="", score=0.5)


def test_rank_of_matches_file_and_page():
    results = [_r("a.pdf", 3), _r("b.pdf", 2), _r("a.pdf", 2)]
    assert rank_of(results, "a.pdf", 2) == 3
    assert rank_of(results, "b.pdf", 2) == 2
    assert rank_of(results, "a.pdf", 9) is None


def test_summarize_hits_and_mrr():
    m = summarize([1, 2, None, 4], ks=(1, 3, 5))
    assert m["hit@1"] == pytest.approx(0.25)
    assert m["hit@3"] == pytest.approx(0.5)
    assert m["hit@5"] == pytest.approx(0.75)
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)


def test_best_rank_takes_the_highest_of_several_pages():
    results = [_r("a.pdf", 3), _r("a.pdf", 5), _r("a.pdf", 2)]
    assert best_rank(results, "a.pdf", [2, 5]) == 2
    assert best_rank(results, "a.pdf", [9]) is None
    assert best_rank(results, "a.pdf", []) is None
