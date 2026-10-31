import io
import json
import urllib.error

import pytest

from app.db.store import SearchResult
from app.services import rerank
from app.services.rerank import Reranker, RerankError


def _r(text):
    return SearchResult(filename="a.pdf", page=1, chunk_index=0, content=text, score=0.5)


def test_rerank_reorders_and_uses_relevance_score(monkeypatch):
    sent = {}

    def fake_urlopen(request, timeout):
        sent.update(json.loads(request.data), auth=request.get_header("Authorization"))
        reply = {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]}
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(rerank.urllib.request, "urlopen", fake_urlopen)
    out = Reranker("k").rerank("q", [_r("x"), _r("y")])
    assert [(r.content, r.score) for r in out] == [("y", 0.9), ("x", 0.2)]
    assert sent["documents"] == ["x", "y"] and sent["auth"] == "Bearer k"


def test_rerank_errors_become_rerank_error(monkeypatch):
    def fail(request, timeout):
        raise urllib.error.HTTPError(rerank.URL, 429, "too many", {}, None)

    monkeypatch.setattr(rerank.urllib.request, "urlopen", fail)
    with pytest.raises(RerankError):
        Reranker("k").rerank("q", [_r("x")])


def test_no_key_means_no_reranker(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    assert Reranker.from_env() is None
