import json

import httpx
import pytest

from app.db.store import SearchResult
from app.services.rerank import Reranker, RerankError


def _r(text):
    return SearchResult(filename="a.pdf", page=1, chunk_index=0, content=text, score=0.5)


def _reranker(handler):
    return Reranker("k", http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_rerank_reorders_and_uses_relevance_score():
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content), auth=request.headers["Authorization"])
        reply = {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]}
        return httpx.Response(200, json=reply)

    out = _reranker(handler).rerank("q", [_r("x"), _r("y")])
    assert [(r.content, r.score) for r in out] == [("y", 0.9), ("x", 0.2)]
    assert sent["documents"] == ["x", "y"] and sent["auth"] == "Bearer k"


def _timeout(request):
    raise httpx.ReadTimeout("slow", request=request)


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(429, json={"message": "too many"}),
        lambda request: httpx.Response(200, json={"unexpected": True}),
        _timeout,
    ],
    ids=["429", "malformed", "timeout"],
)
def test_rerank_errors_become_rerank_error(handler):
    with pytest.raises(RerankError):
        _reranker(handler).rerank("q", [_r("x")])


def test_no_key_means_no_reranker(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    assert Reranker.from_env() is None
