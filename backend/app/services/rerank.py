import os
from dataclasses import replace

import httpx

from app.db.store import SearchResult

URL = "https://api.cohere.com/v2/rerank"
MODEL = os.environ.get("COHERE_RERANK_MODEL", "rerank-v3.5")
# The answer waits on this call. Measured median is about 1.7s; past this, the fused order is used.
TIMEOUT_SECONDS = 5
# A reused connection skips the TLS handshake: median 2.4s -> 1.7s.
KEEPALIVE_SECONDS = 300


class RerankError(Exception):
    pass


class Reranker:
    """Cohere's cross-encoder: reads the question and each chunk together, so it ranks better than vectors alone."""

    def __init__(self, api_key: str, model: str = MODEL, http: httpx.Client | None = None):
        self.model = model
        self._http = http or httpx.Client(
            timeout=TIMEOUT_SECONDS, limits=httpx.Limits(keepalive_expiry=KEEPALIVE_SECONDS)
        )
        self._headers = {"Authorization": f"Bearer {api_key}"}

    @classmethod
    def from_env(cls) -> "Reranker | None":
        key = os.environ.get("COHERE_API_KEY")
        return cls(key) if key else None

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """The same results, best first, with Cohere's 0..1 relevance as the score."""
        if not results:
            return []
        body = {"model": self.model, "query": query, "documents": [r.content for r in results], "top_n": len(results)}
        try:
            response = self._http.post(URL, json=body, headers=self._headers)
            response.raise_for_status()
            ranked = response.json()["results"]
            return [replace(results[x["index"]], score=x["relevance_score"]) for x in ranked]
        # httpx.HTTPError covers 429/5xx, timeouts and dropped connections; the rest is a malformed reply.
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
            raise RerankError(str(e)) from e
