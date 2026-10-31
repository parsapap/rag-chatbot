import json
import os
import urllib.request
from dataclasses import replace

from app.db.store import SearchResult

URL = "https://api.cohere.com/v2/rerank"
MODEL = os.environ.get("COHERE_RERANK_MODEL", "rerank-v3.5")
# The answer waits on this call, so a slow reranker must not hold it up for long.
TIMEOUT_SECONDS = 10


class RerankError(Exception):
    pass


class Reranker:
    """Cohere's cross-encoder: reads the question and each chunk together, so it ranks better than vectors alone."""

    def __init__(self, api_key: str, model: str = MODEL, timeout: float = TIMEOUT_SECONDS):
        self._api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "Reranker | None":
        key = os.environ.get("COHERE_API_KEY")
        return cls(key) if key else None

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """The same results, best first, with Cohere's 0..1 relevance as the score."""
        if not results:
            return []
        body = {"model": self.model, "query": query, "documents": [r.content for r in results], "top_n": len(results)}
        request = urllib.request.Request(
            URL,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                ranked = json.load(response)["results"]
            return [replace(results[x["index"]], score=x["relevance_score"]) for x in ranked]
        # OSError covers HTTP errors (429, 5xx), timeouts and dropped connections; the rest is a malformed reply.
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            raise RerankError(str(e)) from e
