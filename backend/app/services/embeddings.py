import math
import os
import re
import time
from collections.abc import Callable

from google import genai
from google.genai import errors, types

MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
DIM = 768
# The free tier counts each text, not each request (100 texts/minute), so small batches cost
# no extra quota and let upload progress move in small steps.
BATCH_SIZE = 10
MAX_RETRIES = 5
# Without a server hint, wait out the whole per-minute window.
DEFAULT_RETRY_SECONDS = 61.0


class Embedder:
    def __init__(
        self,
        client: genai.Client | None = None,
        model: str = MODEL,
        dim: int = DIM,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._client = client or genai.Client()
        self.model = model
        self.dim = dim
        self._sleep = sleep

    def embed_documents(
        self, texts: list[str], on_progress: Callable[[int], None] | None = None
    ) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT", on_progress)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_QUERY")

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def _embed(
        self, texts: list[str], task_type: str, on_progress: Callable[[int], None] | None = None
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            vectors.extend(self._call(texts[i : i + BATCH_SIZE], task_type))
            if on_progress:
                on_progress(len(vectors))
        return vectors

    def _call(self, batch: list[str], task_type: str) -> list[list[float]]:
        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        for attempt in range(MAX_RETRIES):
            try:
                result = self._client.models.embed_content(
                    model=self.model, contents=batch, config=config
                )
                return [_unit(e.values) for e in result.embeddings]
            except errors.APIError as e:
                # 429 = per-minute quota; every other error is a real failure.
                if e.code != 429 or attempt == MAX_RETRIES - 1:
                    raise
                delay = _retry_delay(e)
                print(f"quota exceeded, waiting {delay:.0f}s…", flush=True)
                self._sleep(delay)
        raise AssertionError("unreachable")


def cosine(a: list[float], b: list[float]) -> float:
    # Vectors from Embedder are unit length, so the dot product is the cosine.
    return sum(x * y for x, y in zip(a, b))


def _unit(values: list[float]) -> list[float]:
    # Below 3072 dims gemini-embedding-001 returns vectors that are not unit length.
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values] if norm else list(values)


def _retry_delay(e: errors.APIError) -> float:
    details = (e.details or {}).get("error", {}).get("details", [])
    for d in details:
        m = re.fullmatch(r"([\d.]+)s", str(d.get("retryDelay", "")))
        if m:
            return float(m.group(1)) + 1
    return DEFAULT_RETRY_SECONDS
