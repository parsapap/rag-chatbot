import math
from types import SimpleNamespace

import pytest
from google.genai import errors

from app.embeddings import BATCH_SIZE, Embedder, cosine


def _quota_error(delay: str = "3s") -> errors.ClientError:
    return errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "quota",
                "status": "RESOURCE_EXHAUSTED",
                "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}],
            }
        },
    )


class FakeModels:
    def __init__(self, failures: list[Exception] | None = None):
        self.calls: list[dict] = []
        self._failures = list(failures or [])

    def embed_content(self, model, contents, config):
        self.calls.append({"n": len(contents), "task": config.task_type, "dim": config.output_dimensionality})
        if self._failures:
            raise self._failures.pop(0)
        # Unnormalized vectors, like the real API returns below 3072 dims.
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0]) for _ in contents])


def _embedder(models: FakeModels, sleeps: list | None = None) -> Embedder:
    sleeps = sleeps if sleeps is not None else []
    return Embedder(client=SimpleNamespace(models=models), dim=2, sleep=sleeps.append)


def test_vectors_are_unit_length():
    [v] = _embedder(FakeModels()).embed_documents(["a"])
    assert v == pytest.approx([0.6, 0.8])
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0)


def test_batches_never_exceed_api_limit():
    models = FakeModels()
    vectors = _embedder(models).embed_documents(["t"] * (BATCH_SIZE * 2 + 5))
    assert [c["n"] for c in models.calls] == [BATCH_SIZE, BATCH_SIZE, 5]
    assert len(vectors) == BATCH_SIZE * 2 + 5


def test_document_and_query_task_types():
    models = FakeModels()
    e = _embedder(models)
    e.embed_documents(["doc"])
    e.embed_query("q")
    assert [c["task"] for c in models.calls] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
    assert all(c["dim"] == 2 for c in models.calls)


def test_quota_error_waits_server_delay_then_retries():
    sleeps: list[float] = []
    models = FakeModels(failures=[_quota_error("3s"), _quota_error("7s")])
    vectors = _embedder(models, sleeps).embed_documents(["a", "b"])
    assert sleeps == [4.0, 8.0]
    assert len(models.calls) == 3
    assert len(vectors) == 2


def test_other_errors_are_not_retried():
    bad = errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})
    models = FakeModels(failures=[bad])
    with pytest.raises(errors.ClientError):
        _embedder(models).embed_documents(["a"])
    assert len(models.calls) == 1


def test_cosine():
    assert cosine([0.6, 0.8], [0.6, 0.8]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
