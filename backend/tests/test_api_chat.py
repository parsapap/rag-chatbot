from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from google.genai import errors

from app.api.deps import get_rag
from app.core.messages import NO_ANSWER
from app.main import app
from app.services.chunking import Chunk
from app.services.embeddings import DIM
from app.services.rag import RAG


def _vec(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


class FakeEmbedder:
    def embed_query(self, text):
        return _vec(0)


class FakeModels:
    def __init__(self):
        self.calls = 0
        self.reply = "فیدیبو آرت از سال 1403 شروع شد [1]."
        self.error: Exception | None = None

    def generate_content(self, model, contents, config):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.reply)


@pytest.fixture
def client(store):
    store.add_document(
        "alice",
        "fidibo.pdf",
        "hash-1",
        16,
        [
            Chunk(text="فیدیبو آرت در سال 1403 متولد شد", source="fidibo.pdf", page=14, index=0),
            Chunk(text="فیدیبو کتاب صوتی دارد", source="fidibo.pdf", page=1, index=1),
        ],
        [_vec(0), _vec(1)],
    )
    models = FakeModels()
    app.dependency_overrides[get_rag] = lambda: RAG(
        store, FakeEmbedder(), client=SimpleNamespace(models=models)
    )
    c = TestClient(app)
    c.models = models
    yield c
    app.dependency_overrides.clear()


def _ask(client, user, body):
    return client.post("/chat", headers={"X-User-Id": user}, json=body)


def test_answer_with_numbered_sources(client):
    r = _ask(client, "alice", {"question": "فیدیبو آرت چیست؟", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == client.models.reply
    assert body["answered"] is True
    first, second = body["sources"]
    assert (first["number"], first["filename"], first["page"], first["cited"]) == (1, "fidibo.pdf", 14, True)
    assert first["score"] == pytest.approx(1.0)
    assert (second["number"], second["page"], second["cited"]) == (2, 1, False)


def test_user_without_documents_gets_no_answer_and_no_llm_call(client):
    body = _ask(client, "bob", {"question": "فیدیبو آرت چیست؟"}).json()
    assert body == {"answer": NO_ANSWER, "answered": False, "sources": []}
    assert client.models.calls == 0


def test_model_no_answer_is_flagged(client):
    client.models.reply = NO_ANSWER
    body = _ask(client, "alice", {"question": "پایتخت فرانسه کجاست؟"}).json()
    assert body["answered"] is False
    assert not any(s["cited"] for s in body["sources"])


@pytest.mark.parametrize(
    "body",
    [{"question": "   "}, {"question": "x" * 2001}, {"question": "q", "k": 0}, {"question": "q", "k": 50}, {}],
)
def test_invalid_request_rejected(client, body):
    assert _ask(client, "alice", body).status_code == 422


def test_missing_user_header_rejected(client):
    assert client.post("/chat", json={"question": "q"}).status_code == 422


def test_quota_error_becomes_503(client):
    client.models.error = errors.ClientError(
        429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}
    )
    r = _ask(client, "alice", {"question": "فیدیبو آرت چیست؟"})
    assert r.status_code == 503
    assert "quota" in r.json()["detail"]


def test_other_model_error_becomes_502(client):
    client.models.error = errors.ClientError(
        400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}}
    )
    assert _ask(client, "alice", {"question": "فیدیبو آرت چیست؟"}).status_code == 502
