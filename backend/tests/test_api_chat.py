import json
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

    def generate_content_stream(self, model, contents, config):
        # Like the real SDK: a generator, so nothing happens until it is consumed.
        self.calls += 1
        if self.error:
            raise self.error
        for word in self.reply.split(" "):
            yield SimpleNamespace(text=word + " ")
        yield SimpleNamespace(text=None)


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
    assert body == {"answer": NO_ANSWER["fa"], "answered": False, "sources": []}
    assert client.models.calls == 0


def test_model_no_answer_is_flagged(client):
    client.models.reply = NO_ANSWER["fa"]
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


def _events(client, user, body):
    r = client.post("/chat/stream", headers={"X-User-Id": user}, json=body)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    out = []
    for block in r.text.strip().split("\n\n"):
        name, data = block.split("\n")
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_stream_sends_sources_then_deltas_then_done(client):
    events = _events(client, "alice", {"question": "فیدیبو آرت چیست؟", "k": 2})
    names = [n for n, _ in events]
    assert names[0] == "sources"
    assert names[-1] == "done"
    assert set(names[1:-1]) == {"delta"}
    assert len(names) > 3

    sources = events[0][1]
    assert [(s["number"], s["page"]) for s in sources] == [(1, 14), (2, 1)]
    assert "cited" not in sources[0]

    text = "".join(d["text"] for n, d in events if n == "delta")
    assert text.strip() == client.models.reply
    assert events[-1][1] == {"answered": True, "cited": [1]}


def test_stream_user_without_documents(client):
    events = _events(client, "bob", {"question": "q"})
    assert events == [
        ("sources", []),
        ("delta", {"text": NO_ANSWER["en"]}),
        ("done", {"answered": False, "cited": []}),
    ]
    assert client.models.calls == 0


def test_stream_model_error_is_sent_as_event(client):
    client.models.error = errors.ClientError(
        429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}
    )
    events = _events(client, "alice", {"question": "فیدیبو آرت چیست؟"})
    assert events[0][0] == "sources"
    assert events[-1] == ("error", {"status": 503, "detail": "model quota exceeded, try again in a minute"})


def test_stream_invalid_request_rejected(client):
    r = client.post("/chat/stream", headers={"X-User-Id": "alice"}, json={"question": " "})
    assert r.status_code == 422


def test_english_question_gets_english_no_answer(client):
    body = _ask(client, "bob", {"question": "What is Fidibo Art?"}).json()
    assert body["answer"] == NO_ANSWER["en"]
    assert body["answered"] is False


def test_model_answering_in_the_other_language_is_still_flagged(client):
    client.models.reply = NO_ANSWER["en"]
    body = _ask(client, "alice", {"question": "پایتخت فرانسه کجاست؟"}).json()
    assert body["answered"] is False
