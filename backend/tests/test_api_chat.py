import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from google.genai import errors

from app.api.deps import get_current_user, get_rag, get_store
from conftest import header_user
from app.core.messages import NO_ANSWER
from app.main import app
from app.services.chunking import Chunk
from app.services.embeddings import DIM
from app.services.rag import REWRITE_SYSTEM_PROMPT, RAG


def _vec(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return _vec(0)


class FakeModels:
    def __init__(self):
        self.calls = 0
        self.reply = "فیدیبو آرت از سال 1403 شروع شد [1]."
        self.error: Exception | None = None
        self.rewrite_reply = "فیدیبو آرت از چه سالی شروع شد؟"
        self.rewrite_error: Exception | None = None
        self.rewrites = 0
        self.prompts: list[str] = []

    def generate_content(self, model, contents, config):
        if config.system_instruction == REWRITE_SYSTEM_PROMPT:
            self.rewrites += 1
            if self.rewrite_error:
                raise self.rewrite_error
            return SimpleNamespace(text=self.rewrite_reply)
        self.prompts.append(contents)
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
    embedder = FakeEmbedder()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_current_user] = header_user
    app.dependency_overrides[get_rag] = lambda: RAG(
        store, embedder, client=SimpleNamespace(models=models)
    )
    c = TestClient(app)
    c.models = models
    c.embedder = embedder
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
    assert (body["answer"], body["answered"], body["sources"]) == (NO_ANSWER["fa"], False, [])
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
    done = events[-1][1]
    assert (done["answered"], done["cited"]) == (True, [1])
    assert isinstance(done["conversation_id"], int)


def test_stream_user_without_documents(client):
    events = _events(client, "bob", {"question": "q"})
    assert events[:2] == [("sources", []), ("delta", {"text": NO_ANSWER["en"]})]
    assert events[2][0] == "done"
    assert (events[2][1]["answered"], events[2][1]["cited"]) == (False, [])
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


def _conversations(client, user):
    return client.get("/conversations", headers={"X-User-Id": user}).json()


def test_first_question_starts_a_conversation_and_follow_up_joins_it(client):
    first = _ask(client, "alice", {"question": "فیدیبو آرت چیست؟", "k": 2}).json()
    cid = first["conversation_id"]
    second = _ask(client, "alice", {"question": "از کی شروع شد؟", "conversation_id": cid}).json()
    assert second["conversation_id"] == cid

    [conv] = _conversations(client, "alice")
    assert (conv["id"], conv["title"]) == (cid, "فیدیبو آرت چیست؟")
    detail = client.get(f"/conversations/{cid}", headers={"X-User-Id": "alice"}).json()
    assert [m["question"] for m in detail["messages"]] == ["فیدیبو آرت چیست؟", "از کی شروع شد؟"]
    msg = detail["messages"][0]
    assert (msg["answer"], msg["answered"]) == (client.models.reply, True)
    assert [(s["number"], s["page"], s["cited"]) for s in msg["sources"]] == [(1, 14, True), (2, 1, False)]


def test_stream_saves_the_finished_answer(client):
    events = _events(client, "alice", {"question": "فیدیبو آرت چیست؟", "k": 2})
    cid = events[-1][1]["conversation_id"]
    detail = client.get(f"/conversations/{cid}", headers={"X-User-Id": "alice"}).json()
    [msg] = detail["messages"]
    assert msg["answer"] == client.models.reply
    assert [s["cited"] for s in msg["sources"]] == [True, False]


def test_failed_answer_is_not_saved(client):
    client.models.error = errors.ClientError(
        429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}
    )
    _ask(client, "alice", {"question": "فیدیبو آرت چیست؟"})
    _events(client, "alice", {"question": "فیدیبو آرت چیست؟"})
    assert _conversations(client, "alice") == []


def test_other_users_cannot_see_join_or_delete_a_conversation(client):
    cid = _ask(client, "alice", {"question": "فیدیبو آرت چیست؟"}).json()["conversation_id"]
    bob = {"X-User-Id": "bob"}
    assert _conversations(client, "bob") == []
    assert client.get(f"/conversations/{cid}", headers=bob).status_code == 404
    assert client.delete(f"/conversations/{cid}", headers=bob).status_code == 404
    r = client.post("/chat", headers=bob, json={"question": "q", "conversation_id": cid})
    assert r.status_code == 404
    r = client.post("/chat/stream", headers=bob, json={"question": "q", "conversation_id": cid})
    assert r.status_code == 404
    assert client.models.calls == 1


def test_delete_removes_conversation_and_messages(client):
    cid = _ask(client, "alice", {"question": "فیدیبو آرت چیست؟"}).json()["conversation_id"]
    alice = {"X-User-Id": "alice"}
    assert client.delete(f"/conversations/{cid}", headers=alice).status_code == 204
    assert client.get(f"/conversations/{cid}", headers=alice).status_code == 404
    assert _conversations(client, "alice") == []


def test_long_question_title_is_cut(client):
    q = "س" * 200
    _ask(client, "alice", {"question": q})
    [conv] = _conversations(client, "alice")
    assert conv["title"] == q[:80]


def test_most_recent_conversation_is_listed_first(client):
    a = _ask(client, "alice", {"question": "اول"}).json()["conversation_id"]
    b = _ask(client, "alice", {"question": "دوم"}).json()["conversation_id"]
    assert [c["id"] for c in _conversations(client, "alice")] == [b, a]
    _ask(client, "alice", {"question": "ادامه", "conversation_id": a})
    assert [c["id"] for c in _conversations(client, "alice")] == [a, b]


QUOTA = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
FOLLOW_UP = "از کی شروع شد؟"


def _first_turn(client):
    return _ask(client, "alice", {"question": "فیدیبو آرت چیست؟"}).json()["conversation_id"]


def test_first_question_is_not_rewritten(client):
    _first_turn(client)
    assert client.models.rewrites == 0
    assert client.embedder.queries == ["فیدیبو آرت چیست؟"]


def test_follow_up_is_searched_with_the_rewritten_question(client):
    cid = _first_turn(client)
    body = _ask(client, "alice", {"question": FOLLOW_UP, "conversation_id": cid}).json()
    assert client.models.rewrites == 1
    assert client.embedder.queries[-1] == client.models.rewrite_reply
    assert body["search_query"] == client.models.rewrite_reply
    assert "Question: " + client.models.rewrite_reply in client.models.prompts[-1]

    detail = client.get(f"/conversations/{cid}", headers={"X-User-Id": "alice"}).json()
    first, second = detail["messages"]
    assert (first["search_query"], second["search_query"]) == (None, client.models.rewrite_reply)
    assert second["question"] == FOLLOW_UP


def test_rewrite_error_falls_back_to_the_typed_question(client):
    cid = _first_turn(client)
    client.models.rewrite_error = QUOTA
    r = _ask(client, "alice", {"question": FOLLOW_UP, "conversation_id": cid})
    assert r.status_code == 200
    assert r.json()["search_query"] is None
    assert client.embedder.queries[-1] == FOLLOW_UP


def test_answer_language_follows_the_typed_question_not_the_rewrite(client):
    cid = _first_turn(client)
    body = {"question": "When did it start?", "conversation_id": cid}
    _ask(client, "alice", body)
    assert "Write the answer in English" in client.models.prompts[-1]
    assert client.models.rewrite_reply in client.models.prompts[-1]


def test_stream_sends_the_rewrite_for_a_follow_up_only(client):
    first = _events(client, "alice", {"question": "فیدیبو آرت چیست؟"})
    assert "rewrite" not in [n for n, _ in first]
    cid = first[-1][1]["conversation_id"]
    events = _events(client, "alice", {"question": FOLLOW_UP, "conversation_id": cid})
    assert [n for n, _ in events][:2] == ["sources", "rewrite"]
    assert events[1][1] == {"search_query": client.models.rewrite_reply}
    assert client.embedder.queries[-1] == client.models.rewrite_reply


def test_other_users_conversation_is_never_rewritten(client):
    cid = _first_turn(client)
    r = client.post("/chat", headers={"X-User-Id": "bob"}, json={"question": FOLLOW_UP, "conversation_id": cid})
    assert r.status_code == 404
    assert client.models.rewrites == 0
