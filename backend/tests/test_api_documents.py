import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.api import documents
from app.api.deps import get_current_user, get_store
from conftest import header_user
from app.main import app
from app.services import ingestion
from app.services.embeddings import DIM


class FakeEmbedder:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def embed_documents(self, texts, on_progress=None):
        if self.fail:
            raise RuntimeError("embedding service down")
        return [[1.0] + [0.0] * (DIM - 1) for _ in texts]


@pytest.fixture
def client(store, monkeypatch):
    embedder = FakeEmbedder()
    # Run the worker's job inline, against the test schema and a fake embedder, instead of queueing it.
    monkeypatch.setattr(
        documents,
        "enqueue_embedding",
        lambda doc_id, user_id, chunks: ingestion.process_document(
            doc_id, user_id, chunks, store=store, embedder=embedder
        ),
    )
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_current_user] = header_user
    # Not used as a context manager, so the lifespan (which touches the real DB) doesn't run.
    c = TestClient(app)
    c.embedder = embedder
    yield c
    app.dependency_overrides.clear()


def _pdf_bytes(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    return doc.tobytes()


def _upload(client, user, name, data):
    return client.post("/documents", headers={"X-User-Id": user}, files={"file": (name, data)})


TEXT = ("This is a sample document about audiobooks and reading habits. " * 3).encode()


def test_upload_txt_becomes_ready(client):
    r = _upload(client, "alice", "notes.txt", TEXT)
    assert r.status_code == 202
    body = r.json()
    assert body["filename"] == "notes.txt"
    assert body["status"] == "processing"

    # TestClient runs background tasks before returning, so it's finished now.
    doc = client.get(f"/documents/{body['id']}", headers={"X-User-Id": "alice"}).json()
    assert doc["status"] == "ready"
    assert doc["chunks"] >= 1
    assert doc["error"] is None


def test_upload_pdf_keeps_pages(client):
    r = _upload(client, "alice", "paper.pdf", _pdf_bytes(["First page text here, long enough.", "Second page text here, long enough."]))
    assert r.status_code == 202
    assert r.json()["pages"] == 2


def test_missing_user_header_rejected(client):
    r = client.post("/documents", files={"file": ("a.txt", TEXT)})
    assert r.status_code == 422


def test_unsupported_extension_rejected(client):
    assert _upload(client, "alice", "a.docx", TEXT).status_code == 415


def test_fake_pdf_rejected(client):
    r = _upload(client, "alice", "evil.pdf", b"MZ\x90\x00 not a pdf")
    assert r.status_code == 415


def test_empty_file_rejected(client):
    assert _upload(client, "alice", "a.txt", b"").status_code == 400


def test_too_large_rejected(client, monkeypatch):
    monkeypatch.setattr(documents, "MAX_UPLOAD_BYTES", 100)
    assert _upload(client, "alice", "a.txt", b"x" * 101).status_code == 413


def test_non_utf8_text_rejected_with_user_filename(client):
    r = _upload(client, "alice", "bad.txt", "سلام".encode("cp1256"))
    assert r.status_code == 422
    assert "bad.txt" in r.json()["detail"]


def test_path_in_filename_is_stripped(client):
    r = _upload(client, "alice", "../../etc/passwd.txt", TEXT)
    assert r.status_code == 202
    assert r.json()["filename"] == "passwd.txt"


def test_duplicate_is_per_user(client):
    first = _upload(client, "alice", "a.txt", TEXT)
    again = _upload(client, "alice", "a-copy.txt", TEXT)
    assert again.status_code == 409
    assert again.json()["detail"]["document_id"] == first.json()["id"]
    assert _upload(client, "bob", "a.txt", TEXT).status_code == 202


def test_users_cannot_see_or_delete_others_documents(client):
    doc_id = _upload(client, "alice", "a.txt", TEXT).json()["id"]
    bob = {"X-User-Id": "bob"}
    assert client.get("/documents", headers=bob).json() == []
    assert client.get(f"/documents/{doc_id}", headers=bob).status_code == 404
    assert client.delete(f"/documents/{doc_id}", headers=bob).status_code == 404
    alice = {"X-User-Id": "alice"}
    assert client.delete(f"/documents/{doc_id}", headers=alice).status_code == 204
    assert client.get(f"/documents/{doc_id}", headers=alice).status_code == 404


def test_embedding_failure_marks_document_failed(client):
    client.embedder.fail = True
    doc_id = _upload(client, "alice", "a.txt", TEXT).json()["id"]
    doc = client.get(f"/documents/{doc_id}", headers={"X-User-Id": "alice"}).json()
    assert doc["status"] == "failed"
    assert "embedding service down" in doc["error"]
    assert doc["chunks"] == 0


def test_queue_down_returns_503_and_leaves_no_document(client, monkeypatch):
    from kombu.exceptions import OperationalError

    def down(*args):
        raise OperationalError("connection refused")

    monkeypatch.setattr(documents, "enqueue_embedding", down)
    r = _upload(client, "alice", "a.txt", TEXT)
    assert r.status_code == 503
    assert "queue" in r.json()["detail"]
    assert client.get("/documents", headers={"X-User-Id": "alice"}).json() == []
    # The same file can be uploaded again once the queue is back.
    monkeypatch.setattr(documents, "enqueue_embedding", lambda *args: None)
    assert _upload(client, "alice", "a.txt", TEXT).status_code == 202


def test_chunks_survive_the_trip_through_json():
    import json
    from dataclasses import asdict

    from app.services.chunking import Chunk

    c = Chunk(text="فیدیبو آرت", source="fidibo.pdf", page=14, index=3)
    assert Chunk(**json.loads(json.dumps(asdict(c), ensure_ascii=False))) == c


def test_upload_reports_total_chunks_and_starts_queued(client, monkeypatch):
    monkeypatch.setattr(documents, "enqueue_embedding", lambda *args: None)
    body = _upload(client, "alice", "notes.txt", TEXT).json()
    assert body["total_chunks"] >= 1
    assert body["embedded_chunks"] is None
