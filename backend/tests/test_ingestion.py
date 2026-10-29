import pytest
from google.genai import errors

from app.services.chunking import Chunk
from app.services.embeddings import DIM
from app.services.ingestion import process_document

QUOTA = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
BAD = errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})
CHUNKS = [Chunk(text="متن نمونه برای تست", source="a.txt", page=1, index=0)]


class Embedder:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def embed_documents(self, texts, on_progress=None):
        self.calls += 1
        if self.error:
            raise self.error
        return [[1.0] + [0.0] * (DIM - 1) for _ in texts]


def _doc(store):
    return store.create_document("u", "a.txt", "hash", 1, len(CHUNKS))


def test_success_marks_ready(store):
    doc_id = _doc(store)
    process_document(doc_id, "u", CHUNKS, store=store, embedder=Embedder())
    assert store.get_document("u", doc_id).status == "ready"


def test_transient_error_is_raised_while_retries_are_left(store):
    doc_id = _doc(store)
    with pytest.raises(errors.ClientError):
        process_document(doc_id, "u", CHUNKS, store=store, embedder=Embedder(QUOTA), can_retry=True)
    assert store.get_document("u", doc_id).status == "processing"


def test_transient_error_on_the_last_try_marks_failed(store):
    doc_id = _doc(store)
    process_document(doc_id, "u", CHUNKS, store=store, embedder=Embedder(QUOTA), can_retry=False)
    doc = store.get_document("u", doc_id)
    assert doc.status == "failed" and "429" in doc.error


def test_permanent_error_marks_failed_even_with_retries_left(store):
    doc_id = _doc(store)
    process_document(doc_id, "u", CHUNKS, store=store, embedder=Embedder(BAD), can_retry=True)
    assert store.get_document("u", doc_id).status == "failed"


def test_redelivered_message_for_a_finished_document_does_nothing(store):
    doc_id = _doc(store)
    process_document(doc_id, "u", CHUNKS, store=store, embedder=Embedder())
    again = Embedder()
    process_document(doc_id, "u", CHUNKS, store=store, embedder=again)
    assert again.calls == 0
    assert store.get_document("u", doc_id).chunks == 1


def test_deleted_document_is_skipped(store):
    doc_id = _doc(store)
    store.delete_document("u", doc_id)
    embedder = Embedder()
    process_document(doc_id, "u", CHUNKS, store=store, embedder=embedder)
    assert embedder.calls == 0


def test_new_document_is_queued_until_the_worker_starts(store):
    doc = store.get_document("u", _doc(store))
    assert (doc.status, doc.total_chunks, doc.embedded_chunks) == ("processing", 1, None)


def test_progress_counts_up_and_ends_at_the_total(store):
    doc_id = _doc(store)
    seen = []

    class Recording(Embedder):
        def embed_documents(self, texts, on_progress=None):
            on_progress(len(texts))
            seen.append(store.get_document("u", doc_id).embedded_chunks)
            return super().embed_documents(texts)

    process_document(doc_id, "u", CHUNKS, store=store, embedder=Recording())
    doc = store.get_document("u", doc_id)
    assert seen == [1]
    assert (doc.status, doc.total_chunks) == ("ready", 1)


def test_progress_does_not_touch_a_finished_document(store):
    doc_id = _doc(store)
    process_document(doc_id, "u", CHUNKS, store=store, embedder=Embedder())
    store.set_progress(doc_id, 0)
    assert store.get_document("u", doc_id).status == "ready"
