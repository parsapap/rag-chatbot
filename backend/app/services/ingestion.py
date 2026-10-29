import logging

from google.genai import errors

from app.db.store import VectorStore
from app.services.chunking import Chunk
from app.services.embeddings import Embedder

log = logging.getLogger(__name__)

MAX_ERROR_CHARS = 500
# Quota, overload and server errors pass with time; anything else will fail again the same way.
TRANSIENT_CODES = {429, 500, 502, 503, 504}


def is_transient(e: Exception) -> bool:
    return isinstance(e, errors.APIError) and e.code in TRANSIENT_CODES


def process_document(
    document_id: int,
    user_id: str,
    chunks: list[Chunk],
    *,
    store: VectorStore,
    embedder: Embedder,
    can_retry: bool = False,
) -> None:
    """Embeds and saves the chunks of a document that is still processing.

    On failure the error is recorded on the document, except a transient error when the
    caller can retry: that one is raised so the caller can try again later.
    """
    doc = store.get_document(user_id, document_id)
    # Deleted meanwhile, or already finished by an earlier delivery of the same queue message.
    if doc is None or doc.status != "processing":
        return
    try:
        # 0 marks "started": the panel switches from queued to a percentage. A retry starts over at 0.
        store.set_progress(document_id, 0)
        vectors = embedder.embed_documents(
            [c.text for c in chunks], on_progress=lambda n: store.set_progress(document_id, n)
        )
        store.complete_document(document_id, user_id, chunks, vectors)
    except Exception as e:
        if can_retry and is_transient(e):
            raise
        # Broad on purpose: otherwise the document would stay "processing" forever. Record why instead.
        log.exception("processing document %s failed", document_id)
        store.fail_document(document_id, f"{type(e).__name__}: {e}"[:MAX_ERROR_CHARS])
