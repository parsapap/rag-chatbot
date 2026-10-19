import logging

from app.db.store import VectorStore
from app.services.chunking import Chunk
from app.services.embeddings import Embedder

log = logging.getLogger(__name__)

MAX_ERROR_CHARS = 500


def process_document(
    document_id: int,
    user_id: str,
    chunks: list[Chunk],
    *,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
) -> None:
    # Runs as a background task after the upload request has returned, so it
    # opens its own DB connection instead of reusing the request's.
    own_store = store is None
    store = store or VectorStore.connect(create_schema=False)
    try:
        vectors = (embedder or Embedder()).embed_documents([c.text for c in chunks])
        store.complete_document(document_id, user_id, chunks, vectors)
    except Exception as e:
        # Broad on purpose: an exception in a background task reaches nobody,
        # and the document would stay "processing" forever. Record why instead.
        # If the document was deleted meanwhile, this updates nothing.
        log.exception("processing document %s failed", document_id)
        store.fail_document(document_id, f"{type(e).__name__}: {e}"[:MAX_ERROR_CHARS])
    finally:
        if own_store:
            store.close()
