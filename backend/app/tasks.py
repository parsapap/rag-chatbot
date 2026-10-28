from app.db.store import VectorStore
from app.services.chunking import Chunk
from app.services.embeddings import Embedder
from app.services.ingestion import process_document
from app.worker import celery_app

MAX_RETRIES = 3
# Doubles each time: 60 s, 120 s, 240 s. The embedder already waits out short quota windows on its own.
RETRY_SECONDS = 60


@celery_app.task(bind=True, max_retries=MAX_RETRIES)
def embed_document(self, document_id: int, user_id: str, chunks: list[dict]) -> None:
    with VectorStore.connect(create_schema=False) as store:
        try:
            process_document(
                document_id,
                user_id,
                [Chunk(**c) for c in chunks],
                store=store,
                embedder=Embedder(),
                can_retry=self.request.retries < self.max_retries,
            )
        except Exception as e:
            # process_document only raises a transient error while retries are left.
            raise self.retry(exc=e, countdown=RETRY_SECONDS * 2**self.request.retries) from e
