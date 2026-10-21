import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import chat, documents
from app.api.deps import Store
from app.db.store import VectorStore

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with VectorStore.connect() as store:
        if n := store.fail_interrupted():
            log.warning("marked %d interrupted document(s) as failed", n)
    yield


app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health")
def health(store: Store) -> dict[str, str]:
    store.conn.execute("SELECT 1")
    return {"status": "ok"}
