from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import auth, chat, conversations, documents
from app.api.deps import Store
from app.db.store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Creates or migrates the schema before the first request.
    VectorStore.connect().close()
    yield


app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(conversations.router)


@app.get("/health")
def health(store: Store) -> dict[str, str]:
    store.conn.execute("SELECT 1")
    return {"status": "ok"}
