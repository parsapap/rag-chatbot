from collections.abc import Iterator
from typing import Annotated

from fastapi import Cookie, Depends, HTTPException

from app.core.security import read_token
from app.db.store import VectorStore
from app.services.embeddings import Embedder
from app.services.rag import RAG
from app.services.rerank import Reranker


def get_current_user(session: Annotated[str | None, Cookie()] = None) -> str:
    # Returned as text because every table keys rows by a TEXT user_id.
    user_id = read_token(session) if session else None
    if user_id is None:
        raise HTTPException(status_code=401, detail="Log in first.")
    return str(user_id)


def get_store() -> Iterator[VectorStore]:
    with VectorStore.connect(create_schema=False) as store:
        yield store


CurrentUser = Annotated[str, Depends(get_current_user)]
# Request scope: the connection stays open until the response ends, so a streamed answer can still be saved.
Store = Annotated[VectorStore, Depends(get_store, scope="request")]


def get_rag(store: Store) -> RAG:
    # Without COHERE_API_KEY the reranker is off and the fused order is used.
    return RAG(store, Embedder(), reranker=Reranker.from_env())


Rag = Annotated[RAG, Depends(get_rag)]
