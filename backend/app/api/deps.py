from collections.abc import Iterator
from functools import cache
from typing import Annotated

import httpx
from fastapi import Cookie, Depends, HTTPException
from google import genai
from google.genai import types

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


KEEPALIVE_SECONDS = 300


@cache
def _clients() -> tuple[genai.Client, Embedder, Reranker | None]:
    # One per process, and idle connections kept for 5 minutes (httpx closes them after 5s by
    # default): a fresh TLS connection made embedding 1.3-1.5s instead of 0.5s.
    # Without COHERE_API_KEY the reranker is off.
    http = httpx.Client(timeout=60, limits=httpx.Limits(keepalive_expiry=KEEPALIVE_SECONDS))
    client = genai.Client(http_options=types.HttpOptions(httpx_client=http))
    return client, Embedder(client=client), Reranker.from_env()


def get_rag(store: Store) -> RAG:
    client, embedder, reranker = _clients()
    return RAG(store, embedder, client=client, reranker=reranker)


Rag = Annotated[RAG, Depends(get_rag)]
