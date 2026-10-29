from pydantic import BaseModel

from app.db.store import Status


class DocumentOut(BaseModel):
    id: int
    filename: str
    pages: int
    chunks: int
    status: Status
    error: str | None
    total_chunks: int
    # None while waiting in the queue; counts up while the worker embeds.
    embedded_chunks: int | None
