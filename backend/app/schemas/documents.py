from pydantic import BaseModel

from app.db.store import Status


class DocumentOut(BaseModel):
    id: int
    filename: str
    pages: int
    chunks: int
    status: Status
    error: str | None
