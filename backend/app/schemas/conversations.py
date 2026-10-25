from datetime import datetime

from pydantic import BaseModel

from app.schemas.chat import SourceOut


class ConversationOut(BaseModel):
    id: int
    title: str
    updated_at: datetime


class MessageOut(BaseModel):
    question: str
    answer: str
    answered: bool
    sources: list[SourceOut]
    search_query: str | None = None


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]
