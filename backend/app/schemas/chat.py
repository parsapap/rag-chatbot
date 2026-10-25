from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from app.services.rag import TOP_K

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ChatRequest(BaseModel):
    question: Question
    k: int = Field(default=TOP_K, ge=1, le=10)
    # None starts a new conversation.
    conversation_id: int | None = None


class RetrievedSource(BaseModel):
    number: int  # the [n] the answer uses to cite this source
    filename: str
    page: int
    score: float
    content: str


class SourceOut(RetrievedSource):
    cited: bool


class ChatResponse(BaseModel):
    answer: str
    answered: bool  # False when the documents had no answer
    sources: list[SourceOut]
    conversation_id: int
    # The rewritten question that was searched, for a follow-up; None when the typed question was used.
    search_query: str | None = None
