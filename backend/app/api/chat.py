import logging

from fastapi import APIRouter, HTTPException
from google.genai import errors

from app.api.deps import CurrentUser, Rag
from app.core.messages import NO_ANSWER
from app.schemas.chat import ChatRequest, ChatResponse, SourceOut

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest, user_id: CurrentUser, rag: Rag) -> ChatResponse:
    try:
        answer = rag.ask(user_id, body.question, k=body.k)
    except errors.APIError as e:
        log.warning("Gemini error %s: %s", e.code, e.status)
        if e.code == 429:
            raise HTTPException(503, "model quota exceeded, try again in a minute") from e
        raise HTTPException(502, "model provider error") from e

    return ChatResponse(
        answer=answer.text,
        answered=answer.text != NO_ANSWER,
        sources=[
            SourceOut(
                number=i,
                filename=s.filename,
                page=s.page,
                score=s.score,
                content=s.content,
                cited=i in answer.cited,
            )
            for i, s in enumerate(answer.sources, 1)
        ],
    )
