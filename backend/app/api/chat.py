import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from google.genai import errors

from app.api.deps import CurrentUser, Rag
from app.core.messages import is_no_answer, no_answer
from app.db.store import SearchResult
from app.schemas.chat import ChatRequest, ChatResponse, RetrievedSource, SourceOut
from app.services.rag import cited_numbers

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def _error_info(e: errors.APIError) -> tuple[int, str]:
    log.warning("Gemini error %s: %s", e.code, e.status)
    if e.code == 429:
        return 503, "model quota exceeded, try again in a minute"
    return 502, "model provider error"


def _retrieved(results: list[SearchResult]) -> list[RetrievedSource]:
    return [
        RetrievedSource(number=i, filename=s.filename, page=s.page, score=s.score, content=s.content)
        for i, s in enumerate(results, 1)
    ]


@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest, user_id: CurrentUser, rag: Rag) -> ChatResponse:
    try:
        answer = rag.ask(user_id, body.question, k=body.k)
    except errors.APIError as e:
        raise HTTPException(*_error_info(e)) from e

    return ChatResponse(
        answer=answer.text,
        answered=not is_no_answer(answer.text),
        sources=[
            SourceOut(**s.model_dump(), cited=s.number in answer.cited)
            for s in _retrieved(answer.sources)
        ],
    )


def _event(name: str, data: object) -> str:
    # json.dumps escapes newlines, so each event's data stays on one line as SSE requires.
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse(results: list[SearchResult], pieces: Iterator[str], fallback: str) -> Iterator[str]:
    yield _event("sources", [s.model_dump() for s in _retrieved(results)])
    parts: list[str] = []
    try:
        for piece in pieces:
            parts.append(piece)
            yield _event("delta", {"text": piece})
    except errors.APIError as e:
        # The 200 status is already sent, so the error goes out as an event.
        status, detail = _error_info(e)
        yield _event("error", {"status": status, "detail": detail})
        return

    text = "".join(parts).strip()
    if not text:
        text = fallback
        yield _event("delta", {"text": text})
    yield _event("done", {"answered": not is_no_answer(text), "cited": cited_numbers(text, len(results))})


@router.post("/stream")
def chat_stream(body: ChatRequest, user_id: CurrentUser, rag: Rag) -> StreamingResponse:
    """Server-sent events: `sources`, then `delta` pieces, then `done` (or `error`)."""
    try:
        results, pieces = rag.stream(user_id, body.question, k=body.k)
    except errors.APIError as e:
        raise HTTPException(*_error_info(e)) from e
    return StreamingResponse(
        _sse(results, pieces, no_answer(body.question)),
        media_type="text/event-stream",
        # no-buffering: proxies would otherwise hold the pieces until the end.
        # no-transform: stops proxies (the Next.js dev server too) from gzipping, which buffers the whole stream.
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
