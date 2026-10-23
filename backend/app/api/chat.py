import json
import logging
from collections.abc import Callable, Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from google.genai import errors

from app.api.deps import CurrentUser, Rag, Store
from app.core.messages import is_no_answer, no_answer
from app.db.store import SearchResult, VectorStore
from app.schemas.chat import ChatRequest, ChatResponse, RetrievedSource, SourceOut
from app.services.rag import cited_numbers

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Saves (answer text, answered, cited numbers) and returns the conversation id.
Save = Callable[[str, bool, list[int]], int]


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


def _sources_out(results: list[SearchResult], cited: list[int]) -> list[SourceOut]:
    return [SourceOut(**s.model_dump(), cited=s.number in cited) for s in _retrieved(results)]


def _check_conversation(store: VectorStore, user_id: str, body: ChatRequest) -> None:
    # Checked before the model call, so a wrong id costs no quota.
    if body.conversation_id is not None and not store.conversation_exists(user_id, body.conversation_id):
        raise HTTPException(404, "conversation not found")


def _saver(store: VectorStore, user_id: str, body: ChatRequest, results: list[SearchResult]) -> Save:
    def save(text: str, answered: bool, cited: list[int]) -> int:
        sources = [s.model_dump() for s in _sources_out(results, cited)]
        return store.save_turn(user_id, body.conversation_id, body.question, text, answered, sources)

    return save


@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest, user_id: CurrentUser, store: Store, rag: Rag) -> ChatResponse:
    _check_conversation(store, user_id, body)
    try:
        answer = rag.ask(user_id, body.question, k=body.k)
    except errors.APIError as e:
        raise HTTPException(*_error_info(e)) from e

    answered = not is_no_answer(answer.text)
    try:
        conversation_id = _saver(store, user_id, body, answer.sources)(answer.text, answered, answer.cited)
    except LookupError as e:
        # Deleted while the model was answering.
        raise HTTPException(404, "conversation not found") from e
    return ChatResponse(
        answer=answer.text,
        answered=answered,
        sources=_sources_out(answer.sources, answer.cited),
        conversation_id=conversation_id,
    )


def _event(name: str, data: object) -> str:
    # json.dumps escapes newlines, so each event's data stays on one line as SSE requires.
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse(results: list[SearchResult], pieces: Iterator[str], fallback: str, save: Save) -> Iterator[str]:
    yield _event("sources", [s.model_dump() for s in _retrieved(results)])
    parts: list[str] = []
    try:
        for piece in pieces:
            parts.append(piece)
            yield _event("delta", {"text": piece})
    except errors.APIError as e:
        # The 200 status is already sent, so the error goes out as an event. Nothing is saved.
        status, detail = _error_info(e)
        yield _event("error", {"status": status, "detail": detail})
        return

    text = "".join(parts).strip()
    if not text:
        text = fallback
        yield _event("delta", {"text": text})
    answered = not is_no_answer(text)
    cited = cited_numbers(text, len(results))
    try:
        conversation_id = save(text, answered, cited)
    except LookupError:
        # Deleted while the answer was streaming.
        yield _event("error", {"status": 404, "detail": "conversation not found"})
        return
    yield _event("done", {"answered": answered, "cited": cited, "conversation_id": conversation_id})


@router.post("/stream")
def chat_stream(body: ChatRequest, user_id: CurrentUser, store: Store, rag: Rag) -> StreamingResponse:
    """Server-sent events: `sources`, then `delta` pieces, then `done` (or `error`)."""
    _check_conversation(store, user_id, body)
    try:
        results, pieces = rag.stream(user_id, body.question, k=body.k)
    except errors.APIError as e:
        raise HTTPException(*_error_info(e)) from e
    return StreamingResponse(
        _sse(results, pieces, no_answer(body.question), _saver(store, user_id, body, results)),
        media_type="text/event-stream",
        # no-buffering: proxies would otherwise hold the pieces until the end.
        # no-transform: stops proxies (the Next.js dev server too) from gzipping, which buffers the whole stream.
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
