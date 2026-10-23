from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Response

from app.api.deps import CurrentUser, Store
from app.schemas.conversations import ConversationDetail, ConversationOut, MessageOut

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationOut])
def list_conversations(user_id: CurrentUser, store: Store) -> list[ConversationOut]:
    return [ConversationOut(**asdict(c)) for c in store.list_conversations(user_id)]


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: int, user_id: CurrentUser, store: Store) -> ConversationDetail:
    found = store.get_conversation(user_id, conversation_id)
    # 404 for other users' conversations too, so their ids are not revealed.
    if found is None:
        raise HTTPException(404, "conversation not found")
    conversation, messages = found
    return ConversationDetail(**asdict(conversation), messages=[MessageOut(**asdict(m)) for m in messages])


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: int, user_id: CurrentUser, store: Store) -> Response:
    if not store.delete_conversation(user_id, conversation_id):
        raise HTTPException(404, "conversation not found")
    return Response(status_code=204)
