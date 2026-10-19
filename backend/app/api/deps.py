import re
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException

from app.db.store import VectorStore

_USER_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def get_current_user(x_user_id: Annotated[str, Header()]) -> str:
    # TEMPORARY until real auth: trusts whichever user the client claims to be.
    # Only safe while the API is reachable from this machine alone.
    if not _USER_ID.fullmatch(x_user_id):
        raise HTTPException(status_code=400, detail="invalid X-User-Id")
    return x_user_id


def get_store() -> Iterator[VectorStore]:
    with VectorStore.connect(create_schema=False) as store:
        yield store


CurrentUser = Annotated[str, Depends(get_current_user)]
Store = Annotated[VectorStore, Depends(get_store)]
