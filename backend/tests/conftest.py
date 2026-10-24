import os
import uuid
from typing import Annotated

import psycopg
import pytest
from fastapi import Header

from app.db.store import VectorStore


def header_user(x_user_id: Annotated[str, Header()]) -> str:
    # Stands in for the session cookie in API tests; test_api_auth.py covers the real cookie.
    return x_user_id


@pytest.fixture
def store():
    # Real Postgres, but each test gets its own throwaway schema.
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    schema = f"test_{uuid.uuid4().hex[:8]}"
    conn.execute(f'CREATE SCHEMA "{schema}"')
    conn.execute(f'SET search_path TO "{schema}", public')
    s = VectorStore(conn)
    s.init_schema()
    s.configure_session()
    yield s
    conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
    conn.close()
