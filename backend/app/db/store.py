import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import psycopg
from psycopg.types.json import Jsonb

from app.services.chunking import Chunk
from app.services.embeddings import DIM

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    filename    TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    pages       INT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, sha256)
);

-- ADD COLUMN IF NOT EXISTS so databases created before these columns existed get them too.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'ready';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS error TEXT;
-- Progress: embedded_chunks stays NULL while the document waits in the queue.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS total_chunks INT NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedded_chunks INT;

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id      TEXT NOT NULL,
    page         INT NOT NULL,
    chunk_index  INT NOT NULL,
    content      TEXT NOT NULL,
    embedding    vector({DIM}) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_user_id_idx ON chunks (user_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS conversations (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_user_id_idx ON conversations (user_id, updated_at DESC);

-- One row per question and its answer. Sources are a snapshot, so history still
-- reads correctly after the document behind them is deleted.
CREATE TABLE IF NOT EXISTS messages (
    id               BIGSERIAL PRIMARY KEY,
    conversation_id  BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    question         TEXT NOT NULL,
    answer           TEXT NOT NULL,
    answered         BOOLEAN NOT NULL,
    sources          JSONB NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_conversation_id_idx ON messages (conversation_id, id);

-- The rewritten question that was searched, for a follow-up; NULL when the user's own words were used.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS search_query TEXT;

-- Other tables store str(users.id) in their user_id column.
CREATE TABLE IF NOT EXISTS users (
    id             BIGSERIAL PRIMARY KEY,
    email          TEXT NOT NULL UNIQUE,
    password_hash  TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Login attempts per email in a fixed window. Unknown emails get rows too, so a lockout
-- does not reveal which emails are registered.
CREATE TABLE IF NOT EXISTS login_attempts (
    email         TEXT PRIMARY KEY,
    attempts      INT NOT NULL,
    window_start  TIMESTAMPTZ NOT NULL
);
"""

TITLE_CHARS = 80

Status = Literal["processing", "ready", "failed"]

_DOCUMENT_SELECT = """
    SELECT d.id, d.filename, d.pages, count(c.id), d.status, d.error, d.total_chunks, d.embedded_chunks
    FROM documents d
    LEFT JOIN chunks c ON c.document_id = d.id
    WHERE d.user_id = %(user_id)s {extra}
    GROUP BY d.id
    ORDER BY d.id
"""


@dataclass(frozen=True)
class Document:
    id: int
    filename: str
    pages: int
    chunks: int
    status: Status
    error: str | None
    total_chunks: int
    embedded_chunks: int | None


@dataclass(frozen=True)
class SearchResult:
    filename: str
    page: int
    chunk_index: int
    content: str
    score: float


@dataclass(frozen=True)
class User:
    id: int
    email: str


@dataclass(frozen=True)
class Conversation:
    id: int
    title: str
    updated_at: datetime


@dataclass(frozen=True)
class Message:
    question: str
    answer: str
    answered: bool
    sources: list[dict]
    search_query: str | None


def _vector_literal(v: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


class VectorStore:
    """Every per-user method takes user_id and filters by it: a user only sees their own data."""

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    @classmethod
    def connect(cls, url: str | None = None, *, create_schema: bool = True) -> "VectorStore":
        store = cls(psycopg.connect(url or os.environ["DATABASE_URL"], autocommit=True))
        if create_schema:
            store.init_schema()
        store.configure_session()
        return store

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def init_schema(self) -> None:
        with self.conn.transaction():
            self.conn.execute(SCHEMA)

    def configure_session(self) -> None:
        # With a WHERE filter, HNSW could return fewer than LIMIT rows (it finds
        # nearest neighbours first, then drops other users' rows). This makes it
        # keep scanning until enough rows match. Per connection, so run on each.
        self.conn.execute("SET hnsw.iterative_scan = strict_order")

    def find_document(self, user_id: str, sha256: str) -> int | None:
        row = self.conn.execute(
            "SELECT id FROM documents WHERE user_id = %s AND sha256 = %s",
            (user_id, sha256),
        ).fetchone()
        return row[0] if row else None

    def add_document(
        self,
        user_id: str,
        filename: str,
        sha256: str,
        pages: int,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> int:
        # One transaction: either the document and all its chunks are saved, or nothing.
        with self.conn.transaction():
            doc_id = self._insert_document(user_id, filename, sha256, pages, "ready", len(chunks))
            self._insert_chunks(doc_id, user_id, chunks, vectors)
        return doc_id

    def create_document(self, user_id: str, filename: str, sha256: str, pages: int, total_chunks: int) -> int:
        return self._insert_document(user_id, filename, sha256, pages, "processing", total_chunks)

    def set_progress(self, document_id: int, embedded_chunks: int) -> None:
        self.conn.execute(
            "UPDATE documents SET embedded_chunks = %s WHERE id = %s AND status = 'processing'",
            (embedded_chunks, document_id),
        )

    def complete_document(
        self, document_id: int, user_id: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        with self.conn.transaction():
            # Lock the row: if the user deleted the document meanwhile, stop here
            # instead of failing on the foreign key.
            row = self.conn.execute(
                "SELECT 1 FROM documents WHERE id = %s AND status = 'processing' FOR UPDATE",
                (document_id,),
            ).fetchone()
            if row is None:
                return
            self._insert_chunks(document_id, user_id, chunks, vectors)
            self.conn.execute(
                "UPDATE documents SET status = 'ready', error = NULL WHERE id = %s",
                (document_id,),
            )

    def fail_document(self, document_id: int, error: str) -> None:
        self.conn.execute(
            "UPDATE documents SET status = 'failed', error = %s "
            "WHERE id = %s AND status = 'processing'",
            (error, document_id),
        )

    def get_document(self, user_id: str, document_id: int) -> Document | None:
        row = self.conn.execute(
            _DOCUMENT_SELECT.format(extra="AND d.id = %(id)s"),
            {"user_id": user_id, "id": document_id},
        ).fetchone()
        return Document(*row) if row else None

    def list_documents(self, user_id: str) -> list[Document]:
        rows = self.conn.execute(_DOCUMENT_SELECT.format(extra=""), {"user_id": user_id}).fetchall()
        return [Document(*r) for r in rows]

    def delete_document(self, user_id: str, document_id: int) -> bool:
        # Chunks go with it via ON DELETE CASCADE.
        cur = self.conn.execute(
            "DELETE FROM documents WHERE id = %s AND user_id = %s",
            (document_id, user_id),
        )
        return cur.rowcount == 1

    def search(self, user_id: str, query_vector: list[float], k: int = 5) -> list[SearchResult]:
        # Only "ready" documents have chunks, so no status filter is needed here.
        # <=> is cosine distance (0 = same direction), so score = 1 - distance.
        rows = self.conn.execute(
            """
            SELECT d.filename, c.page, c.chunk_index, c.content,
                   1 - (c.embedding <=> %(q)s::vector) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.user_id = %(user_id)s
            ORDER BY c.embedding <=> %(q)s::vector
            LIMIT %(k)s
            """,
            {"q": _vector_literal(query_vector), "user_id": user_id, "k": k},
        ).fetchall()
        return [SearchResult(*r) for r in rows]

    def create_user(self, email: str, password_hash: str) -> User | None:
        # None when the email is taken; ON CONFLICT also covers two sign-ups racing.
        row = self.conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (%s, %s) "
            "ON CONFLICT (email) DO NOTHING RETURNING id, email",
            (email, password_hash),
        ).fetchone()
        return User(*row) if row else None

    def find_login(self, email: str) -> tuple[User, str] | None:
        row = self.conn.execute(
            "SELECT id, email, password_hash FROM users WHERE email = %s", (email,)
        ).fetchone()
        return (User(row[0], row[1]), row[2]) if row else None

    def get_user(self, user_id: int) -> User | None:
        row = self.conn.execute("SELECT id, email FROM users WHERE id = %s", (user_id,)).fetchone()
        return User(*row) if row else None

    def record_login_attempt(self, email: str, window: timedelta) -> tuple[int, float]:
        """Counts one attempt; returns (attempts in the current window, seconds until it ends)."""
        # One atomic statement, so parallel attempts each get their own count.
        # shortcut: rows of emails that stop trying are never deleted; prune old windows if the table grows.
        row = self.conn.execute(
            """
            INSERT INTO login_attempts AS a (email, attempts, window_start) VALUES (%(email)s, 1, now())
            ON CONFLICT (email) DO UPDATE SET
                attempts = CASE WHEN a.window_start <= now() - %(window)s THEN 1 ELSE a.attempts + 1 END,
                window_start = CASE WHEN a.window_start <= now() - %(window)s THEN now() ELSE a.window_start END
            RETURNING attempts, extract(epoch FROM a.window_start + %(window)s - now())
            """,
            {"email": email, "window": window},
        ).fetchone()
        return row[0], float(row[1])

    def clear_login_attempts(self, email: str) -> None:
        self.conn.execute("DELETE FROM login_attempts WHERE email = %s", (email,))

    def conversation_exists(self, user_id: str, conversation_id: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM conversations WHERE id = %s AND user_id = %s",
            (conversation_id, user_id),
        ).fetchone()
        return row is not None

    def save_turn(
        self,
        user_id: str,
        conversation_id: int | None,
        question: str,
        answer: str,
        answered: bool,
        sources: list[dict],
        search_query: str | None = None,
    ) -> int:
        """Saves one question and answer; with no conversation_id, starts a new conversation."""
        with self.conn.transaction():
            if conversation_id is None:
                conversation_id = self.conn.execute(
                    "INSERT INTO conversations (user_id, title) VALUES (%s, %s) RETURNING id",
                    (user_id, question[:TITLE_CHARS]),
                ).fetchone()[0]
            else:
                # The user filter means another user's id updates nothing instead of joining their conversation.
                row = self.conn.execute(
                    "UPDATE conversations SET updated_at = now() WHERE id = %s AND user_id = %s RETURNING id",
                    (conversation_id, user_id),
                ).fetchone()
                if row is None:
                    raise LookupError(f"conversation {conversation_id} not found")
            self.conn.execute(
                "INSERT INTO messages (conversation_id, question, answer, answered, sources, search_query) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (conversation_id, question, answer, answered, Jsonb(sources), search_query),
            )
        return conversation_id

    def list_conversations(self, user_id: str) -> list[Conversation]:
        # shortcut: returns every conversation, add paging if a user ever has hundreds.
        rows = self.conn.execute(
            "SELECT id, title, updated_at FROM conversations WHERE user_id = %s ORDER BY updated_at DESC, id DESC",
            (user_id,),
        ).fetchall()
        return [Conversation(*r) for r in rows]

    def get_conversation(self, user_id: str, conversation_id: int) -> tuple[Conversation, list[Message]] | None:
        row = self.conn.execute(
            "SELECT id, title, updated_at FROM conversations WHERE id = %s AND user_id = %s",
            (conversation_id, user_id),
        ).fetchone()
        if row is None:
            return None
        messages = self.conn.execute(
            "SELECT question, answer, answered, sources, search_query FROM messages "
            "WHERE conversation_id = %s ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return Conversation(*row), [Message(*m) for m in messages]

    def recent_turns(self, user_id: str, conversation_id: int, limit: int) -> list[tuple[str, str]]:
        """The last `limit` (question, answer) pairs, oldest first; empty for another user's conversation."""
        rows = self.conn.execute(
            """
            SELECT m.question, m.answer
            FROM messages m JOIN conversations c ON c.id = m.conversation_id
            WHERE c.id = %s AND c.user_id = %s
            ORDER BY m.id DESC LIMIT %s
            """,
            (conversation_id, user_id, limit),
        ).fetchall()
        return [(q, a) for q, a in reversed(rows)]

    def delete_conversation(self, user_id: str, conversation_id: int) -> bool:
        # Messages go with it via ON DELETE CASCADE.
        cur = self.conn.execute(
            "DELETE FROM conversations WHERE id = %s AND user_id = %s",
            (conversation_id, user_id),
        )
        return cur.rowcount == 1

    def _insert_document(
        self, user_id: str, filename: str, sha256: str, pages: int, status: Status, total_chunks: int
    ) -> int:
        embedded = total_chunks if status == "ready" else None
        return self.conn.execute(
            "INSERT INTO documents (user_id, filename, sha256, pages, status, total_chunks, embedded_chunks) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (user_id, filename, sha256, pages, status, total_chunks, embedded),
        ).fetchone()[0]

    def _insert_chunks(
        self, document_id: int, user_id: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        rows = [
            (document_id, user_id, c.page, c.index, c.text, _vector_literal(v))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks "
                "(document_id, user_id, page, chunk_index, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s::vector)",
                rows,
            )
