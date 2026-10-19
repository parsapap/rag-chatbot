import os
from dataclasses import dataclass

import psycopg

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
"""


@dataclass(frozen=True)
class Document:
    id: int
    filename: str
    pages: int
    chunks: int


@dataclass(frozen=True)
class SearchResult:
    filename: str
    page: int
    chunk_index: int
    content: str
    score: float


def _vector_literal(v: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


class VectorStore:
    """Every method takes user_id and filters by it: a user only sees their own data."""

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    @classmethod
    def connect(cls, url: str | None = None) -> "VectorStore":
        store = cls(psycopg.connect(url or os.environ["DATABASE_URL"], autocommit=True))
        store.init_schema()
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
        # With a WHERE filter, HNSW could return fewer than LIMIT rows (it finds
        # nearest neighbours first, then drops other users' rows). This makes it
        # keep scanning until enough rows match.
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
        rows = [
            (c.page, c.index, c.text, _vector_literal(v))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        # One transaction: either the document and all its chunks are saved, or nothing.
        with self.conn.transaction():
            doc_id = self.conn.execute(
                "INSERT INTO documents (user_id, filename, sha256, pages) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (user_id, filename, sha256, pages),
            ).fetchone()[0]
            with self.conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks "
                    "(document_id, user_id, page, chunk_index, content, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s::vector)",
                    [(doc_id, user_id, *r) for r in rows],
                )
        return doc_id

    def search(self, user_id: str, query_vector: list[float], k: int = 5) -> list[SearchResult]:
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

    def list_documents(self, user_id: str) -> list[Document]:
        rows = self.conn.execute(
            """
            SELECT d.id, d.filename, d.pages, count(c.id)
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            WHERE d.user_id = %s
            GROUP BY d.id
            ORDER BY d.id
            """,
            (user_id,),
        ).fetchall()
        return [Document(*r) for r in rows]

    def delete_document(self, user_id: str, document_id: int) -> bool:
        # Chunks go with it via ON DELETE CASCADE.
        cur = self.conn.execute(
            "DELETE FROM documents WHERE id = %s AND user_id = %s",
            (document_id, user_id),
        )
        return cur.rowcount == 1
