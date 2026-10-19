import os
import uuid

import psycopg
import pytest

from app.services.chunking import Chunk
from app.services.embeddings import DIM
from app.db.store import VectorStore


@pytest.fixture
def store():
    # Real Postgres, but each test gets its own throwaway schema.
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    schema = f"test_{uuid.uuid4().hex[:8]}"
    conn.execute(f'CREATE SCHEMA "{schema}"')
    conn.execute(f'SET search_path TO "{schema}", public')
    s = VectorStore(conn)
    s.init_schema()
    yield s
    conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
    conn.close()


def _vec(*hot: int) -> list[float]:
    v = [0.0] * DIM
    for i in hot:
        v[i] = 1.0
    n = len(hot) ** 0.5
    return [x / n for x in v]


def _chunks(name: str, n: int) -> list[Chunk]:
    return [Chunk(text=f"{name} متن {i}", source=name, page=i + 1, index=i) for i in range(n)]


def _add(store, user, name, vectors, sha=None):
    return store.add_document(user, name, sha or name, len(vectors), _chunks(name, len(vectors)), vectors)


def test_search_returns_closest_first(store):
    _add(store, "a", "a.pdf", [_vec(0), _vec(1), _vec(2)])
    results = store.search("a", _vec(1), k=3)
    assert [r.chunk_index for r in results][0] == 1
    assert results[0].score == pytest.approx(1.0)
    assert results[0].page == 2
    assert results[0].filename == "a.pdf"
    assert results[1].score == pytest.approx(0.0)


def test_users_never_see_each_others_chunks(store):
    _add(store, "a", "secret.pdf", [_vec(5)])
    _add(store, "b", "b.pdf", [_vec(9)])
    results = store.search("b", _vec(5), k=10)
    assert [r.filename for r in results] == ["b.pdf"]
    assert store.search("nobody", _vec(5)) == []


def test_duplicate_is_per_user(store):
    doc_id = _add(store, "a", "x.pdf", [_vec(0)], sha="same-hash")
    assert store.find_document("a", "same-hash") == doc_id
    assert store.find_document("b", "same-hash") is None
    with pytest.raises(psycopg.errors.UniqueViolation):
        _add(store, "a", "x-copy.pdf", [_vec(0)], sha="same-hash")
    assert [d.filename for d in store.list_documents("a")] == ["x.pdf"]


def test_list_documents_counts_chunks(store):
    _add(store, "a", "one.pdf", [_vec(0), _vec(1)])
    _add(store, "a", "two.pdf", [_vec(2)])
    docs = store.list_documents("a")
    assert [(d.filename, d.chunks) for d in docs] == [("one.pdf", 2), ("two.pdf", 1)]


def test_only_owner_can_delete_and_chunks_go_too(store):
    doc_id = _add(store, "a", "a.pdf", [_vec(0), _vec(1)])
    assert store.delete_document("b", doc_id) is False
    assert len(store.search("a", _vec(0))) == 2
    assert store.delete_document("a", doc_id) is True
    assert store.search("a", _vec(0)) == []
    assert store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


def test_chunk_vector_count_mismatch_saves_nothing(store):
    with pytest.raises(ValueError):
        store.add_document("a", "bad.pdf", "h", 1, _chunks("bad.pdf", 2), [_vec(0)])
    assert store.list_documents("a") == []
