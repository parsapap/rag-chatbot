import jwt
import pytest
from fastapi.testclient import TestClient

from app.api import documents
from app.api.deps import get_store
from app.core import security
from app.main import app

PASSWORD = "correct horse battery"


@pytest.fixture
def client(store):
    app.dependency_overrides[get_store] = lambda: store
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def _register(client, email="alice@example.com", password=PASSWORD):
    return client.post("/auth/register", json={"email": email, "password": password})


def test_register_logs_in_with_an_httponly_cookie(client):
    r = _register(client, email="  Alice@Example.COM ")
    assert r.status_code == 201
    assert r.json()["email"] == "alice@example.com"
    cookie = r.headers["set-cookie"].lower()
    assert "session=" in cookie and "httponly" in cookie and "samesite=lax" in cookie
    assert client.get("/auth/me").json()["email"] == "alice@example.com"


def test_password_is_stored_hashed(client, store):
    _register(client)
    stored = store.conn.execute("SELECT password_hash FROM users").fetchone()[0]
    assert PASSWORD not in stored and stored.startswith("scrypt$")
    assert security.verify_password(PASSWORD, stored)
    assert not security.verify_password("wrong password", stored)


def test_same_email_cannot_register_twice(client):
    _register(client)
    assert _register(client, email="ALICE@example.com").status_code == 409


@pytest.mark.parametrize(
    "body",
    [
        {"email": "not-an-email", "password": PASSWORD},
        {"email": "a@b.co", "password": "short"},
        {"email": "a@b.co", "password": "x" * 129},
        {"email": "a@b.co"},
    ],
)
def test_bad_registration_rejected(client, body):
    assert client.post("/auth/register", json=body).status_code == 422


def test_login_wrong_password_and_unknown_email_get_the_same_answer(client):
    _register(client)
    client.cookies.clear()
    wrong = client.post("/auth/login", json={"email": "alice@example.com", "password": "nope-nope"})
    unknown = client.post("/auth/login", json={"email": "bob@example.com", "password": "nope-nope"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()
    assert "set-cookie" not in wrong.headers


def test_login_then_logout(client):
    _register(client)
    client.cookies.clear()
    r = client.post("/auth/login", json={"email": "alice@example.com", "password": PASSWORD})
    assert r.status_code == 200
    assert client.get("/auth/me").status_code == 200
    assert client.post("/auth/logout").status_code == 204
    assert client.get("/auth/me").status_code == 401


@pytest.mark.parametrize("path", ["/documents", "/conversations", "/auth/me"])
def test_endpoints_need_a_session(client, path):
    assert client.get(path).status_code == 401


def test_chat_needs_a_session(client):
    assert client.post("/chat", json={"question": "q"}).status_code == 401


def test_old_user_id_header_no_longer_works(client):
    assert client.get("/documents", headers={"X-User-Id": "demo_a"}).status_code == 401


def test_forged_or_expired_tokens_are_rejected(client):
    _register(client)
    client.cookies.clear()
    forged = jwt.encode({"sub": "1", "exp": 9999999999}, "not-the-secret" * 3, algorithm="HS256")
    client.cookies.set("session", forged)
    assert client.get("/auth/me").status_code == 401
    expired = jwt.encode({"sub": "1", "exp": 1}, security.JWT_SECRET, algorithm="HS256")
    client.cookies.set("session", expired)
    assert client.get("/auth/me").status_code == 401
    unsigned = jwt.encode({"sub": "1", "exp": 9999999999}, None, algorithm="none")
    client.cookies.set("session", unsigned)
    assert client.get("/auth/me").status_code == 401


def test_two_accounts_only_see_their_own_documents(client, monkeypatch):
    # Skip background embedding: it would call Gemini and open its own connection to the real database.
    monkeypatch.setattr(documents, "process_document", lambda *args: None)
    _register(client, email="alice@example.com")
    client.post("/documents", files={"file": ("a.txt", b"Alice's private notes about the budget. " * 3)})
    assert len(client.get("/documents").json()) == 1
    client.post("/auth/logout")
    _register(client, email="bob@example.com")
    assert client.get("/documents").json() == []
