import base64
import hashlib
import hmac
import os
import secrets
from datetime import UTC, datetime, timedelta

import jwt

SESSION_COOKIE = "session"
SESSION_DAYS = 7

JWT_SECRET = os.environ["JWT_SECRET"]
if len(JWT_SECRET) < 32:
    raise RuntimeError("JWT_SECRET must be at least 32 characters; generate one with secrets.token_hex(32)")

# OWASP's scrypt option with 16 MiB of memory: about 0.3 s per hash on this machine.
_N, _R, _P = 2**14, 8, 5
_MAXMEM = 64 * 1024 * 1024


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, maxmem=_MAXMEM, dklen=32)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    return f"scrypt${_N}${_R}${_P}${_b64(salt)}${_b64(_scrypt(password, salt, _N, _R, _P))}"


def verify_password(password: str, stored: str) -> bool:
    _, n, r, p, salt, key = stored.split("$")
    actual = _scrypt(password, base64.b64decode(salt), int(n), int(r), int(p))
    return hmac.compare_digest(actual, base64.b64decode(key))


# Checked against when the email is unknown, so login takes the same time either way.
DUMMY_HASH = hash_password(secrets.token_hex(16))


def create_token(user_id: int) -> str:
    # shortcut: logout only deletes the cookie, a copied token stays valid until it expires; add a denylist if that matters.
    now = datetime.now(UTC)
    claims = {"sub": str(user_id), "iat": now, "exp": now + timedelta(days=SESSION_DAYS)}
    return jwt.encode(claims, JWT_SECRET, algorithm="HS256")


def read_token(token: str) -> int | None:
    try:
        claims = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], options={"require": ["exp", "sub"]})
    except jwt.InvalidTokenError:
        return None
    return int(claims["sub"])
