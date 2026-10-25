import math
from datetime import timedelta

from fastapi import APIRouter, HTTPException, Response

from app.api.deps import CurrentUser, Store
from app.core.security import (
    DUMMY_HASH,
    SESSION_COOKIE,
    SESSION_DAYS,
    create_token,
    hash_password,
    verify_password,
)
from app.db.store import User
from app.schemas.auth import LoginRequest, RegisterRequest, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])

MAX_LOGIN_ATTEMPTS = 5
LOGIN_WINDOW = timedelta(minutes=15)

# shortcut: the limit is per email only; guessing across many emails and floods of /register (each one
# a scrypt hash) need a per-IP limit at the reverse proxy before the app is exposed.
_COOKIE = {
    "httponly": True,  # page scripts cannot read the token, so an XSS bug cannot steal it
    "samesite": "lax",  # other sites cannot send POSTs that carry the cookie
    # shortcut: not Secure because the app is served over plain http on 127.0.0.1; set secure=True once behind https.
    "secure": False,
    "path": "/",
}


def _start_session(response: Response, user: User) -> UserOut:
    response.set_cookie(SESSION_COOKIE, create_token(user.id), max_age=SESSION_DAYS * 24 * 3600, **_COOKIE)
    return UserOut(id=user.id, email=user.email)


@router.post("/register", status_code=201, response_model=UserOut)
def register(body: RegisterRequest, response: Response, store: Store) -> UserOut:
    user = store.create_user(body.email, hash_password(body.password))
    if user is None:
        raise HTTPException(409, "This email is already registered.")
    return _start_session(response, user)


@router.post("/login", response_model=UserOut)
def login(body: LoginRequest, response: Response, store: Store) -> UserOut:
    # Counted before the password check: blocked attempts cost no scrypt time, and a burst of
    # parallel guesses cannot all pass on the same count.
    attempts, seconds_left = store.record_login_attempt(body.email, LOGIN_WINDOW)
    if attempts > MAX_LOGIN_ATTEMPTS:
        minutes = math.ceil(seconds_left / 60)
        raise HTTPException(
            429,
            f"Too many login attempts. Try again in {minutes} minute{'' if minutes == 1 else 's'}.",
            headers={"Retry-After": str(math.ceil(seconds_left))},
        )
    found = store.find_login(body.email)
    # Hash even for an unknown email, so the response time does not reveal which emails exist.
    ok = verify_password(body.password, found[1] if found else DUMMY_HASH)
    if found is None or not ok:
        raise HTTPException(401, "Wrong email or password.")
    store.clear_login_attempts(body.email)
    return _start_session(response, found[0])


@router.post("/logout", status_code=204)
def logout() -> Response:
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE, **_COOKIE)
    return response


@router.get("/me", response_model=UserOut)
def me(user_id: CurrentUser, store: Store) -> UserOut:
    user = store.get_user(int(user_id))
    if user is None:
        raise HTTPException(401, "Log in first.")
    return UserOut(id=user.id, email=user.email)
