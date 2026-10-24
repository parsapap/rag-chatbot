from typing import Annotated

from pydantic import BaseModel, StringConstraints

# Lowercased so the unique constraint treats Alice@x.com and alice@x.com as one account.
Email = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_lower=True, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
]


class LoginRequest(BaseModel):
    email: Email
    # The upper bound keeps one request from making scrypt hash megabytes.
    password: Annotated[str, StringConstraints(min_length=1, max_length=128)]


class RegisterRequest(LoginRequest):
    password: Annotated[str, StringConstraints(min_length=8, max_length=128)]


class UserOut(BaseModel):
    id: int
    email: str
