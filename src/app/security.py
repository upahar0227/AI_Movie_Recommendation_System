"""Password hashing and JWT creation. Tokens and passwords are never logged."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from dotenv import load_dotenv

from src.recsys.config import ROOT

load_dotenv(ROOT / ".env")

ALGORITHM = "HS256"
PERSONALISED_AFTER = 5


@dataclass(frozen=True)
class AuthSettings:
    secret: str
    access_minutes: int
    refresh_days: int
    cors_origins: list[str]


def auth_settings() -> AuthSettings:
    """Read auth settings from the environment. The secret is required for account routes."""
    secret = os.environ.get("JWT_SECRET", "").strip()
    if not secret:
        raise RuntimeError("JWT_SECRET is missing. Set it in .env before starting the API.")
    origins = [item.strip() for item in os.environ.get("CORS_ORIGINS", "http://127.0.0.1:8000").split(",") if item.strip()]
    return AuthSettings(
        secret=secret,
        access_minutes=int(os.environ.get("ACCESS_TOKEN_MINUTES", "30")),
        refresh_days=int(os.environ.get("REFRESH_TOKEN_DAYS", "7")),
        cors_origins=origins,
    )


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def _encode(user_id: int, token_type: str, expires: datetime, secret: str) -> tuple[str, str]:
    jti = uuid.uuid4().hex
    payload = {"sub": str(user_id), "type": token_type, "jti": jti, "exp": expires}
    return jwt.encode(payload, secret, algorithm=ALGORITHM), jti


def issue_tokens(user_id: int, settings: AuthSettings) -> dict:
    now = datetime.now(timezone.utc)
    access, _access_jti = _encode(user_id, "access", now + timedelta(minutes=settings.access_minutes), settings.secret)
    refresh, _refresh_jti = _encode(user_id, "refresh", now + timedelta(days=settings.refresh_days), settings.secret)
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "expires_in": settings.access_minutes * 60,
    }


def decode_token(token: str, settings: AuthSettings) -> dict:
    return jwt.decode(token, settings.secret, algorithms=[ALGORITHM])


def recommendation_mode(n_ratings: int) -> str:
    """cold_start with no ratings, warming_up under five, then personalised."""
    if n_ratings <= 0:
        return "cold_start"
    if n_ratings < PERSONALISED_AFTER:
        return "warming_up"
    return "personalised"
