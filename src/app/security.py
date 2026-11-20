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
    """JWT secret, token lifetimes, and allowed browser origins."""

    secret: str
    access_minutes: int
    refresh_days: int
    cors_origins: list[str]


def auth_settings() -> AuthSettings:
    """Read auth settings from the environment. The secret is required for account routes.

    Access tokens default to 30 minutes and refresh tokens to 7 days. CORS defaults
    to ``http://127.0.0.1:8000`` when ``CORS_ORIGINS`` is unset.

    Returns:
        Secret, lifetimes, and the non-empty origins.

    Raises:
        RuntimeError: ``JWT_SECRET`` is missing or blank.
        ValueError: A lifetime setting is not an integer.

    Side effects:
        Reads those environment variables.
    """
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
    """Hash a password with bcrypt and a new salt.

    Args:
        password: Plain text. It is encoded as UTF-8 before hashing.

    Returns:
        The bcrypt hash as text, including the salt.
    """
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Check a plain password against a stored bcrypt hash.

    Args:
        password: Plain text submitted at sign-in.
        password_hash: Hash stored on the user row.

    Returns:
        True when the password matches the hash.

    Raises:
        ValueError: ``password_hash`` is not a bcrypt hash.
    """
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def _encode(user_id: int, token_type: str, expires: datetime, secret: str) -> tuple[str, str]:
    """Sign one HS256 JWT and return it with a new token id.

    Args:
        user_id: Account id stored in the ``sub`` claim.
        token_type: ``access`` or ``refresh``, stored in the ``type`` claim.
        expires: Expiry stored in the ``exp`` claim.
        secret: HMAC secret.

    Returns:
        The encoded token and its ``jti`` hex string.
    """
    jti = uuid.uuid4().hex
    payload = {"sub": str(user_id), "type": token_type, "jti": jti, "exp": expires}
    return jwt.encode(payload, secret, algorithm=ALGORITHM), jti


def issue_tokens(user_id: int, settings: AuthSettings) -> dict:
    """Create a new access token and refresh token for an account.

    Args:
        user_id: Account id placed in both ``sub`` claims.
        settings: Secret and lifetimes. Expiry is measured from the current UTC time.

    Returns:
        ``access_token``, ``refresh_token``, ``token_type`` of ``bearer``, and
        ``expires_in`` as the access lifetime in seconds.
    """
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
    """Verify an HS256 token and return its claims.

    Args:
        token: Compact JWT.
        settings: Secret used to check the signature. Only HS256 is accepted.

    Returns:
        Claims including ``sub``, ``type``, ``jti``, and ``exp``.

    Raises:
        jwt.PyJWTError: The token is malformed, signed with another secret, or expired.
    """
    return jwt.decode(token, settings.secret, algorithms=[ALGORITHM])


def recommendation_mode(n_ratings: int) -> str:
    """Label how many ratings the account has supplied.

    Zero ratings are ``cold_start``. Fewer than five are ``warming_up``. Five or more are ``personalised``.

    Args:
        n_ratings: Saved ratings for the account. Negative counts are treated as none.

    Returns:
        ``cold_start``, ``warming_up``, or ``personalised``.
    """
    if n_ratings <= 0:
        return "cold_start"
    if n_ratings < PERSONALISED_AFTER:
        return "warming_up"
    return "personalised"
