"""Request dependencies for the account database and the signed-in user."""

from __future__ import annotations

from datetime import datetime, timezone

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from src.app.models import RevokedToken, User
from src.app.security import auth_settings, decode_token

_bearer = HTTPBearer(auto_error=False)


def get_db(request: Request):
    """Yield a database session and commit it when the request succeeds.

    Args:
        request: The current request. Its app must already hold a session factory.

    Returns:
        A SQLAlchemy session. The caller uses it as a FastAPI dependency.

    Raises:
        HTTPException: 503 when the accounts database is not configured.

    Side effects:
        Commits on success, rolls back on any exception, and always closes the session.
    """
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise HTTPException(status_code=503, detail="Accounts database is not configured.")
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: Session = Depends(get_db),
) -> User:
    """Return the account named by a valid access token.

    Args:
        credentials: Bearer token. Missing credentials are rejected.
        session: Account database session.

    Returns:
        The active user whose id is in the token's ``sub`` claim.

    Raises:
        HTTPException: 401 when the token is missing, not an access token, revoked,
        expired, or names an inactive user. 503 when ``JWT_SECRET`` is missing.
    """
    if credentials is None:
        raise HTTPException(status_code=401, detail="Sign in required.", headers={"WWW-Authenticate": "Bearer"})
    try:
        settings = auth_settings()
        payload = decode_token(credentials.credentials, settings)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Accounts are not configured.") from exc
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Session expired.") from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Sign in required.") from exc
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Sign in required.")
    revoked = session.get(RevokedToken, payload.get("jti"))
    if revoked is not None:
        raise HTTPException(status_code=401, detail="Session expired.")
    user = session.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="Sign in required.")
    return user


def revoke(session: Session, jti: str, expires_at: datetime) -> None:
    """Store a token id so it can no longer be accepted.

    Args:
        session: Account database session. An id that is already stored is left as it is.
        jti: JWT id.
        expires_at: When that token would have expired.

    Side effects:
        Adds a revoked-token row. The caller commits.
    """
    if session.get(RevokedToken, jti) is None:
        session.add(RevokedToken(jti=jti, expires_at=expires_at))


def expiry_from_payload(payload: dict) -> datetime:
    """Read the token expiry as a timezone-aware UTC datetime.

    Args:
        payload: Decoded claims containing ``exp`` as a Unix timestamp.

    Returns:
        That timestamp in UTC.
    """
    return datetime.fromtimestamp(int(payload["exp"]), tz=timezone.utc)
