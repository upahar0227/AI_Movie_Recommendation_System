"""Register, log in, refresh, and delete an account."""

from __future__ import annotations

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.deps import expiry_from_payload, get_current_user, get_db, revoke
from api.limiter import limiter
from src.app.models import RevokedToken, User
from src.app.security import auth_settings, decode_token, hash_password, issue_tokens, verify_password

router = APIRouter(tags=["auth"])


class RegisterBody(BaseModel):
    """Email, username, and password submitted to create an account."""

    email: EmailStr
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def username_chars(cls, value: str) -> str:
        """Keep a username made of letters, numbers, and underscores.

        Args:
            value: Username after the length check.

        Returns:
            The same username.

        Raises:
            ValueError: The name contains any other character. FastAPI reports this as 422.
        """
        if not value.replace("_", "").isalnum():
            raise ValueError("Username can use letters, numbers, and underscores.")
        return value

    @field_validator("password")
    @classmethod
    def password_strength(cls, value: str) -> str:
        """Reject a password that is only digits.

        Args:
            value: Password after the length check.

        Returns:
            The same password.

        Raises:
            ValueError: Every character is a digit. FastAPI reports this as 422.
        """
        if value.isdecimal():
            raise ValueError("Password cannot be only numbers.")
        return value


class UserOut(BaseModel):
    """Public account fields. The password hash is not included."""

    id: int
    email: EmailStr
    username: str
    onboarding_completed: bool

    model_config = {"from_attributes": True}


class RefreshBody(BaseModel):
    """A refresh token sent to obtain a new pair of tokens."""

    refresh_token: str


class LogoutBody(BaseModel):
    """Optional refresh token to revoke along with the access token."""

    refresh_token: str | None = None


def _issue(user_id: int) -> dict:
    """Issue an access token and a refresh token for an account.

    Args:
        user_id: Account id placed in both tokens.

    Returns:
        The token payload from ``issue_tokens``.

    Raises:
        HTTPException: 503 when ``JWT_SECRET`` is missing.
    """
    try:
        return issue_tokens(user_id, auth_settings())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Accounts are not configured.") from exc


def _revoke_raw(token: str, session: Session) -> None:
    """Revoke one raw JWT. An invalid token is ignored.

    Args:
        token: Compact JWT, without the ``Bearer`` prefix.
        session: Account database session.

    Side effects:
        Stores the token id when the signature and expiry check succeed.
    """
    try:
        payload = decode_token(token, auth_settings())
    except jwt.PyJWTError:
        return
    revoke(session, str(payload["jti"]), expiry_from_payload(payload))


@router.post("/auth/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
def register(request: Request, body: RegisterBody, session: Session = Depends(get_db)):
    """POST /auth/register. Sign-in is not required. Limited to 5 requests a minute.

    Body: email, username, and password. The email is stored in lowercase.

    Returns:
        201 with the new account's id, email, username, and onboarding flag.

    Raises:
        HTTPException: 409 when the email or username is taken, 422 when the body
        fails validation, 429 when the rate limit is exceeded, 503 when the database is missing.

    Side effects:
        Inserts the user. ``get_db`` commits.
    """
    user = User(email=str(body.email).lower(), username=body.username, password_hash=hash_password(body.password))
    session.add(user)
    try:
        session.flush()
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail="That email or username is already registered.") from exc
    return user


@router.post("/auth/login")
@limiter.limit("5/minute")
def login(request: Request, form: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_db)):
    """POST /auth/login. Sign-in is not required. Limited to 5 requests a minute.

    Form: ``username`` is the email address, and ``password`` is the account password.

    Returns:
        Access token, refresh token, bearer type, and access lifetime in seconds.

    Raises:
        HTTPException: 401 when the email, password, or active flag does not match,
        429 when the rate limit is exceeded, 503 when the secret or database is missing.
    """
    email = form.username.strip().lower()
    user = session.scalar(select(User).where(User.email == email))
    if user is None or not user.is_active or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Email or password is wrong.")
    return _issue(user.id)


@router.post("/auth/refresh")
def refresh(body: RefreshBody, session: Session = Depends(get_db)):
    """POST /auth/refresh. Sign-in is not required.

    Body: ``refresh_token``. An access token, a revoked token, or an inactive account is rejected.

    Returns:
        A new access token and refresh token.

    Raises:
        HTTPException: 401 when the refresh token is unusable, 503 when the secret or database is missing.
    """
    try:
        settings = auth_settings()
        payload = decode_token(body.refresh_token, settings)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Accounts are not configured.") from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Session expired.") from exc
    if payload.get("type") != "refresh" or session.get(RevokedToken, payload.get("jti")) is not None:
        raise HTTPException(status_code=401, detail="Session expired.")
    user = session.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="Session expired.")
    return issue_tokens(user.id, settings)


@router.post("/auth/logout")
def logout(
    request: Request,
    body: LogoutBody,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_db),
):
    """POST /auth/logout. Sign-in is required.

    Body: optional ``refresh_token``. The access token is taken from the Authorization header.
    A token that fails verification is skipped.

    Returns:
        ``{"ok": true}``.

    Raises:
        HTTPException: 401 when the access token is missing or unusable, 503 when accounts are not configured.

    Side effects:
        Stores the token ids that verify. ``get_db`` commits.
    """
    del user
    header = request.headers.get("authorization", "")
    _revoke_raw(header.removeprefix("Bearer ").strip(), session)
    if body.refresh_token:
        _revoke_raw(body.refresh_token, session)
    return {"ok": True}


@router.get("/users/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    """GET /users/me. Sign-in is required.

    Returns:
        The signed-in account's id, email, username, and onboarding flag.

    Raises:
        HTTPException: 401 when the access token is missing or unusable, 503 when accounts are not configured.
    """
    return user


@router.delete("/users/me")
def delete_me(user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    """DELETE /users/me. Sign-in is required.

    Returns:
        ``{"ok": true}``.

    Raises:
        HTTPException: 401 when the access token is missing or unusable, 503 when accounts are not configured.

    Side effects:
        Deletes the user row. Related rows follow the foreign-key cascade. ``get_db`` commits.
    """
    session.delete(user)
    return {"ok": True}
