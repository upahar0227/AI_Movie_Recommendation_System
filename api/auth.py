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
    email: EmailStr
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def username_chars(cls, value: str) -> str:
        if not value.replace("_", "").isalnum():
            raise ValueError("Username can use letters, numbers, and underscores.")
        return value

    @field_validator("password")
    @classmethod
    def password_strength(cls, value: str) -> str:
        if value.isdecimal():
            raise ValueError("Password cannot be only numbers.")
        return value


class UserOut(BaseModel):
    id: int
    email: EmailStr
    username: str
    onboarding_completed: bool

    model_config = {"from_attributes": True}


class RefreshBody(BaseModel):
    refresh_token: str


class LogoutBody(BaseModel):
    refresh_token: str | None = None


def _issue(user_id: int) -> dict:
    try:
        return issue_tokens(user_id, auth_settings())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Accounts are not configured.") from exc


def _revoke_raw(token: str, session: Session) -> None:
    try:
        payload = decode_token(token, auth_settings())
    except jwt.PyJWTError:
        return
    revoke(session, str(payload["jti"]), expiry_from_payload(payload))


@router.post("/auth/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
def register(request: Request, body: RegisterBody, session: Session = Depends(get_db)):
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
    email = form.username.strip().lower()
    user = session.scalar(select(User).where(User.email == email))
    if user is None or not user.is_active or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Email or password is wrong.")
    return _issue(user.id)


@router.post("/auth/refresh")
def refresh(body: RefreshBody, session: Session = Depends(get_db)):
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
    del user
    header = request.headers.get("authorization", "")
    _revoke_raw(header.removeprefix("Bearer ").strip(), session)
    if body.refresh_token:
        _revoke_raw(body.refresh_token, session)
    return {"ok": True}


@router.get("/users/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


@router.delete("/users/me")
def delete_me(user: User = Depends(get_current_user), session: Session = Depends(get_db)):
    session.delete(user)
    return {"ok": True}
