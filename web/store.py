from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "web" / "users.db"
SECRET_PATH = ROOT / "data" / "web" / "secret.key"
SESSION_DAYS = 14


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                genres TEXT NOT NULL DEFAULT '[]',
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS interactions (
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                rating REAL NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (user_id, movie_id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )


def _secret() -> bytes:
    SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SECRET_PATH.exists():
        return SECRET_PATH.read_bytes()
    key = secrets.token_bytes(32)
    SECRET_PATH.write_bytes(key)
    return key


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split(":", 1)
    candidate = hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(candidate, f"{salt_hex}:{digest_hex}")


def create_user(username: str, password: str) -> dict:
    username = username.strip()
    if len(username) < 3 or len(username) > 32:
        raise ValueError("Username must be 3–32 characters.")
    if not username.replace("_", "").isalnum():
        raise ValueError("Username can use letters, numbers, and underscores.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    now = int(time.time())
    try:
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, hash_password(password), now),
            )
            user_id = int(cur.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise ValueError("That username is already taken.") from exc
    return get_user(user_id)


def authenticate(username: str, password: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return _public_user(row)


def get_user(user_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _public_user(row) if row else None


def _public_user(row: sqlite3.Row) -> dict:
    genres = json.loads(row["genres"] or "[]")
    interactions = list_interactions(int(row["id"]))
    return {
        "id": int(row["id"]),
        "username": row["username"],
        "genres": genres,
        "interactionCount": len(interactions),
        "isNew": len(interactions) == 0,
    }


def set_genres(user_id: int, genres: list[str]) -> dict:
    cleaned = []
    for genre in genres:
        name = str(genre).strip()
        if name and name not in cleaned:
            cleaned.append(name)
    if not cleaned:
        raise ValueError("Pick at least one genre.")
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET genres = ? WHERE id = ?",
            (json.dumps(cleaned), user_id),
        )
    user = get_user(user_id)
    if user is None:
        raise ValueError("Account not found.")
    return user


def list_interactions(user_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT movie_id, rating, created_at
            FROM interactions
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [
        {"movieId": int(row["movie_id"]), "rating": float(row["rating"]), "createdAt": int(row["created_at"])}
        for row in rows
    ]


def save_rating(user_id: int, movie_id: int, rating: float) -> list[dict]:
    if rating < 1 or rating > 5:
        raise ValueError("Rate from 1 to 5.")
    now = int(time.time())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO interactions (user_id, movie_id, rating, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, movie_id) DO UPDATE SET
                rating = excluded.rating,
                created_at = excluded.created_at
            """,
            (user_id, int(movie_id), float(rating), now),
        )
    return list_interactions(user_id)


def sign_session(user_id: int) -> str:
    exp = int(time.time()) + SESSION_DAYS * 24 * 3600
    payload = f"{user_id}.{exp}"
    sig = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def read_session(token: str | None) -> int | None:
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    user_id_s, exp_s, sig = parts
    payload = f"{user_id_s}.{exp_s}"
    expected = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        if int(exp_s) < int(time.time()):
            return None
        return int(user_id_s)
    except ValueError:
        return None
