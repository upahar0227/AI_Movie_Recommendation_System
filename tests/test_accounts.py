"""Account registration, tokens, ratings, and personalised recommendations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from api.main import create_app
from src.app.database import make_engine, make_session_factory
from src.app.models import Interaction, Rating, User, WatchlistItem
from src.recsys.cf_model import fit_collaborative
from src.recsys.content_model import ContentModel
from src.recsys.serving import RecommenderService
from tests.test_schema import _upgrade


def _service() -> RecommenderService:
    import pandas as pd

    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5, 6],
            "title": ["One", "Two", "Three", "Four", "Five", "Six"],
            "genre_names": ["Drama"] * 6,
            "tags": ["alpha space", "alpha war", "beta love", "beta drama", "gamma space", "delta war"],
            "poster_url": [None] * 6,
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 2, 3, 4, 5, 6],
            "rating": [5, 4, 3, 5, 4, 3],
        }
    )
    return RecommenderService(
        movies=movies,
        content=ContentModel().fit(movies),
        collaborative=fit_collaborative(movies, ratings),
        sentiment=[0.2, 0.4, 0.6, 0.8, 0.3, 0.5],
        histories={},
        hybrid_k=5,
        hybrid_beta=0.15,
        reception_method="bayesian_average_movielens",
    )


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("JWT_SECRET", "test-secret-value")
    url = _upgrade(tmp_path, monkeypatch)
    return TestClient(create_app(_service(), database_url=url, enforce_secret=False))


def _login(client: TestClient, email: str, password: str) -> dict:
    response = client.post("/auth/login", data={"username": email, "password": password})
    assert response.status_code == 200
    return response.json()


def test_register_duplicate_and_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    created = client.post(
        "/auth/register",
        json={"email": "reader@example.com", "username": "reader", "password": "correct-horse"},
    )
    assert created.status_code == 201
    assert "password" not in created.json()
    duplicate = client.post(
        "/auth/register",
        json={"email": "reader@example.com", "username": "other", "password": "correct-horse"},
    )
    assert duplicate.status_code == 409
    weak = client.post("/auth/register", json={"email": "a@example.com", "username": "alpha", "password": "12345678"})
    assert weak.status_code == 422
    tokens = _login(client, "reader@example.com", "correct-horse")
    bad = client.post("/auth/login", data={"username": "missing@example.com", "password": "correct-horse"})
    wrong = client.post("/auth/login", data={"username": "reader@example.com", "password": "not-the-password"})
    assert bad.status_code == 401 and wrong.status_code == 401
    assert bad.json()["detail"] == wrong.json()["detail"]
    me = client.get("/users/me", headers={"Authorization": "Bearer " + tokens["access_token"]})
    assert me.status_code == 200
    assert me.json()["username"] == "reader"
    assert me.json()["onboarding_completed"] is False


def test_protected_and_expired_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    assert client.get("/ratings/me").status_code == 401
    expired = jwt.encode(
        {"sub": "1", "type": "access", "jti": "old", "exp": datetime.now(timezone.utc) - timedelta(minutes=5)},
        "test-secret-value",
        algorithm="HS256",
    )
    rejected = client.get("/ratings/me", headers={"Authorization": "Bearer " + expired})
    assert rejected.status_code == 401


def test_ratings_watchlist_recommendations_and_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    client.post("/auth/register", json={"email": "fan@example.com", "username": "fan", "password": "correct-horse"})
    token = _login(client, "fan@example.com", "correct-horse")["access_token"]
    headers = {"Authorization": "Bearer " + token}

    cold = client.get("/recommendations/me", headers=headers)
    assert cold.status_code == 200
    assert cold.json()["mode"] == "ask_genres"
    assert cold.json()["results"] == []
    genres = client.get("/genres", headers=headers)
    assert "Drama" in genres.json()["genres"]
    saved = client.post("/genres", headers=headers, json={"genres": ["Drama"]})
    assert saved.status_code == 200
    assert client.get("/users/me", headers=headers).json()["onboarding_completed"] is True
    by_genre = client.get("/recommendations/me", headers=headers)
    assert by_genre.json()["mode"] == "genre"
    assert by_genre.json()["results"]
    assert all("Drama" in item["genres"] for item in by_genre.json()["results"])

    first = client.post("/ratings", headers=headers, json={"movie_id": 1, "rating": 4})
    assert first.status_code == 200
    updated = client.post("/ratings", headers=headers, json={"movie_id": 1, "rating": 5})
    assert updated.status_code == 200
    invalid = client.post("/ratings", headers=headers, json={"movie_id": 1, "rating": 4.2})
    assert invalid.status_code == 422
    listed = client.get("/ratings/me", headers=headers).json()["ratings"]
    assert listed[0]["movie_id"] == 1
    assert listed[0]["rating"] == 5.0
    assert listed[0]["title"] == "One"

    assert client.post("/watchlist/2", headers=headers).status_code == 200
    assert client.post("/watchlist/2", headers=headers).status_code == 409
    watch = client.get("/watchlist/me", headers=headers).json()["watchlist"]
    assert watch[0]["movie_id"] == 2
    assert watch[0]["title"] == "Two"

    onboard = client.post(
        "/onboarding",
        headers=headers,
        json={"ratings": [{"movie_id": movie_id, "rating": 4} for movie_id in (1, 2, 3, 4, 5)]},
    )
    assert onboard.status_code == 200
    personal = client.get("/recommendations/me", headers=headers)
    assert personal.json()["mode"] == "personalised"
    assert all(item["movie_id"] not in {1, 2, 3, 4, 5} for item in personal.json()["results"])

    assert client.delete("/users/me", headers=headers).status_code == 200
    url = "sqlite:///" + (tmp_path / "app.db").resolve().as_posix()
    session = make_session_factory(make_engine(url))()
    assert session.scalar(select(func.count()).select_from(User)) == 0
    assert session.scalar(select(func.count()).select_from(Rating)) == 0
    assert session.scalar(select(func.count()).select_from(WatchlistItem)) == 0
    assert session.scalar(select(func.count()).select_from(Interaction)) == 0
    session.close()
