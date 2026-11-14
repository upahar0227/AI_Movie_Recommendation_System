"""Stable base recommendations mixed with a capped recent-activity profile."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.limiter import limiter
from api.main import create_app
from src.recsys.cf_model import fit_collaborative
from src.recsys.config import W_RECENT_MAX
from src.recsys.content_model import ContentModel
from src.recsys.personalised import RatedMovie, recent_weight
from src.recsys.serving import RecommenderService
from tests.test_accounts import _login
from tests.test_schema import _upgrade


def test_recent_weight_never_exceeds_the_cap() -> None:
    assert recent_weight(0) == 0
    assert recent_weight(1) < W_RECENT_MAX
    assert recent_weight(10_000) == W_RECENT_MAX


def _catalog() -> pd.DataFrame:
    drama = list(range(1, 13))
    action = list(range(13, 17))
    movie_ids = drama + action
    return pd.DataFrame(
        {
            "movieId": movie_ids,
            "title": [f"Drama {movie_id}" for movie_id in drama] + [f"Action {movie_id}" for movie_id in action],
            "genre_names": ["Drama"] * len(drama) + ["Action"] * len(action),
            "tags": [f"drama stable story {movie_id}" for movie_id in drama] + ["action chase"] * len(action),
            "poster_url": [None] * len(movie_ids),
        }
    )


def _service() -> RecommenderService:
    movies = _catalog()
    rows = []
    for movie_id in range(1, 13):
        rows.append({"userId": 1, "movieId": movie_id, "rating": 5.0 - (movie_id * 0.05)})
        rows.append({"userId": 2, "movieId": movie_id, "rating": 4.5})
    for movie_id in range(13, 17):
        rows.append({"userId": 3, "movieId": movie_id, "rating": 2.0})
    ratings = pd.DataFrame(rows)
    return RecommenderService(
        movies=movies,
        content=ContentModel().fit(movies),
        collaborative=fit_collaborative(movies, ratings),
        sentiment=[0.2] * len(movies),
        histories={},
        hybrid_k=5,
        hybrid_beta=0.0,
        reception_method="bayesian_average_movielens",
    )


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("JWT_SECRET", "test-secret-value")
    monkeypatch.setenv("RECS_DEBUG", "1")
    url = _upgrade(tmp_path, monkeypatch)
    return TestClient(create_app(_service(), database_url=url, enforce_secret=False))


def _headers(client: TestClient) -> dict:
    limiter.reset()
    client.post("/auth/register", json={"email": "blend@example.com", "username": "blend", "password": "correct-horse"})
    token = _login(client, "blend@example.com", "correct-horse")["access_token"]
    return {"Authorization": "Bearer " + token}


def test_without_recent_events_the_list_matches_the_base_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    headers = _headers(client)
    client.post("/genres", headers=headers, json={"genres": ["Drama"]})
    first = client.get("/recommendations/me", params={"n": 10, "include_debug": True}, headers=headers)
    second = client.get("/recommendations/me", params={"n": 10, "include_debug": True}, headers=headers)
    assert first.status_code == 200
    assert [item["movie_id"] for item in first.json()["results"]] == [item["movie_id"] for item in second.json()["results"]]
    assert all(item["w_recent"] == 0 for item in first.json()["results"])
    assert all(item["source"] == "genres" for item in first.json()["results"])
    assert all("Drama" in item["genres"] for item in first.json()["results"])


def test_one_rating_keeps_most_of_the_list_and_adds_its_genre(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    headers = _headers(client)
    client.post("/genres", headers=headers, json={"genres": ["Drama"]})
    before = [item["movie_id"] for item in client.get("/recommendations/me", params={"n": 10}, headers=headers).json()["results"]]
    assert len(before) == 10
    saved = client.post("/ratings", headers=headers, json={"movie_id": 13, "rating": 5})
    assert saved.status_code == 200
    payload = client.get("/recommendations/me", params={"n": 10, "include_debug": True}, headers=headers).json()
    after = [item["movie_id"] for item in payload["results"]]
    assert 13 not in after
    overlap = len(set(before) & set(after))
    assert overlap >= 6
    assert any(item["genres"] == "Action" for item in payload["results"])
    assert any(item["reason"].startswith("Because you rated") for item in payload["results"])
    assert all(item["w_recent"] <= W_RECENT_MAX for item in payload["results"])


def test_a_strong_dislike_pushes_that_genre_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    headers = _headers(client)
    client.post("/genres", headers=headers, json={"genres": ["Drama"]})
    client.get("/recommendations/me", params={"n": 10}, headers=headers)
    client.post("/ratings", headers=headers, json={"movie_id": 13, "rating": 1})
    payload = client.get("/recommendations/me", params={"n": 10, "include_debug": True}, headers=headers).json()["results"]
    assert all(item["movie_id"] != 13 for item in payload)
    assert not any(item["genres"] == "Action" for item in payload)
    assert all(item["w_recent"] <= W_RECENT_MAX for item in payload)


def test_debug_fields_stay_hidden_without_dev_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "test-secret-value")
    monkeypatch.delenv("RECS_DEBUG", raising=False)
    url = _upgrade(tmp_path, monkeypatch)
    client = TestClient(create_app(_service(), database_url=url, enforce_secret=False))
    headers = _headers(client)
    client.post("/genres", headers=headers, json={"genres": ["Drama"]})
    payload = client.get("/recommendations/me", params={"include_debug": True, "w_recent_override": 0.9}, headers=headers)
    assert "base_score" not in payload.json()["results"][0]
    assert "w_recent" not in payload.json()["results"][0]


def test_genre_edit_replaces_the_base_profile_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    headers = _headers(client)
    listed = client.get("/users/me/genres", headers=headers)
    assert listed.status_code == 200
    assert "Drama" in listed.json()["allowed"]
    assert listed.json()["genres"] == []
    saved = client.put("/users/me/genres", headers=headers, json={"genres": ["Drama", "Drama"]})
    assert saved.status_code == 200
    assert saved.json()["genres"] == ["Drama"]
    client.post("/ratings", headers=headers, json={"movie_id": 1, "rating": 5})
    client.post("/watchlist/2", headers=headers)
    changed = client.put("/users/me/genres", headers=headers, json={"genres": ["Action"]})
    assert changed.status_code == 200
    ratings = client.get("/ratings/me", headers=headers).json()["ratings"]
    watch = client.get("/watchlist/me", headers=headers).json()["watchlist"]
    assert ratings[0]["movie_id"] == 1
    assert watch[0]["movie_id"] == 2
    empty = client.put("/users/me/genres", headers=headers, json={"genres": []})
    assert empty.status_code == 422
    too_many = client.put(
        "/users/me/genres",
        headers=headers,
        json={"genres": ["Drama", "Action", "Comedy", "Horror", "War", "Western", "Romance", "Mystery", "Music"]},
    )
    assert too_many.status_code == 422
    unknown = client.put("/users/me/genres", headers=headers, json={"genres": ["Not A Genre"]})
    assert unknown.status_code == 422


def test_rating_signal_helper_matches_the_configured_weights() -> None:
    assert recent_weight(5, w_max=0.4, k=5) == pytest.approx(0.4)
    old = datetime.now(timezone.utc) - timedelta(days=400)
    assert RatedMovie(1, 5.0, old).rating == 5.0
