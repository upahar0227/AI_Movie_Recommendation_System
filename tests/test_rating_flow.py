"""Ratings from search, model gaps, and dislike weighting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.limiter import limiter
from api.main import create_app
from src.recsys.cf_model import fit_collaborative
from src.recsys.content_model import ContentModel
from src.recsys.serving import RecommenderService
from tests.test_accounts import _login
from tests.test_schema import _upgrade


def _movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title": ["Space One", "Space Two", "Love One", "Love Two"],
            "genre_names": ["Sci-Fi", "Sci-Fi", "Romance", "Romance"],
            "tags": ["space alpha", "space alpha", "romance beta", "romance beta"],
            "poster_url": [None] * 4,
        }
    )


def _service(collaborative_movies: pd.DataFrame | None = None) -> RecommenderService:
    movies = _movies()
    trained = movies if collaborative_movies is None else collaborative_movies
    ratings = pd.DataFrame(
        {"userId": [1, 1, 1], "movieId": [1, 2, 3], "rating": [5.0, 4.0, 3.0]}
    )
    ratings = ratings.loc[ratings["movieId"].isin(set(trained["movieId"]))].reset_index(drop=True)
    return RecommenderService(
        movies=movies,
        content=ContentModel().fit(movies),
        collaborative=fit_collaborative(trained, ratings),
        sentiment=[0.2, 0.4, 0.6, 0.8][: len(trained)],
        histories={},
        hybrid_k=5,
        hybrid_beta=0.0,
        reception_method="bayesian_average_movielens",
    )


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, service: RecommenderService) -> TestClient:
    monkeypatch.setenv("JWT_SECRET", "test-secret-value")
    url = _upgrade(tmp_path, monkeypatch)
    return TestClient(create_app(service, database_url=url, enforce_secret=False))


def _headers(client: TestClient) -> dict:
    limiter.reset()
    client.post("/auth/register", json={"email": "fan@example.com", "username": "fan", "password": "correct-horse"})
    token = _login(client, "fan@example.com", "correct-horse")["access_token"]
    return {"Authorization": "Bearer " + token}


def test_search_rating_is_saved_and_returned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch, _service())
    headers = _headers(client)
    found = client.get("/movies/search", params={"q": "Space"}, headers=headers)
    assert found.status_code == 200
    movie_id = found.json()["results"][0]["movie_id"]
    assert found.json()["results"][0]["user_rating"] is None
    saved = client.post("/ratings", headers=headers, json={"movie_id": movie_id, "rating": 4.5})
    assert saved.status_code == 200
    again = client.post("/ratings", headers=headers, json={"movie_id": movie_id, "rating": 5})
    assert again.status_code == 200
    listed = client.get("/ratings/me", headers=headers).json()["ratings"]
    assert len(listed) == 1
    refreshed = client.get("/movies/search", params={"q": "Space"}, headers=headers).json()["results"]
    match = next(item for item in refreshed if item["movie_id"] == movie_id)
    assert match["user_rating"] == 5.0


def test_rating_outside_collaborative_model_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trained = _movies().loc[_movies()["movieId"] != 4].reset_index(drop=True)
    client = _client(tmp_path, monkeypatch, _service(trained))
    headers = _headers(client)
    saved = client.post("/ratings", headers=headers, json={"movie_id": 4, "rating": 4})
    assert saved.status_code == 200
    recommendations = client.get("/recommendations/me", headers=headers)
    assert recommendations.status_code == 200
    assert all(item["movie_id"] != 4 for item in recommendations.json()["results"])


def test_recommendations_change_and_exclude_rated_movies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch, _service())
    headers = _headers(client)
    client.post("/genres", headers=headers, json={"genres": ["Sci-Fi"]})
    before = [item["movie_id"] for item in client.get("/recommendations/me", headers=headers).json()["results"]]
    client.post("/ratings", headers=headers, json={"movie_id": 1, "rating": 5})
    payload = client.get("/recommendations/me", headers=headers).json()
    after = [item["movie_id"] for item in payload["results"]]
    assert 1 not in after
    assert before != after
    assert any(item["explanation"].startswith("Because you rated") for item in payload["results"])


def test_disliked_movies_lower_content_scores() -> None:
    model = ContentModel().fit(_movies())
    liked_only = model.score_profile([(1, 5.0)])
    with_dislike = model.score_profile([(1, 5.0), (3, 1.0)], dislike_weight=1.0)
    love_two = model.id_to_idx[4]
    assert with_dislike[love_two] < liked_only[love_two]
