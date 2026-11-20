"""Submitted searches change Home without replacing the taste profile."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.limiter import limiter
from api.main import create_app
from src.app.database import make_engine, make_session_factory
from src.app.models import SearchEvent
from src.recsys.cf_model import fit_collaborative
from src.recsys.config import S_MAX, SEARCH_MIN_SLOTS
from src.recsys.content_model import ContentModel
from src.recsys.personalised import blend_weights, search_weight
from src.recsys.search_profile import SearchHit, search_scores
from src.recsys.serving import RecommenderService
from tests.test_accounts import _login
from tests.test_schema import _upgrade


def test_search_weight_never_exceeds_the_cap() -> None:
    assert search_weight(0) == 0
    assert search_weight(1) <= S_MAX
    assert search_weight(10_000) == S_MAX
    base, recent, search = blend_weights(10_000, 10_000, recent_active=True, search_active=True)
    assert search <= S_MAX
    assert base >= 0.45
    assert base + recent + search == pytest.approx(1)


def _service() -> RecommenderService:
    movie_ids = list(range(1, 13))
    titles = ["Quiet Storm"] + [f"Drama {movie_id}" for movie_id in range(2, 9)] + ["Chase One", "Chase Two", "Orbit", "Lifeboat"]
    genres = ["Drama"] * 8 + ["Action", "Action", "Drama", "Drama"]
    leads = [""] * 8 + ["Tom Hanks", "Tom Hanks", "", ""]
    tags = [f"drama family story {movie_id}" for movie_id in range(1, 9)] + [
        "action chase hanks",
        "action chase hanks",
        "space survival drama",
        "space survival drama",
    ]
    movies = pd.DataFrame(
        {
            "movieId": movie_ids,
            "title": titles,
            "genre_names": genres,
            "lead": leads,
            "tags": tags,
            "poster_url": [None] * len(movie_ids),
        }
    )
    rows = []
    for movie_id in movie_ids:
        rows.append({"userId": 1, "movieId": movie_id, "rating": 4.0})
    return RecommenderService(
        movies=movies,
        content=ContentModel().fit(movies),
        collaborative=fit_collaborative(movies, pd.DataFrame(rows)),
        sentiment=[0.2] * len(movie_ids),
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
    client.post("/auth/register", json={"email": "search@example.com", "username": "searcher", "password": "correct-horse"})
    token = _login(client, "search@example.com", "correct-horse")["access_token"]
    return {"Authorization": "Bearer " + token}


def _ids(client: TestClient, headers: dict) -> list[int]:
    payload = client.get("/recommendations/me", params={"n": 10, "include_debug": True}, headers=headers)
    assert payload.status_code == 200
    return payload.json()


def test_a_click_outweighs_the_query_alone() -> None:
    service = _service()
    moment = datetime.now(timezone.utc)
    alone = search_scores(service, [SearchHit("Quiet Storm", moment, None, [1])], moment)
    clicked = search_scores(service, [SearchHit("Quiet Storm", moment, 9, [1])], moment)
    chase = list(service.collaborative.movie_ids).index(10)
    assert clicked[chase] > alone[chase]


def test_logged_out_search_creates_no_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    found = client.get("/movies/search", params={"q": "Quiet"})
    assert found.status_code == 200
    denied = client.post("/search/events", json={"query": "Quiet Storm", "matched_movie_ids": [1]})
    assert denied.status_code == 401
    headers = _headers(client)
    assert client.get("/users/me/searches", headers=headers).json()["searches"] == []


def test_repeat_search_updates_one_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    headers = _headers(client)
    body = {"query": "Quiet Storm", "matched_movie_ids": [1]}
    first = client.post("/search/events", headers=headers, json=body)
    second = client.post("/search/events", headers=headers, json={"query": "quiet   storm", "matched_movie_ids": [1]})
    assert first.json()["logged"] is True
    assert second.json()["deduplicated"] is True
    assert first.json()["id"] == second.json()["id"]
    listed = client.get("/users/me/searches", headers=headers).json()["searches"]
    assert len(listed) == 1
    clicked = client.post("/search/events", headers=headers, json={"query": "Quiet Storm", "clicked_movie_id": 1})
    assert clicked.json()["id"] == first.json()["id"]
    assert client.get("/users/me/searches", headers=headers).json()["searches"][0]["clicked_movie_id"] == 1
    short = client.post("/search/events", headers=headers, json={"query": "a"})
    assert short.json()["logged"] is False


def test_no_search_matches_itself_and_a_title_search_keeps_most_of_the_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    headers = _headers(client)
    client.post("/genres", headers=headers, json={"genres": ["Drama"]})
    client.post("/ratings", headers=headers, json={"movie_id": 2, "rating": 5})
    before = _ids(client, headers)
    again = _ids(client, headers)
    assert [item["movie_id"] for item in before["results"]] == [item["movie_id"] for item in again["results"]]
    assert all(item["w_search"] == 0 for item in before["results"])
    client.post("/search/events", headers=headers, json={"query": "Quiet Storm", "matched_movie_ids": [1]})
    after = _ids(client, headers)
    after_ids = [item["movie_id"] for item in after["results"]]
    before_ids = [item["movie_id"] for item in before["results"]]
    assert len(set(before_ids) & set(after_ids)) >= 10 - SEARCH_MIN_SLOTS
    assert sum(1 for movie_id in after_ids if movie_id <= 8) >= SEARCH_MIN_SLOTS
    assert any(item["reason"] == "Based on your recent activity" for item in after["results"])
    assert "search_results" not in after
    assert "Quiet Storm" not in " ".join(item["reason"] for item in after["results"])


def test_genre_actor_and_free_text_searches_surface_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    headers = _headers(client)
    client.post("/genres", headers=headers, json={"genres": ["Drama"]})
    client.post("/ratings", headers=headers, json={"movie_id": 2, "rating": 5})
    client.post("/search/events", headers=headers, json={"query": "Action", "matched_movie_ids": [9, 10]})
    action = [item["movie_id"] for item in _ids(client, headers)["results"]]
    assert {9, 10}.issubset(set(action))
    client.delete("/users/me/searches", headers=headers)
    client.post("/search/events", headers=headers, json={"query": "Tom Hanks", "matched_movie_ids": [9, 10]})
    actor = [item["movie_id"] for item in _ids(client, headers)["results"]]
    assert {9, 10}.issubset(set(actor))
    client.delete("/users/me/searches", headers=headers)
    client.post("/search/events", headers=headers, json={"query": "space survival", "matched_movie_ids": [11, 12]})
    payload = _ids(client, headers)
    text = [item["movie_id"] for item in payload["results"]]
    assert {11, 12}.issubset(set(text))
    assert "space survival" not in str(payload).lower()


def test_rated_and_watchlisted_movies_stay_out_and_clearing_removes_the_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    headers = _headers(client)
    client.post("/genres", headers=headers, json={"genres": ["Drama"]})
    client.post("/ratings", headers=headers, json={"movie_id": 9, "rating": 5})
    client.post("/watchlist/10", headers=headers)
    before = [item["movie_id"] for item in _ids(client, headers)["results"]]
    client.post("/search/events", headers=headers, json={"query": "Tom Hanks", "matched_movie_ids": [9, 10]})
    during = [item["movie_id"] for item in _ids(client, headers)["results"]]
    assert 9 not in during
    assert 10 not in during
    client.delete("/users/me/searches", headers=headers)
    after = [item["movie_id"] for item in _ids(client, headers)["results"]]
    assert after == before
    assert client.get("/users/me/searches", headers=headers).json()["searches"] == []


def test_old_search_drops_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(tmp_path, monkeypatch)
    headers = _headers(client)
    client.post("/genres", headers=headers, json={"genres": ["Drama"]})
    client.post("/ratings", headers=headers, json={"movie_id": 2, "rating": 5})
    fresh = client.post("/search/events", headers=headers, json={"query": "Action", "matched_movie_ids": [9]})
    assert 9 in [item["movie_id"] for item in _ids(client, headers)["results"]]
    session = client.app.state.session_factory()
    row = session.get(SearchEvent, fresh.json()["id"])
    row.created_at = datetime.now(timezone.utc) - timedelta(days=20)
    session.commit()
    session.close()
    faded = _ids(client, headers)
    assert all(item["w_search"] == 0 for item in faded["results"])
