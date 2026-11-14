"""Community scores: TMDB first, MovieLens fallback, null when nobody voted."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from src.recsys.community_ratings import community_table
from src.recsys.serving import _community
from tests.test_api import _service


def _table() -> pd.DataFrame:
    tmdb = pd.DataFrame(
        {
            "id": [10, 20, 30],
            "vote_average": [8.0, 10.0, 0.0],
            "vote_count": [100, 2, 0],
        }
    )
    links = pd.DataFrame({"movieId": [1, 2, 3], "tmdbId": [10, 20, 30]})
    ratings = pd.DataFrame(
        {
            "movieId": [1] * 10 + [2] * 20,
            "rating": [4.0] * 10 + [3.0] * 20,
        }
    )
    return community_table(tmdb, links, ratings, prior_m=50, min_tmdb_votes=10)


def test_tmdb_is_primary_and_scaled() -> None:
    row = _table().set_index("movieId").loc[1]
    assert row["rating_source"] == "tmdb"
    assert row["rating_scale"] == 10
    assert row["tmdb_vote_average"] == 8.0
    assert row["tmdb_vote_count"] == 100
    assert row["movielens_average"] == 8.0
    assert row["avg_rating"] == 8.3
    assert row["rating_count"] == 100


def test_low_tmdb_count_falls_back_to_movielens() -> None:
    row = _table().set_index("movieId").loc[2]
    assert row["rating_source"] == "movielens"
    assert row["movielens_average"] == 6.0
    assert row["movielens_count"] == 20
    assert row["tmdb_vote_count"] == 2
    assert row["avg_rating"] != 10.0
    assert row["rating_count"] == 20


def test_missing_votes_are_null_not_zero() -> None:
    card = _community(_table().set_index("movieId").loc[3])
    assert card["avg_rating"] is None
    assert card["rating_count"] is None
    assert card["rating_source"] is None
    assert card["tmdb_vote_count"] is None
    assert card["movielens_count"] is None
    assert 0 not in card.values()


def test_search_keeps_community_and_user_ratings_separate() -> None:
    client = TestClient(create_app(_service()))
    payload = client.get("/movies/search", params={"q": "space"}).json()["results"][0]
    assert payload["avg_rating"] is None
    assert payload["rating_count"] is None
    assert payload["rating_source"] is None
    assert payload["rating_scale"] == 10
    assert "user_rating" not in payload
    similar = client.get("/recommendations/similar/1").json()["results"][0]
    assert similar["avg_rating"] is None
    assert "user_rating" not in similar
    profile = client.post("/recommendations/profile", json={"ratings": [{"movie_id": 1, "rating": 5}]}).json()["results"][0]
    assert profile["avg_rating"] is None
    assert "user_rating" not in profile
