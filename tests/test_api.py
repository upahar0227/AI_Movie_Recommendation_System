from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from src.recsys.cf_model import fit_collaborative
from src.recsys.content_model import ContentModel
from src.recsys.serving import RecommenderService


def _service() -> RecommenderService:
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "tmdbId": [10, 20, 30, 40],
            "title": ["Space Story", "Space War", "Love Story", "War Drama"],
            "genre_names": ["Sci-Fi", "Sci-Fi", "Romance", "Drama"],
            "tags": ["space story hero", "space war hero", "love romance", "war drama"],
            "poster_url": [None, None, None, None],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [7, 7, 7, 8, 8, 8],
            "movieId": [1, 2, 3, 1, 2, 4],
            "rating": [5, 4, 1, 5, 5, 3],
        }
    )
    content = ContentModel().fit(movies)
    collaborative = fit_collaborative(movies, ratings)
    return RecommenderService(
        movies=movies,
        content=content,
        collaborative=collaborative,
        sentiment=[0.2, 0.8, 0.4, 0.5],
        histories={7: [(1, 5.0), (2, 4.0), (3, 1.0)], 8: [(1, 5.0), (2, 5.0), (4, 3.0)]},
        hybrid_k=10,
        hybrid_beta=0.15,
        reception_method="bayesian_average_movielens",
    )


def test_api_search_similar_user_and_profile() -> None:
    client = TestClient(create_app(_service()))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["movies"] == 4

    search = client.get("/movies/search", params={"q": "space"})
    assert search.status_code == 200
    assert {item["movie_id"] for item in search.json()["results"]} == {1, 2}

    similar = client.get("/recommendations/similar/1", params={"n": 2})
    assert similar.status_code == 200
    assert len(similar.json()["results"]) == 2
    assert similar.json()["results"][0]["explanation"] == "similar genres and cast"

    missing = client.get("/recommendations/similar/99")
    assert missing.status_code == 404

    user = client.get("/recommendations/user/7")
    assert user.status_code == 200
    assert user.json()["results"]
    assert client.get("/recommendations/user/99").status_code == 404

    profile = client.post(
        "/recommendations/profile",
        json={"ratings": [{"movie_id": 1, "rating": 5}]},
    )
    assert profile.status_code == 200
    assert profile.json()["results"][0]["movie_id"] != 1

    empty = client.post("/recommendations/profile", json={"ratings": []})
    assert empty.status_code == 200
    assert empty.json()["results"][0]["explanation"] == "popular with viewers"

    invalid = client.post("/recommendations/profile", json={"ratings": [{"movie_id": 1, "rating": 9}]})
    assert invalid.status_code == 422
