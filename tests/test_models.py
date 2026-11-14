from __future__ import annotations

import numpy as np
import pandas as pd

from src.recsys.cf_model import fit_collaborative
from src.recsys.content_model import ContentModel
from src.recsys.hybrid import adaptive_alpha
from src.recsys.metrics import f1_at_k, ndcg_at_k, precision_at_k, recall_at_k, rmse
from src.recsys.splits import per_user_folds


def test_precision_recall_f1_hand_computed() -> None:
    # top 3 = [1, 2, 3], relevant = {2, 4}
    # precision@3 = 1/3, recall@3 = 1/2
    # f1 = 2 * (1/3) * (1/2) / ((1/3) + (1/2)) = 0.4
    recommended = [1, 2, 3, 4]
    relevant = {2, 4}
    assert precision_at_k(recommended, relevant, 3) == 1 / 3
    assert recall_at_k(recommended, relevant, 3) == 0.5
    assert f1_at_k(recommended, relevant, 3) == 0.4


def test_ndcg_hand_computed() -> None:
    # gains A=3, B=0, C=1 at ranks 1..3
    # DCG = 3/log2(2) + 0/log2(3) + 1/log2(4) = 3.5
    # ideal gains 3, 1, 0
    # IDCG = 3 + 1/log2(3) = 3.63092975357
    # NDCG = 3.5 / 3.63092975357 = 0.963940
    score = ndcg_at_k([1, 2, 3], {1: 3.0, 2: 0.0, 3: 1.0}, 3)
    assert score == pytest_approx(0.963940)


def test_rmse_hand_computed() -> None:
    # errors 1 and 3 -> sqrt((1+9)/2) = sqrt(5)
    assert rmse([4.0, 2.0], [5.0, 5.0]) == pytest_approx(5**0.5)


def test_content_similarity_shape_and_profile() -> None:
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "tags": ["hero space battle", "space war battle", "romance love story"],
        }
    )
    model = ContentModel().fit(movies)
    assert model.similar_scores(1).shape == (3,)
    similar = model.recommend_similar(1, n=2)
    assert len(similar) == 2
    assert similar[0][0] == 2
    ranked = model.recommend_for_user([(1, 5.0)], n=2)
    assert ranked[0][0] == 2


def test_holdout_has_no_user_movie_leakage() -> None:
    rows = []
    for user_id in range(1, 8):
        for movie_id in range(1, 9):
            rows.append({"userId": user_id, "movieId": movie_id, "rating": 4.0})
    ratings = pd.DataFrame(rows)
    for train, test in per_user_folds(ratings, n_folds=5, seed=42):
        overlap = train.merge(test, on=["userId", "movieId"])
        assert overlap.empty
        assert len(train) + len(test) == len(ratings)


def test_adaptive_alpha() -> None:
    assert adaptive_alpha(0, 10) == 0.0
    assert adaptive_alpha(10, 10) == 0.5
    assert adaptive_alpha(30, 10) == 0.75


def test_knn_score_length() -> None:
    movies = pd.DataFrame({"movieId": [1, 2, 3, 4], "tags": ["a", "b", "c", "d"]})
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2, 3, 3],
            "movieId": [1, 2, 3, 1, 2, 4, 2, 3],
            "rating": [5, 4, 3, 4, 5, 2, 5, 4],
        }
    )
    model = fit_collaborative(movies, ratings)
    scores = model.knn_scores([(1, 5.0), (2, 4.0)])
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, abs=1e-4)
