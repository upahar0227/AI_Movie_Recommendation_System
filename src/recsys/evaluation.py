"""Offline comparison of popularity, content, collaborative, and hybrid rankers."""

from __future__ import annotations

import logging
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from .cf_model import CollaborativeModels, fit_collaborative
from .config import (
    COLD_START_K,
    COLD_START_RATING_COUNTS,
    K_LIST,
    MIN_RATING_LIKED,
    RESULTS_DIR,
    SEED,
    SENTIMENT_BETA,
    SIGNIFICANCE_ALPHA,
)
from .content_model import ContentModel
from .hybrid import hybrid_scores
from .metrics import coverage, f1_at_k, mae, ndcg_at_k, precision_at_k, recall_at_k, rmse
from .sentiment import bayesian_reception
from .splits import per_user_folds

logger = logging.getLogger(__name__)

RANKERS = ("popularity", "content", "item_knn", "svd", "hybrid", "hybrid_sentiment")
RATING_MODELS = ("popularity", "item_knn", "svd")
TUNE_K = (5, 10, 20)
TUNE_BETA = (0.0, 0.1, 0.15, 0.3)


def top_ids(scores: np.ndarray, movie_ids: np.ndarray, exclude: set[int], limit: int) -> list[int]:
    """Highest-scoring movies, skipping ids in ``exclude``."""
    order = np.argsort(-scores)
    picked: list[int] = []
    for index in order:
        movie_id = int(movie_ids[index])
        if movie_id in exclude:
            continue
        picked.append(movie_id)
        if len(picked) >= limit:
            break
    return picked


def _histories(ratings: pd.DataFrame) -> dict[int, list[tuple[int, float]]]:
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for user_id, movie_id, rating in ratings[["userId", "movieId", "rating"]].itertuples(index=False):
        grouped[int(user_id)].append((int(movie_id), float(rating)))
    return grouped


def _score_user(
    name: str,
    ratings: list[tuple[int, float]],
    content: ContentModel,
    collaborative: CollaborativeModels,
    sentiment: np.ndarray,
    k_param: float,
    beta: float,
) -> np.ndarray:
    popularity = collaborative.popularity_scores()
    if name == "popularity":
        return popularity
    content_scores = content.score_profile(ratings)
    if name == "content":
        return content_scores
    if name == "item_knn":
        return collaborative.knn_scores(ratings)
    svd_scores = collaborative.svd_scores(ratings)
    if name == "svd":
        return svd_scores
    return hybrid_scores(
        collaborative=svd_scores,
        content=content_scores,
        popularity=popularity,
        sentiment=sentiment,
        n_ratings=len(ratings),
        k=k_param,
        beta=beta,
        use_sentiment=name == "hybrid_sentiment",
    )


def _rating_prediction(name: str, scores: np.ndarray, collaborative: CollaborativeModels, item_index: int) -> float:
    if name == "popularity":
        return float(collaborative.item_mean[item_index])
    predicted = float(scores[item_index])
    if name == "item_knn" and predicted == 0.0:
        return float(collaborative.item_mean[item_index])
    return predicted


def tune_hybrid(
    movies: pd.DataFrame,
    train_ratings: pd.DataFrame,
    content: ContentModel,
) -> tuple[float, float]:
    """Pick k and beta on an inner hold-out carved from the training ratings only."""
    inner_train, inner_test = per_user_folds(train_ratings, n_folds=5, seed=SEED + 1)[0]
    collaborative = fit_collaborative(movies, inner_train)
    sentiment = bayesian_reception(inner_train, movies["movieId"].astype(int).to_numpy())
    histories = _histories(inner_train)
    best = (COLD_START_K, SENTIMENT_BETA, -1.0)
    for k_param in TUNE_K:
        for beta in TUNE_BETA:
            values = _ndcg_values(
                "hybrid_sentiment",
                inner_test,
                histories,
                content,
                collaborative,
                sentiment,
                k_param,
                beta,
            )
            mean_ndcg = float(np.mean(values)) if values else 0.0
            logger.info("Tune k=%s beta=%s NDCG@10=%.4f", k_param, beta, mean_ndcg)
            if mean_ndcg > best[2]:
                best = (float(k_param), float(beta), mean_ndcg)
    logger.info("Selected hybrid k=%s beta=%s", best[0], best[1])
    return best[0], best[1]


def _ndcg_values(
    model_name: str,
    test_ratings: pd.DataFrame,
    histories: dict[int, list[tuple[int, float]]],
    content: ContentModel,
    collaborative: CollaborativeModels,
    sentiment: np.ndarray,
    k_param: float,
    beta: float,
) -> list[float]:
    values: list[float] = []
    for user_id, group in test_ratings.groupby("userId"):
        relevant = group.loc[group["rating"] >= MIN_RATING_LIKED]
        if relevant.empty:
            continue
        history = histories.get(int(user_id), [])
        scores = _score_user(model_name, history, content, collaborative, sentiment, k_param, beta)
        ranked = top_ids(scores, collaborative.movie_ids, {movie_id for movie_id, _rating in history}, max(K_LIST))
        gains = {int(movie_id): float(rating) for movie_id, rating in group[["movieId", "rating"]].itertuples(index=False)}
        values.append(ndcg_at_k(ranked, gains, 10))
    return values


def evaluate_fold(
    movies: pd.DataFrame,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    content: ContentModel,
    k_param: float,
    beta: float,
) -> tuple[pd.DataFrame, dict[str, dict[int, float]], dict[str, tuple[list[float], list[float]]]]:
    """Fit collaborative models on the fold's training rows and score the test rows."""
    collaborative = fit_collaborative(movies, train_ratings)
    sentiment = bayesian_reception(train_ratings, movies["movieId"].astype(int).to_numpy())
    histories = _histories(train_ratings)
    lists: dict[str, list[list[int]]] = {name: [] for name in RANKERS}
    per_user: dict[str, dict[int, float]] = {name: {} for name in RANKERS}
    errors: dict[str, tuple[list[float], list[float]]] = {name: ([], []) for name in RATING_MODELS}
    ranking_rows = []

    test_groups = list(test_ratings.groupby("userId"))
    logger.info("Scoring %s test users", len(test_groups))
    for user_id, group in test_groups:
        history = histories.get(int(user_id), [])
        exclude = {movie_id for movie_id, _rating in history}
        gains = {int(movie_id): float(rating) for movie_id, rating in group[["movieId", "rating"]].itertuples(index=False)}
        relevant = {movie_id for movie_id, rating in gains.items() if rating >= MIN_RATING_LIKED}
        score_cache = {
            name: _score_user(name, history, content, collaborative, sentiment, k_param, beta) for name in RANKERS
        }
        if relevant:
            for name in RANKERS:
                ranked = top_ids(score_cache[name], collaborative.movie_ids, exclude, max(K_LIST))
                lists[name].append(ranked)
                per_user[name][int(user_id)] = ndcg_at_k(ranked, gains, 10)
        for name in RATING_MODELS:
            actual, predicted = errors[name]
            for movie_id, rating in gains.items():
                index = collaborative.id_to_idx.get(movie_id)
                if index is None:
                    continue
                actual.append(rating)
                predicted.append(_rating_prediction(name, score_cache[name], collaborative, index))

    catalog_size = len(collaborative.movie_ids)
    evaluated_groups = []
    for user_id, group in test_groups:
        relevant = {
            int(movie_id)
            for movie_id, rating in group[["movieId", "rating"]].itertuples(index=False)
            if float(rating) >= MIN_RATING_LIKED
        }
        if relevant:
            evaluated_groups.append((group, relevant))
    for name in RANKERS:
        row: dict[str, float | str] = {"model": name}
        for k in K_LIST:
            precision_values = []
            recall_values = []
            f1_values = []
            ndcg_values = []
            for ranked, (group, relevant) in zip(lists[name], evaluated_groups, strict=True):
                gains = {
                    int(movie_id): float(rating)
                    for movie_id, rating in group[["movieId", "rating"]].itertuples(index=False)
                }
                precision_values.append(precision_at_k(ranked, relevant, k))
                recall_values.append(recall_at_k(ranked, relevant, k))
                f1_values.append(f1_at_k(ranked, relevant, k))
                ndcg_values.append(ndcg_at_k(ranked, gains, k))
            row[f"precision@{k}"] = float(np.mean(precision_values)) if precision_values else 0.0
            row[f"recall@{k}"] = float(np.mean(recall_values)) if recall_values else 0.0
            row[f"f1@{k}"] = float(np.mean(f1_values)) if f1_values else 0.0
            row[f"ndcg@{k}"] = float(np.mean(ndcg_values)) if ndcg_values else 0.0
        row["coverage@10"] = coverage([ranked[:10] for ranked in lists[name]], catalog_size)
        ranking_rows.append(row)
    return pd.DataFrame(ranking_rows), per_user, errors


def cold_start_curve(
    movies: pd.DataFrame,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    content: ContentModel,
    k_param: float,
    beta: float,
    sample_size: int = 120,
) -> pd.DataFrame:
    """NDCG@10 when each sampled user keeps only 1, 3, 5, or 10 training ratings."""
    collaborative = fit_collaborative(movies, train_ratings)
    sentiment = bayesian_reception(train_ratings, movies["movieId"].astype(int).to_numpy())
    histories = _histories(train_ratings)
    eligible = [user_id for user_id, history in histories.items() if len(history) >= 10 and user_id in set(test_ratings["userId"].astype(int))]
    rng = np.random.default_rng(SEED)
    if len(eligible) > sample_size:
        eligible = list(rng.choice(np.array(eligible), size=sample_size, replace=False))
    logger.info("Cold-start sample: %s users", len(eligible))
    rows = []
    test_by_user = {int(user_id): group for user_id, group in test_ratings.groupby("userId")}
    for known in COLD_START_RATING_COUNTS:
        totals = {name: [] for name in RANKERS}
        for user_id in eligible:
            group = test_by_user.get(int(user_id))
            if group is None:
                continue
            gains = {int(movie_id): float(rating) for movie_id, rating in group[["movieId", "rating"]].itertuples(index=False)}
            if not any(rating >= MIN_RATING_LIKED for rating in gains.values()):
                continue
            history = histories[int(user_id)]
            picked_idx = rng.choice(len(history), size=known, replace=False)
            limited = [history[int(index)] for index in picked_idx]
            exclude = {movie_id for movie_id, _rating in limited}
            for name in RANKERS:
                scores = _score_user(name, limited, content, collaborative, sentiment, k_param, beta)
                ranked = top_ids(scores, collaborative.movie_ids, exclude, 10)
                totals[name].append(ndcg_at_k(ranked, gains, 10))
        for name in RANKERS:
            rows.append(
                {
                    "known_ratings": known,
                    "model": name,
                    "ndcg@10": float(np.mean(totals[name])) if totals[name] else 0.0,
                    "users": len(totals[name]),
                }
            )
        logger.info("Cold-start known=%s done", known)
    return pd.DataFrame(rows)


def _significance(per_user: dict[str, dict[int, list[float]]]) -> pd.DataFrame:
    hybrid_users = per_user["hybrid"]
    rows = []
    for baseline in RANKERS:
        if baseline == "hybrid":
            continue
        shared = sorted(set(hybrid_users) & set(per_user[baseline]))
        left = np.array([float(np.mean(hybrid_users[user])) for user in shared])
        right = np.array([float(np.mean(per_user[baseline][user])) for user in shared])
        difference = left - right
        if len(shared) < 10 or np.allclose(difference, 0):
            stat_w, p_w, stat_t, p_t = np.nan, np.nan, np.nan, np.nan
        else:
            stat_w, p_w = wilcoxon(left, right, alternative="two-sided")
            stat_t, p_t = ttest_rel(left, right)
        rows.append(
            {
                "comparison": f"hybrid vs {baseline}",
                "users": len(shared),
                "mean_ndcg_hybrid": float(left.mean()) if len(left) else 0.0,
                "mean_ndcg_baseline": float(right.mean()) if len(right) else 0.0,
                "wilcoxon_stat": float(stat_w),
                "wilcoxon_p": float(p_w),
                "paired_t_stat": float(stat_t),
                "paired_t_p": float(p_t),
                "significant_0.05": bool(p_w < SIGNIFICANCE_ALPHA) if p_w == p_w else False,
            }
        )
    return pd.DataFrame(rows)


def run_evaluation(movies: pd.DataFrame, ratings: pd.DataFrame, content: ContentModel) -> dict[str, pd.DataFrame]:
    """5-fold per-user evaluation, significance tests, and the cold-start curve."""
    folds = per_user_folds(ratings)
    k_param, beta = tune_hybrid(movies, folds[0][0], content)
    ranking_frames = []
    pooled: dict[str, dict[int, list[float]]] = {name: defaultdict(list) for name in RANKERS}
    error_pool: dict[str, tuple[list[float], list[float]]] = {name: ([], []) for name in RATING_MODELS}
    for fold_id, (train_ratings, test_ratings) in enumerate(folds):
        logger.info("Evaluating fold %s", fold_id)
        ranking, per_user, errors = evaluate_fold(movies, train_ratings, test_ratings, content, k_param, beta)
        ranking.insert(0, "fold", fold_id)
        ranking_frames.append(ranking)
        for name in RANKERS:
            for user_id, value in per_user[name].items():
                pooled[name][user_id].append(value)
        for name in RATING_MODELS:
            error_pool[name][0].extend(errors[name][0])
            error_pool[name][1].extend(errors[name][1])
    ranking_all = pd.concat(ranking_frames, ignore_index=True)
    summary = ranking_all.drop(columns=["fold"]).groupby("model", as_index=False).mean()
    error_rows = []
    for name in RATING_MODELS:
        actual, predicted = error_pool[name]
        error_rows.append({"model": name, "rmse": rmse(actual, predicted), "mae": mae(actual, predicted)})
    errors_frame = pd.DataFrame(error_rows)
    significance = _significance(pooled)
    cold = cold_start_curve(movies, folds[0][0], folds[0][1], content, k_param, beta)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ranking_all.to_csv(RESULTS_DIR / "ranking_metrics_folds.csv", index=False)
    summary.to_csv(RESULTS_DIR / "ranking_metrics.csv", index=False)
    errors_frame.to_csv(RESULTS_DIR / "rating_errors.csv", index=False)
    significance.to_csv(RESULTS_DIR / "significance.csv", index=False)
    cold.to_csv(RESULTS_DIR / "cold_start.csv", index=False)
    _plot_cold_start(cold)
    pd.DataFrame([{"hybrid_k": k_param, "hybrid_beta": beta}]).to_csv(RESULTS_DIR / "hybrid_params.csv", index=False)
    return {
        "ranking": summary,
        "errors": errors_frame,
        "significance": significance,
        "cold_start": cold,
        "params": pd.DataFrame([{"hybrid_k": k_param, "hybrid_beta": beta}]),
    }


def _plot_cold_start(frame: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    for model, group in frame.groupby("model"):
        ordered = group.sort_values("known_ratings")
        axis.plot(ordered["known_ratings"], ordered["ndcg@10"], marker="o", label=model)
    axis.set_xlabel("Known training ratings")
    axis.set_ylabel("NDCG@10")
    axis.set_title("Cold-start: NDCG@10 vs number of known ratings")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "cold_start.png", dpi=120)
    plt.close(figure)
