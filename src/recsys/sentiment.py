"""IMDb review sentiment and a movie-level reception score.

The IMDb file has no movie ids, so it trains the classifier only. Movie-level scores
use TMDB reviews when ``TMDB_API_KEY`` is set. Otherwise they use a Bayesian average
of MovieLens ratings. That fallback is intentional and is labeled in the saved artifact.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import json
import urllib.error
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .config import RESULTS_DIR, SEED

logger = logging.getLogger(__name__)

BAYESIAN_PRIOR_STRENGTH = 20.0
RECEPTION_FALLBACK = "bayesian_average_movielens"
RECEPTION_TMDB = "tmdb_review_sentiment"


@dataclass
class SentimentReport:
    """Held-out classification metrics for the three sentiment models."""

    metrics: pd.DataFrame
    best_name: str
    best_model: Pipeline
    method: str


def _pipelines() -> dict[str, Pipeline]:
    """Build unfitted TF-IDF classifiers for the three sentiment models.

    Returns:
        Logistic regression, multinomial naive Bayes, and random forest.
        Each vectorizer is capped at 10000 features and is not yet fit.
    """
    return {
        "logistic_regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
                ("clf", LogisticRegression(max_iter=500, random_state=SEED)),
            ]
        ),
        "multinomial_nb": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
                ("clf", MultinomialNB()),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=80,
                        max_depth=40,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def train_sentiment(reviews: pd.DataFrame) -> SentimentReport:
    """Train three classifiers on a stratified 80/20 split and keep the best F1.

    The vectorizer is fit on the training split only. Positive labels are reviews
    whose sentiment text is "positive".

    Args:
        reviews: IMDb rows with ``review`` and ``sentiment``. Nulls in either are dropped.

    Returns:
        Held-out metrics, the highest-F1 pipeline, and the MovieLens fallback method name.

    Side effects:
        Writes sentiment metric csv files and a precision-recall plot under results.
    """
    frame = reviews.dropna(subset=["review", "sentiment"]).copy()
    labels = (frame["sentiment"].str.lower() == "positive").astype(int)
    texts = frame["review"].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )
    rows = []
    fitted: dict[str, Pipeline] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    matrices: dict[str, np.ndarray] = {}
    for name, pipeline in _pipelines().items():
        logger.info("Training sentiment model: %s", name)
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        fitted[name] = pipeline
        matrices[name] = confusion_matrix(y_test, predicted, labels=[0, 1])
        precision, recall, _thresholds = precision_recall_curve(y_test, probabilities)
        curves[name] = (recall, precision)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predicted),
                "precision": precision_score(y_test, predicted, zero_division=0),
                "recall": recall_score(y_test, predicted, zero_division=0),
                "f1": f1_score(y_test, predicted, zero_division=0),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best_name = str(metrics.iloc[0]["model"])
    _save_sentiment_plots(metrics, curves, matrices, best_name)
    logger.info("Best sentiment model by F1: %s", best_name)
    return SentimentReport(
        metrics=metrics,
        best_name=best_name,
        best_model=fitted[best_name],
        method=RECEPTION_FALLBACK,
    )


def _save_sentiment_plots(
    metrics: pd.DataFrame,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    matrices: dict[str, np.ndarray],
    best_name: str,
) -> None:
    """Save sentiment metrics, the best confusion matrix, and the precision-recall plot.

    Args:
        metrics: One row per model, already sorted by the caller.
        curves: Model name to (recall, precision) arrays.
        matrices: Model name to a 2x2 confusion matrix ordered negative then positive.
        best_name: Model whose matrix is written to csv.

    Side effects:
        Creates the results directory and writes sentiment_metrics.csv,
        sentiment_confusion_matrix.csv, and sentiment_pr.png.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS_DIR / "sentiment_metrics.csv", index=False)
    confusion = matrices[best_name]
    pd.DataFrame(confusion, index=["actual_neg", "actual_pos"], columns=["pred_neg", "pred_pos"]).to_csv(
        RESULTS_DIR / "sentiment_confusion_matrix.csv"
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    for name, (recall, precision) in curves.items():
        axis.plot(recall, precision, label=name)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("IMDb sentiment precision-recall")
    axis.legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "sentiment_pr.png", dpi=120)
    plt.close(figure)


def bayesian_reception(ratings: pd.DataFrame, movie_ids: np.ndarray, prior_strength: float = BAYESIAN_PRIOR_STRENGTH) -> np.ndarray:
    """Bayesian average rating, scaled from the 0.5–5 star range into 0–1.

    This is the explicit fallback used when TMDB review text is not available.
    Unrated movies use the global mean with a count of zero.

    Args:
        ratings: Rows with movieId and rating, on the 0.5–5 star scale.
        movie_ids: Catalog ids, one output position each.
        prior_strength: Pseudo-count of the global mean. Default is 20.

    Returns:
        Array of shape (n_movies,) in 0–1, scaled as (average - 0.5) / 4.5.
    """
    grouped = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    global_mean = float(ratings["rating"].mean()) if len(ratings) else 3.0
    scores = np.empty(len(movie_ids), dtype=np.float64)
    for index, movie_id in enumerate(movie_ids):
        if movie_id in grouped.index:
            mean = float(grouped.loc[movie_id, "mean"])
            count = float(grouped.loc[movie_id, "count"])
        else:
            mean, count = global_mean, 0.0
        average = (count / (count + prior_strength)) * mean + (prior_strength / (count + prior_strength)) * global_mean
        scores[index] = (average - 0.5) / 4.5
    return np.clip(scores, 0.0, 1.0)


def tmdb_review_scores(
    movie_ids: np.ndarray,
    tmdb_ids: np.ndarray,
    model: Pipeline,
    api_key: str,
) -> np.ndarray | None:
    """Average the predicted positive probability of each movie's TMDB reviews.

    Args:
        movie_ids: Catalog ids. Only the length is used.
        tmdb_ids: TMDB ids aligned with movie_ids.
        model: Fitted sentiment pipeline with predict_proba.
        api_key: TMDB key sent as the reviews-endpoint query parameter.

    Returns:
        Array of shape (n_movies,) with NaN where a movie has no review text.
        None when any request fails or every score is NaN.

    Side effects:
        Calls the TMDB reviews API once per movie and logs progress and failures.
    """
    scores = np.full(len(movie_ids), np.nan, dtype=np.float64)
    for index, tmdb_id in enumerate(tmdb_ids):
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}/reviews?api_key={api_key}"
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8")).get("results", [])
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("TMDB reviews failed for %s: %s", tmdb_id, exc)
            return None
        texts = [str(item.get("content", "")) for item in payload if item.get("content")]
        if texts:
            scores[index] = float(model.predict_proba(texts)[:, 1].mean())
        if index and index % 250 == 0:
            logger.info("Fetched TMDB reviews for %s movies", index)
    if np.isnan(scores).all():
        return None
    return scores


def movie_reception_scores(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    model: Pipeline | None,
) -> tuple[np.ndarray, str]:
    """Prefer TMDB review sentiment. Otherwise use the MovieLens Bayesian average.

    Args:
        ratings: MovieLens ratings used for the Bayesian fallback.
        movies: Catalog with movieId and tmdbId.
        model: Fitted sentiment pipeline. Required, with TMDB_API_KEY, for the TMDB path.

    Returns:
        Scores of shape (n_movies,) and a method label: tmdb review sentiment, or
        the Bayesian MovieLens fallback. Missing TMDB scores are filled from that fallback.

    Side effects:
        Reads TMDB_API_KEY and may call the TMDB reviews API. Logs which source was used.
    """
    movie_ids = movies["movieId"].astype(int).to_numpy()
    api_key = os.environ.get("TMDB_API_KEY", "").strip()
    if api_key and model is not None:
        logger.info("TMDB_API_KEY is set; fetching review sentiment")
        fetched = tmdb_review_scores(movie_ids, movies["tmdbId"].astype(int).to_numpy(), model, api_key)
        if fetched is not None:
            fallback = bayesian_reception(ratings, movie_ids)
            filled = np.where(np.isnan(fetched), fallback, fetched)
            return filled, RECEPTION_TMDB
        logger.info("TMDB review fetch failed; using the Bayesian MovieLens fallback")
    else:
        logger.info("TMDB_API_KEY is not set; using the Bayesian MovieLens reception fallback")
    return bayesian_reception(ratings, movie_ids), RECEPTION_FALLBACK
