from __future__ import annotations

import numpy as np
from nltk.corpus import movie_reviews
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .config import SEED
from .preprocess import tokenize


def load_imdb_style_reviews() -> tuple[list[str], np.ndarray]:
    """50/50 labelled polar reviews from NLTK (proxy for the IMDb 50k corpus)."""
    texts: list[str] = []
    labels: list[int] = []
    for category in movie_reviews.categories():
        for file_id in movie_reviews.fileids(category):
            texts.append(movie_reviews.raw(file_id))
            labels.append(1 if category == "pos" else 0)
    return texts, np.asarray(labels)


def _pipeline(model) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=tokenize,
                    token_pattern=None,
                    min_df=2,
                    max_features=20_000,
                ),
            ),
            ("clf", model),
        ]
    )


def build_sentiment_models() -> dict[str, Pipeline]:
    return {
        "naive_bayes": _pipeline(MultinomialNB()),
        "log_reg": _pipeline(LogisticRegression(max_iter=400, random_state=SEED)),
        "random_forest": _pipeline(
            RandomForestClassifier(
                n_estimators=120,
                max_depth=25,
                n_jobs=-1,
                random_state=SEED,
            )
        ),
    }


def evaluate_sentiment(seed: int = SEED) -> dict:
    texts, labels = load_imdb_style_reviews()
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    results = {}
    fitted = {}
    for name, pipe in build_sentiment_models().items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_pred = cross_val_predict(pipe, x_train, y_train, cv=cv)
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        results[name] = {
            "cv_f1": float(f1_score(y_train, cv_pred)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred)),
            "recall": float(recall_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        fitted[name] = pipe

    best_name = max(results, key=lambda key: results[key]["f1"])
    return {"metrics": results, "best": best_name, "models": fitted}


def fit_production_sentiment():
    """Logistic regression on the labelled review corpus — fast path for the app."""
    texts, labels = load_imdb_style_reviews()
    pipe = build_sentiment_models()["log_reg"]
    pipe.fit(texts, labels)
    return pipe


class MovieSentimentIndex:
    def __init__(self, scores: dict[int, float]):
        self.scores = scores

    def score(self, movie_id: int) -> float:
        return float(self.scores.get(int(movie_id), 0.5))


def movie_sentiment_scores(data, pipeline: Pipeline) -> MovieSentimentIndex:
    texts = (data.movies["soup"]).tolist()
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(texts)[:, 1]
    else:
        proba = pipeline.predict(texts).astype(float)
    scores = {
        int(movie_id): float(p)
        for movie_id, p in zip(data.movies["movieId"], proba)
    }
    return MovieSentimentIndex(scores)
