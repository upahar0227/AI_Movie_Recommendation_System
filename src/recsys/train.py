"""Preprocess, train, evaluate, and save the serving artifact.

Run with ``python -m src.recsys.train``.
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict

import pandas as pd

from .config import ARTIFACT_DIR, PROCESSED_DIR, RESULTS_DIR
from .content_model import fit_content
from .data_loader import load_catalog
from .evaluation import run_evaluation
from .preprocessing import add_tags
from .cf_model import fit_collaborative
from .sentiment import movie_reception_scores, train_sentiment
from .serving import save_artifact

logger = logging.getLogger(__name__)


def _histories(ratings: pd.DataFrame) -> dict[int, list[tuple[int, float]]]:
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for user_id, movie_id, rating in ratings[["userId", "movieId", "rating"]].itertuples(index=False):
        grouped[int(user_id)].append((int(movie_id), float(rating)))
    return grouped


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    movies = add_tags(catalog.movies)
    movies.to_csv(PROCESSED_DIR / "movies.csv", index=False)
    pd.DataFrame(
        [
            {
                "movies": catalog.n_movies,
                "ratings": catalog.n_ratings,
                "reviews": len(catalog.reviews),
            }
        ]
    ).to_csv(RESULTS_DIR / "data_summary.csv", index=False)

    sentiment_report = train_sentiment(catalog.reviews)
    content = fit_content(movies)
    evaluation = run_evaluation(movies, catalog.ratings, content)
    params = evaluation["params"].iloc[0]
    collaborative = fit_collaborative(movies, catalog.ratings)
    reception, method = movie_reception_scores(catalog.ratings, movies, sentiment_report.best_model)
    artifact = {
        "movies": movies,
        "content": content,
        "collaborative": collaborative,
        "sentiment": reception,
        "histories": _histories(catalog.ratings),
        "hybrid_k": float(params["hybrid_k"]),
        "hybrid_beta": float(params["hybrid_beta"]),
        "reception_method": method,
        "sentiment_model_name": sentiment_report.best_name,
    }
    save_artifact(ARTIFACT_DIR / "recommender.joblib", artifact)
    logger.info("Training and evaluation finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
