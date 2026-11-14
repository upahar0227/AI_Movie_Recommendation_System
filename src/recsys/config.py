"""Paths, seed, and hyperparameters. No absolute machine paths."""

from __future__ import annotations

from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
ARTIFACT_DIR = PROCESSED_DIR / "artifacts"

TMDB_MOVIES_NAME = "tmdb_5000_movies.csv"
TMDB_CREDITS_NAME = "tmdb_5000_credits.csv"
IMDB_REVIEWS_NAME = "IMDB Dataset.csv"

K_LIST = (5, 10)
HOLDOUT_FRACTION = 0.20
N_FOLDS = 5
SIGNIFICANCE_ALPHA = 0.05
MIN_RATING_LIKED = 4.0
TFIDF_MAX_FEATURES = 5000
SVD_COMPONENTS = 40
KNN_NEIGHBORS = 40
COLD_START_K = 10
SENTIMENT_BETA = 0.15
COLD_START_RATING_COUNTS = (1, 3, 5, 10)
# Content scores subtract this times the similarity to movies rated 2 or below.
CONTENT_DISLIKE_WEIGHT = 0.5
# Clicks are not ratings. This fraction of a clicked movie's text vector is added
# to the content profile, well below the weight of a movie rated 4 or higher.
CLICK_SIGNAL_WEIGHT = 0.15
# Community score is Bayesian on a 0–10 scale. m is the prior strength.
# C is the global mean of the source that is shown (TMDB vote_average, or MovieLens mean × 2).
COMMUNITY_PRIOR_M = 50
COMMUNITY_MIN_TMDB_VOTES = 10
COMMUNITY_SCALE = 10
# Recent activity is a short window, not a new rating history.
RECENT_WINDOW_DAYS = 14
RECENT_EVENT_LIMIT = 30
SIGNAL_RATING_HIGH = 1.0
SIGNAL_RATING_LOW = -1.0
SIGNAL_RATING_MID = 0.2
SIGNAL_WATCHLIST = 0.6
SIGNAL_CLICK = 0.3
SIGNAL_SEARCH = 0.1
# Recent activity can never outweigh the long-term profile.
W_RECENT_MAX = 0.40
W_RECENT_K = 5
BASE_HALF_LIFE_DAYS = 365
RECENT_HALF_LIFE_DAYS = 7
BASE_GENRE_WEIGHT = 0.35
SNAPSHOT_BONUS = 0.05
CHURN_MAX_FRACTION = 0.40
MMR_LAMBDA = 0.7
GENRE_SELECTION_MAX = 8
