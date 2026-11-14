from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MOVIELENS_DIR = RAW_DIR / "ml-latest-small"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

SEED = 42
K = 10
N_LATENT = 40
N_NEIGHBORS = 40
MIN_RATING_RELEVANT = 4.0
TEST_FRACTION = 0.2

HYBRID_WEIGHTS = {"content": 0.40, "svd": 0.60, "sentiment": 0.0}
HYBRID_SENTIMENT_WEIGHTS = {"content": 0.35, "svd": 0.50, "sentiment": 0.15}
