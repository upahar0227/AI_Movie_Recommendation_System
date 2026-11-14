from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.config import MOVIELENS_DIR, MOVIELENS_URL, RAW_DIR  # noqa: E402


NLTK_PACKAGES = [
    "movie_reviews",
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
]


def download_movielens() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    marker = MOVIELENS_DIR / "ratings.csv"
    if marker.exists():
        print(f"MovieLens already present at {MOVIELENS_DIR}")
        return MOVIELENS_DIR
    zip_path = RAW_DIR / "ml-latest-small.zip"
    print(f"Downloading MovieLens latest-small from GroupLens...")
    urlretrieve(MOVIELENS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(RAW_DIR)
    print(f"Extracted to {MOVIELENS_DIR}")
    return MOVIELENS_DIR


def download_nltk() -> None:
    import nltk

    for package in NLTK_PACKAGES:
        nltk.download(package)


if __name__ == "__main__":
    download_movielens()
    download_nltk()
    print("Data ready.")
