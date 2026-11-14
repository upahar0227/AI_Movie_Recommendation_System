"""Load MovieLens, TMDB 5000, and IMDb reviews, then keep the overlapping catalog."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import (
    IMDB_REVIEWS_NAME,
    RAW_DIR,
    TMDB_CREDITS_NAME,
    TMDB_MOVIES_NAME,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawPaths:
    """Locations of the three source datasets."""

    raw_dir: Path
    movielens_dir: Path | None
    tmdb_movies: Path
    tmdb_credits: Path
    imdb_reviews: Path

    def missing(self) -> list[Path]:
        """Return files that must exist before a real merge can run."""
        missing: list[Path] = []
        if self.movielens_dir is None:
            for name in ("ratings.csv", "movies.csv", "links.csv"):
                missing.append(self.raw_dir / name)
        if not self.tmdb_movies.is_file():
            missing.append(self.tmdb_movies)
        if not self.tmdb_credits.is_file():
            missing.append(self.tmdb_credits)
        if not self.imdb_reviews.is_file():
            missing.append(self.imdb_reviews)
        return missing


@dataclass(frozen=True)
class Catalog:
    """Movies present in both MovieLens and TMDB, plus the ratings of those movies."""

    movies: pd.DataFrame
    ratings: pd.DataFrame
    credits: pd.DataFrame
    reviews: pd.DataFrame

    @property
    def n_movies(self) -> int:
        return int(self.movies["movieId"].nunique())

    @property
    def n_ratings(self) -> int:
        return int(len(self.ratings))


def resolve_raw_paths(raw_dir: Path = RAW_DIR) -> RawPaths:
    """Find MovieLens either in ``raw_dir`` or in ``raw_dir/ml-latest-small``."""
    movielens_dir = None
    for candidate in (raw_dir, raw_dir / "ml-latest-small"):
        if all((candidate / name).is_file() for name in ("ratings.csv", "movies.csv", "links.csv")):
            movielens_dir = candidate
            break
    return RawPaths(
        raw_dir=raw_dir,
        movielens_dir=movielens_dir,
        tmdb_movies=raw_dir / TMDB_MOVIES_NAME,
        tmdb_credits=raw_dir / TMDB_CREDITS_NAME,
        imdb_reviews=raw_dir / IMDB_REVIEWS_NAME,
    )


def load_catalog(paths: RawPaths | None = None) -> Catalog:
    """Merge MovieLens to TMDB on ``tmdbId`` and drop movies missing from either side.

    IMDb reviews are loaded for the sentiment model only. They are not joined to movies,
    because that file has no movie identifier.
    """
    paths = paths or resolve_raw_paths()
    missing = paths.missing()
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Dataset files are missing. Place them here and rerun:\n" + lines
        )
    assert paths.movielens_dir is not None

    logger.info("Loading MovieLens from %s", paths.movielens_dir)
    ml_movies = pd.read_csv(paths.movielens_dir / "movies.csv")
    ratings = pd.read_csv(paths.movielens_dir / "ratings.csv")
    links = pd.read_csv(paths.movielens_dir / "links.csv")
    logger.info(
        "MovieLens before join: %s movies, %s ratings",
        ml_movies["movieId"].nunique(),
        len(ratings),
    )

    logger.info("Loading TMDB movies from %s", paths.tmdb_movies)
    tmdb = pd.read_csv(paths.tmdb_movies)
    credits = pd.read_csv(paths.tmdb_credits)
    reviews = pd.read_csv(paths.imdb_reviews)
    _require_columns(
        reviews,
        ("review", "sentiment"),
        paths.imdb_reviews,
    )

    movies = merge_movielens_tmdb(ml_movies, links, tmdb, credits)
    kept_ids = set(movies["movieId"].astype(int))
    ratings = ratings.loc[ratings["movieId"].astype(int).isin(kept_ids)].reset_index(drop=True)
    logger.info("Movies remaining after MovieLens-TMDB join: %s", movies["movieId"].nunique())
    logger.info("Ratings remaining after MovieLens-TMDB join: %s", len(ratings))
    logger.info("IMDb reviews loaded for sentiment training only: %s", len(reviews))
    return Catalog(movies=movies, ratings=ratings, credits=credits, reviews=reviews)


def merge_movielens_tmdb(
    ml_movies: pd.DataFrame,
    links: pd.DataFrame,
    tmdb_movies: pd.DataFrame,
    credits: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join MovieLens movies to TMDB through ``links.tmdbId``."""
    link_rows = links.dropna(subset=["tmdbId"]).copy()
    link_rows["tmdbId"] = link_rows["tmdbId"].astype(int)
    linked = ml_movies.merge(link_rows[["movieId", "imdbId", "tmdbId"]], on="movieId", how="inner")

    tmdb = tmdb_movies.copy()
    tmdb["id"] = tmdb["id"].astype(int)
    credit_rows = credits.copy()
    credit_rows["movie_id"] = credit_rows["movie_id"].astype(int)
    tmdb = tmdb.merge(
        credit_rows[["movie_id", "cast", "crew"]],
        left_on="id",
        right_on="movie_id",
        how="left",
    )
    tmdb["cast"] = tmdb["cast"].fillna("[]")
    tmdb["crew"] = tmdb["crew"].fillna("[]")

    merged = linked.merge(tmdb, left_on="tmdbId", right_on="id", how="inner", suffixes=("_ml", "_tmdb"))
    merged = merged.drop_duplicates(subset=["movieId"]).reset_index(drop=True)
    return merged


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        catalog = load_catalog()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    logger.info(
        "Catalog ready: %s movies, %s ratings, %s review rows",
        catalog.n_movies,
        catalog.n_ratings,
        len(catalog.reviews),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
