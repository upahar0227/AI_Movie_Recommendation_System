"""Merge behaviour for the MovieLens-TMDB catalog. Fixtures are not the research data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.recsys.data_loader import merge_movielens_tmdb, resolve_raw_paths


def test_merge_keeps_only_movies_in_both_catalogs() -> None:
    ml_movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "title": ["Alpha (1999)", "Beta (2001)", "Gamma (2004)"],
            "genres": ["Drama", "Comedy", "Action"],
        }
    )
    links = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "imdbId": [10, 20, 30],
            "tmdbId": [100, 200, None],
        }
    )
    tmdb = pd.DataFrame(
        {
            "id": [100, 999],
            "title": ["Alpha", "Unlinked"],
            "overview": ["A film", "Not in MovieLens"],
            "genres": ["[]", "[]"],
            "keywords": ["[]", "[]"],
        }
    )
    credits = pd.DataFrame(
        {
            "movie_id": [100, 999],
            "cast": ["[]", "[]"],
            "crew": ["[]", "[]"],
        }
    )

    merged = merge_movielens_tmdb(ml_movies, links, tmdb, credits)

    assert merged["movieId"].tolist() == [1]
    assert int(merged.iloc[0]["tmdbId"]) == 100
    assert "cast" in merged.columns
    assert "overview" in merged.columns


def test_resolve_raw_paths_reports_missing_tmdb_and_imdb(tmp_path: Path) -> None:
    movielens = tmp_path / "ml-latest-small"
    movielens.mkdir()
    for name in ("ratings.csv", "movies.csv", "links.csv"):
        (movielens / name).write_text("col\n", encoding="utf-8")

    paths = resolve_raw_paths(tmp_path)

    assert paths.movielens_dir == movielens
    missing = {path.name for path in paths.missing()}
    assert missing == {"tmdb_5000_movies.csv", "tmdb_5000_credits.csv", "IMDB Dataset.csv"}


def test_load_catalog_refuses_to_invent_missing_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.recsys import data_loader

    monkeypatch.setattr(data_loader, "RAW_DIR", tmp_path)
    paths = data_loader.resolve_raw_paths(tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        data_loader.load_catalog(paths)
    message = str(exc.value)
    assert "tmdb_5000_movies.csv" in message
    assert "tmdb_5000_credits.csv" in message
    assert "IMDB Dataset.csv" in message
