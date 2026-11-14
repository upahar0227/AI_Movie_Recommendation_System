from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import MOVIELENS_DIR
from .preprocess import clean_title, parse_genres, parse_year


@dataclass
class MovieLensData:
    movies: pd.DataFrame
    ratings: pd.DataFrame
    tags: pd.DataFrame
    links: pd.DataFrame
    movie_ids: np.ndarray
    movie_id_to_idx: dict[int, int] = field(repr=False)

    def movie_row(self, movie_id: int) -> pd.Series:
        return self.movies.loc[self.movies["movieId"] == movie_id].iloc[0]


def load_movielens(path=MOVIELENS_DIR) -> MovieLensData:
    movies = pd.read_csv(path / "movies.csv")
    ratings = pd.read_csv(path / "ratings.csv")
    tags = pd.read_csv(path / "tags.csv")
    links = pd.read_csv(path / "links.csv")

    movies["year"] = movies["title"].map(parse_year)
    movies["title_clean"] = movies["title"].map(clean_title)
    movies["genre_list"] = movies["genres"].map(parse_genres)
    tag_text = (
        tags.groupby("movieId")["tag"]
        .apply(lambda s: " ".join(str(x) for x in s))
        .rename("tag_text")
    )
    movies = movies.merge(tag_text, left_on="movieId", right_index=True, how="left")
    movies["tag_text"] = movies["tag_text"].fillna("")
    movies["soup"] = (
        movies["title_clean"].str.lower()
        + " "
        + movies["genre_list"].map(lambda g: " ".join(g).lower())
        + " "
        + movies["tag_text"].str.lower()
    )

    movie_ids = movies["movieId"].to_numpy()
    movie_id_to_idx = {int(mid): i for i, mid in enumerate(movie_ids)}
    return MovieLensData(
        movies=movies,
        ratings=ratings,
        tags=tags,
        links=links,
        movie_ids=movie_ids,
        movie_id_to_idx=movie_id_to_idx,
    )


def user_item_matrix(data: MovieLensData, ratings: pd.DataFrame | None = None):
    frame = data.ratings if ratings is None else ratings
    n_users = int(frame["userId"].max()) + 1
    n_items = len(data.movie_ids)
    matrix = np.zeros((n_users, n_items), dtype=np.float32)
    for user_id, movie_id, rating in frame[["userId", "movieId", "rating"]].itertuples(
        index=False
    ):
        idx = data.movie_id_to_idx.get(int(movie_id))
        if idx is not None:
            matrix[int(user_id), idx] = float(rating)
    return matrix
