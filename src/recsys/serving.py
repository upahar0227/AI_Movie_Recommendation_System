"""Load trained artifacts and answer recommendation requests."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from .cf_model import CollaborativeModels
from .config import ARTIFACT_DIR, CLICK_SIGNAL_WEIGHT, CONTENT_DISLIKE_WEIGHT, MIN_RATING_LIKED
from .content_model import ContentModel
from .hybrid import adaptive_alpha, hybrid_scores
from .posters import enrich_display

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """One movie card returned to the site, including score and display text."""

    movie_id: int
    title: str
    genres: str
    score: float
    explanation: str
    poster_url: str | None
    overview: str = ""
    director: str = ""
    lead: str = ""
    link: str | None = None
    avg_rating: float | None = None
    rating_count: int | None = None
    rating_source: str | None = None
    rating_scale: int = 10
    tmdb_vote_average: float | None = None
    tmdb_vote_count: int | None = None
    movielens_average: float | None = None
    movielens_count: int | None = None
    source: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        """Return the JSON fields for this card.

        Returns:
            Movie id, title, genres, rounded score, explanation, poster, credits,
            community-rating fields, source, and reason. An empty reason uses explanation.
        """
        return {
            "movie_id": self.movie_id,
            "title": self.title,
            "genres": self.genres,
            "score": self.score,
            "explanation": self.explanation,
            "poster_url": self.poster_url,
            "overview": self.overview,
            "director": self.director,
            "lead": self.lead,
            "link": self.link,
            "avg_rating": self.avg_rating,
            "rating_count": self.rating_count,
            "rating_source": self.rating_source,
            "rating_scale": self.rating_scale,
            "tmdb_vote_average": self.tmdb_vote_average,
            "tmdb_vote_count": self.tmdb_vote_count,
            "movielens_average": self.movielens_average,
            "movielens_count": self.movielens_count,
            "source": self.source,
            "reason": self.reason or self.explanation,
        }


@dataclass
class RecommenderService:
    """In-memory models used by the API."""

    movies: pd.DataFrame
    content: ContentModel
    collaborative: CollaborativeModels
    sentiment: np.ndarray
    histories: dict[int, list[tuple[int, float]]]
    hybrid_k: float
    hybrid_beta: float
    reception_method: str

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Find titles that contain the query, case-insensitively.

        Args:
            query: Substring matched against title. Blank text returns no rows.
            limit: Maximum cards. Matching is not a regex.

        Returns:
            Summary dicts in catalog order, not ranked by similarity.
        """
        text = query.strip().lower()
        if not text:
            return []
        matches = self.movies.loc[self.movies["title"].str.lower().str.contains(text, regex=False)].head(limit)
        return [self._summary(row) for _, row in matches.iterrows()]

    def similar(self, movie_id: int, n: int = 10) -> list[Recommendation]:
        """Return content neighbours of one movie.

        Args:
            movie_id: Query MovieLens id.
            n: Maximum neighbours.

        Returns:
            Cards explained as "similar genres and cast".

        Raises:
            KeyError: movie_id is not in the content model.
        """
        if int(movie_id) not in self.content.id_to_idx:
            raise KeyError(f"Unknown movie id {movie_id}")
        pairs = self.content.recommend_similar(int(movie_id), n)
        return [self._recommendation(movie_id, score, "similar genres and cast") for movie_id, score in pairs]

    def for_user(self, user_id: int, n: int = 10) -> list[Recommendation]:
        """Recommend from a MovieLens user's saved training history.

        Args:
            user_id: MovieLens user id stored in the artifact.
            n: Maximum cards.

        Returns:
            Hybrid cards that exclude movies already in that history.

        Raises:
            KeyError: user_id is not in the artifact histories.
        """
        history = self.histories.get(int(user_id))
        if history is None:
            raise KeyError(f"Unknown user id {user_id}")
        return self._from_ratings(history, n)

    def catalog_genres(self) -> list[str]:
        """Return the unique genre names stored on the movie frame.

        Returns:
            Sorted genre strings. Blank parts and the text "nan" are dropped.
        """
        names: set[str] = set()
        for raw in self.movies["genre_names"].fillna(""):
            for part in str(raw).split(","):
                name = part.strip()
                if name and name.lower() != "nan":
                    names.add(name)
        return sorted(names)

    def top_rated_for_genres(self, genres: list[str], exclude: set[int], n: int = 10) -> list[Recommendation]:
        """Return the highest average MovieLens ratings among movies in the chosen genres.

        Args:
            genres: Genre names, matched case-insensitively. Any overlap counts.
            exclude: Movie ids to skip.
            n: Maximum cards.

        Returns:
            Cards ordered by item mean, then by rating count. The explanation names the genres.
        """
        wanted = {genre.casefold() for genre in genres}
        label = ", ".join(genres)
        by_id = self.movies.set_index("movieId")
        ranked: list[tuple[int, float, float]] = []
        for index, movie_id in enumerate(self.collaborative.movie_ids):
            movie_id = int(movie_id)
            if movie_id in exclude or movie_id not in by_id.index:
                continue
            names = [part.strip().casefold() for part in str(by_id.loc[movie_id, "genre_names"]).split(",")]
            if not wanted.intersection(names):
                continue
            ranked.append((movie_id, float(self.collaborative.item_mean[index]), float(self.collaborative.popularity[index])))
        ranked.sort(key=lambda item: (item[1], item[2]), reverse=True)
        return [
            self._recommendation(movie_id, score, f"Top rated in {label}")
            for movie_id, score, _count in ranked[:n]
        ]

    def describe(self, movie_id: int) -> dict:
        """Return title, genres, and display fields for a catalog id.

        Used by ratings and the watchlist. An unknown id still returns a card
        titled "Movie {id}" with empty community scores.

        Args:
            movie_id: MovieLens id.

        Returns:
            The same summary dict shape as search results.
        """
        matches = self.movies.loc[self.movies["movieId"] == int(movie_id)]
        if matches.empty:
            return {
                "movie_id": int(movie_id),
                "title": f"Movie {movie_id}",
                "genres": "",
                "poster_url": None,
                "overview": "",
                "director": "",
                "lead": "",
                "link": None,
                **_empty_community(),
            }
        return self._summary(matches.iloc[0])

    def for_profile(
        self,
        ratings: list[tuple[int, float]],
        n: int = 10,
        clicks: list[int] | None = None,
    ) -> list[Recommendation]:
        """Recommend from an account's ratings, ignoring ids outside the catalog.

        Args:
            ratings: (movie id, star rating) pairs.
            n: Maximum cards.
            clicks: Movie ids added to the content profile. None means no clicks.

        Returns:
            Hybrid cards, or popularity cards when there are no known ratings and no clicks.
        """
        known = [
            (int(movie_id), float(rating))
            for movie_id, rating in ratings
            if int(movie_id) in self.content.id_to_idx or int(movie_id) in set(self.collaborative.movie_ids.tolist())
        ]
        return self._from_ratings(known, n, clicks or [])

    def _from_ratings(
        self,
        ratings: list[tuple[int, float]],
        n: int,
        clicks: list[int] | None = None,
    ) -> list[Recommendation]:
        """Blend content, SVD, and reception, or fall back to popularity.

        Args:
            ratings: Known (movie id, rating) pairs. They are excluded from the list.
            n: Maximum cards.
            clicks: Clicked ids. They change content scores only when the list is non-empty.

        Returns:
            Popularity cards when both inputs are empty. Content-only cards when
            there are clicks but no ratings. Otherwise sentiment-aware hybrid cards.
        """
        popularity = self.collaborative.popularity_scores()
        if not ratings and not clicks:
            ranked = _top(popularity, self.collaborative.movie_ids, set(), n)
            return [self._recommendation(movie_id, score, "popular with viewers") for movie_id, score in ranked]
        content_scores = self._align_content(
            self.content.score_profile(
                ratings,
                clicks=clicks,
                click_weight=CLICK_SIGNAL_WEIGHT if clicks else 0.0,
                dislike_weight=CONTENT_DISLIKE_WEIGHT,
            )
        )
        svd_scores = self.collaborative.svd_scores(ratings)
        if not ratings:
            blended = content_scores
        else:
            blended = hybrid_scores(
                collaborative=svd_scores,
                content=content_scores,
                popularity=popularity,
                sentiment=np.asarray(self.sentiment, dtype=np.float64),
                n_ratings=len(ratings),
                k=self.hybrid_k,
                beta=self.hybrid_beta,
                use_sentiment=True,
            )
        exclude = {int(movie_id) for movie_id, _rating in ratings}
        ranked = _top(blended, self.collaborative.movie_ids, exclude, n)
        return [self._recommendation(movie_id, score, self._explain(movie_id, ratings)) for movie_id, score in ranked]

    def _align_content(self, scores: np.ndarray) -> np.ndarray:
        """Put content scores in collaborative-model order. Movies missing there stay 0.

        Args:
            scores: Content scores in content-model order, or already aligned.

        Returns:
            Array of shape (n_collaborative_movies,). The input is reused when the
            two movie-id sequences are identical.
        """
        if len(scores) == len(self.collaborative.movie_ids):
            same = all(int(movie_id) in self.content.id_to_idx for movie_id in self.collaborative.movie_ids)
            if same and list(self.collaborative.movie_ids) == [int(movie_id) for movie_id in self.content.movie_ids]:
                return scores
        aligned = np.zeros(len(self.collaborative.movie_ids), dtype=np.float64)
        for index, movie_id in enumerate(self.collaborative.movie_ids):
            content_index = self.content.id_to_idx.get(int(movie_id))
            if content_index is not None:
                aligned[index] = scores[content_index]
        return aligned

    def _explain(self, movie_id: int, ratings: list[tuple[int, float]]) -> str:
        """Explain a card from the closest liked movie, or from the blend weight.

        Args:
            movie_id: Recommended movie.
            ratings: Ratings used to build the profile. Only scores of 4 or more
                compete as the "because you rated" title.

        Returns:
            "Because you rated {title}" when a liked movie is similar. Otherwise
            genres and cast when alpha is below 0.5, or "highly rated by users".
            A positive beta appends an audience-reception clause when ratings exist.
        """
        liked = [int(liked_id) for liked_id, rating in ratings if rating >= MIN_RATING_LIKED and int(liked_id) in self.content.id_to_idx]
        target = self.content.id_to_idx.get(int(movie_id))
        best_id = None
        best_score = 0.0
        if target is not None:
            for liked_id in liked:
                score = float(self.content.similar_scores(liked_id)[target])
                if score > best_score:
                    best_score = score
                    best_id = liked_id
        if best_id is not None:
            title = self._title(best_id)
            return f"Because you rated {title}"
        alpha = adaptive_alpha(len(ratings), self.hybrid_k)
        if not ratings or alpha < 0.5:
            text = "similar genres and cast"
        else:
            text = "highly rated by users"
        if self.hybrid_beta > 0 and ratings:
            text = f"{text}, adjusted for audience reception"
        return text

    def _title(self, movie_id: int) -> str:
        """Return the catalog title, or "movie {id}" when the id is absent.

        Args:
            movie_id: MovieLens id.

        Returns:
            The title string.
        """
        matches = self.movies.loc[self.movies["movieId"] == int(movie_id)]
        if matches.empty:
            return f"movie {movie_id}"
        return str(matches.iloc[0]["title"])

    def _summary(self, row: pd.Series) -> dict:
        """Build the display dict for one movie row.

        Args:
            row: A movies-frame row with movieId, title, and optional display columns.

        Returns:
            movie_id, title, genres, poster, credits, link, and community fields.
        """
        return {
            "movie_id": int(row["movieId"]),
            "title": str(row["title"]),
            "genres": "" if pd.isna(row.get("genre_names")) else str(row.get("genre_names", "")),
            "poster_url": _text(row.get("poster_url")),
            "overview": _text(row.get("overview")) or "",
            "director": _text(row.get("director")) or "",
            "lead": _text(row.get("lead")) or "",
            "link": _text(row.get("link")),
            **_community(row),
        }

    def _recommendation(self, movie_id: int, score: float, explanation: str, source: str = "", reason: str = "") -> Recommendation:
        """Build a card from a catalog row and a score.

        Args:
            movie_id: MovieLens id that must exist in the movies frame.
            score: Ranker score, stored rounded to 4 decimals.
            explanation: Text used when reason is empty.
            source: Short source label. reason: display sentence; empty copies explanation.

        Returns:
            A Recommendation filled from the movie row.
        """
        row = self.movies.loc[self.movies["movieId"] == int(movie_id)].iloc[0]
        card = self._summary(row)
        return Recommendation(
            movie_id=card["movie_id"],
            title=card["title"],
            genres=card["genres"],
            score=round(float(score), 4),
            explanation=explanation,
            poster_url=card["poster_url"],
            overview=card["overview"],
            director=card["director"],
            lead=card["lead"],
            link=card["link"],
            avg_rating=card["avg_rating"],
            rating_count=card["rating_count"],
            rating_source=card["rating_source"],
            rating_scale=card["rating_scale"],
            tmdb_vote_average=card["tmdb_vote_average"],
            tmdb_vote_count=card["tmdb_vote_count"],
            movielens_average=card["movielens_average"],
            movielens_count=card["movielens_count"],
            source=source,
            reason=reason or explanation,
        )


def _empty_community() -> dict:
    """Return community-rating fields with no score.

    Returns:
        Null averages and counts, rating_scale 10, and a null source.
    """
    return {
        "avg_rating": None,
        "rating_count": None,
        "rating_source": None,
        "rating_scale": 10,
        "tmdb_vote_average": None,
        "tmdb_vote_count": None,
        "movielens_average": None,
        "movielens_count": None,
    }


def _community(row: pd.Series) -> dict:
    """Read community-rating columns from a movie row.

    Args:
        row: A row that may omit any of the community columns.

    Returns:
        The community dict. A missing scale becomes 10. Missing numbers stay None.
    """
    payload = _empty_community()
    payload["avg_rating"] = _number(row.get("avg_rating"))
    payload["rating_count"] = _count(row.get("rating_count"))
    payload["rating_source"] = _text(row.get("rating_source"))
    scale = _count(row.get("rating_scale"))
    payload["rating_scale"] = 10 if scale is None else scale
    payload["tmdb_vote_average"] = _number(row.get("tmdb_vote_average"))
    payload["tmdb_vote_count"] = _count(row.get("tmdb_vote_count"))
    payload["movielens_average"] = _number(row.get("movielens_average"))
    payload["movielens_count"] = _count(row.get("movielens_count"))
    return payload


def _number(value: object) -> float | None:
    """Convert a cell to float, treating missing values as None.

    Args:
        value: A number, None, or a pandas missing value.

    Returns:
        The float, or None when the cell is missing.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _count(value: object) -> int | None:
    """Convert a cell to int after the same missing-value check as _number.

    Args:
        value: A count cell.

    Returns:
        The truncated int, or None when _number returns None.
    """
    number = _number(value)
    if number is None:
        return None
    return int(number)


def _text(value: object) -> str | None:
    """Return stripped text, or None for missing, blank, or "nan" cells.

    Args:
        value: A display cell.

    Returns:
        The stripped string, or None.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _top(scores: np.ndarray, movie_ids: np.ndarray, exclude: set[int], limit: int) -> list[tuple[int, float]]:
    """Return the highest scores, skipping excluded ids.

    Args:
        scores: One score per movie, aligned with movie_ids.
        movie_ids: Catalog ids.
        exclude: Ids to skip.
        limit: Maximum pairs.

    Returns:
        (movie id, score) pairs, highest score first.
    """
    order = np.argsort(-scores)
    picked: list[tuple[int, float]] = []
    for index in order:
        movie_id = int(movie_ids[index])
        if movie_id in exclude:
            continue
        picked.append((movie_id, float(scores[index])))
        if len(picked) >= limit:
            break
    return picked


def save_artifact(path, payload: dict) -> None:
    """Write the trained models with joblib.

    Args:
        path: Destination file. Its parent directory is created when missing.
        payload: Artifact dict consumed by load_service.

    Side effects:
        Creates the parent directory, overwrites the file, and logs the path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    logger.info("Saved artifact to %s", path)


def load_service(path=None) -> RecommenderService:
    """Load the joblib artifact and attach display fields.

    Args:
        path: Artifact file. None uses data/processed/artifacts/recommender.joblib.

    Returns:
        A service whose movie frame includes posters, credits, and community scores.

    Raises:
        FileNotFoundError: The artifact file does not exist.

    Side effects:
        Reads the artifact and the raw csv files used by enrich_display.
    """
    artifact_path = path or (ARTIFACT_DIR / "recommender.joblib")
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Trained artifact not found at {artifact_path}. Run python -m src.recsys.train first."
        )
    payload = joblib.load(artifact_path)
    logger.info("Loaded artifact from %s (%s)", artifact_path, payload["reception_method"])
    return RecommenderService(
        movies=enrich_display(payload["movies"]),
        content=payload["content"],
        collaborative=payload["collaborative"],
        sentiment=payload["sentiment"],
        histories=payload["histories"],
        hybrid_k=float(payload["hybrid_k"]),
        hybrid_beta=float(payload["hybrid_beta"]),
        reception_method=str(payload["reception_method"]),
    )
