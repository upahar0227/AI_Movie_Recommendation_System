"""Item-kNN and TruncatedSVD collaborative filters. No Surprise, no deep learning."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from .config import KNN_NEIGHBORS, SEED, SVD_COMPONENTS

logger = logging.getLogger(__name__)


@dataclass
class CollaborativeModels:
    """Collaborative filters aligned to one shared movie index."""

    movie_ids: np.ndarray
    id_to_idx: dict[int, int]
    user_ids: np.ndarray
    user_to_idx: dict[int, int]
    neighbor_index: np.ndarray
    neighbor_similarity: np.ndarray
    svd: TruncatedSVD
    item_factors: np.ndarray
    train_matrix: sparse.csr_matrix
    item_mean: np.ndarray
    popularity: np.ndarray

    def item_index(self, movie_id: int) -> int:
        try:
            return self.id_to_idx[int(movie_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown movie id {movie_id}") from exc

    def user_vector(self, ratings: list[tuple[int, float]]) -> np.ndarray:
        """Dense rating vector in movie-index order. Unrated movies stay 0."""
        vector = np.zeros(len(self.movie_ids), dtype=np.float64)
        for movie_id, rating in ratings:
            index = self.id_to_idx.get(int(movie_id))
            if index is not None:
                vector[index] = float(rating)
        return vector

    def knn_scores(self, ratings: list[tuple[int, float]]) -> np.ndarray:
        """Predicted rating for every movie from the user's top-k item neighbors."""
        observed = self.user_vector(ratings)
        gathered = observed[self.neighbor_index]
        weights = self.neighbor_similarity * (gathered > 0)
        denominator = np.abs(weights).sum(axis=1)
        numerator = (weights * gathered).sum(axis=1)
        scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        return scores

    def svd_scores(self, ratings: list[tuple[int, float]]) -> np.ndarray:
        """Fold the user into the fitted SVD and reconstruct a rating for every movie."""
        observed = self.user_vector(ratings)
        if not np.any(observed):
            return np.zeros(len(self.movie_ids), dtype=np.float64)
        row = sparse.csr_matrix(observed.reshape(1, -1))
        factors = self.svd.transform(row)
        predicted = np.asarray(factors @ self.item_factors.T).ravel()
        return np.clip(predicted, 0.5, 5.0)

    def popularity_scores(self) -> np.ndarray:
        """Ranking score: log of the training rating count."""
        return np.log1p(self.popularity)


def _index_maps(ids: np.ndarray) -> dict[int, int]:
    return {int(value): index for index, value in enumerate(ids)}


def _top_neighbors(similarity: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    np.fill_diagonal(similarity, 0.0)
    similarity[similarity < 0] = 0.0
    width = similarity.shape[0]
    k = min(neighbors, max(width - 1, 1))
    chosen = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(width)[:, None]
    values = similarity[rows, chosen]
    order = np.argsort(-values, axis=1)
    return chosen[rows, order], values[rows, order]


def fit_collaborative(movies: pd.DataFrame, ratings: pd.DataFrame) -> CollaborativeModels:
    """Fit item-kNN and SVD on the training ratings only."""
    movie_ids = movies["movieId"].astype(int).to_numpy()
    id_to_idx = _index_maps(movie_ids)
    known = ratings.loc[ratings["movieId"].astype(int).isin(id_to_idx)].copy()
    user_ids = np.array(sorted(known["userId"].astype(int).unique()))
    user_to_idx = _index_maps(user_ids)

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for user_id, movie_id, rating in known[["userId", "movieId", "rating"]].itertuples(index=False):
        rows.append(user_to_idx[int(user_id)])
        cols.append(id_to_idx[int(movie_id)])
        values.append(float(rating))
    matrix = sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
        dtype=np.float64,
    )

    logger.info("Fitting item-kNN on %s users and %s items", matrix.shape[0], matrix.shape[1])
    similarity = cosine_similarity(matrix.T, dense_output=True)
    neighbor_index, neighbor_similarity = _top_neighbors(similarity, KNN_NEIGHBORS)

    limit = min(matrix.shape) - 1
    components = max(1, min(SVD_COMPONENTS, limit))
    logger.info("Fitting TruncatedSVD with %s components", components)
    svd = TruncatedSVD(n_components=components, random_state=SEED)
    svd.fit(matrix)

    counts = np.asarray(matrix.getnnz(axis=0)).ravel().astype(np.float64)
    sums = np.asarray(matrix.sum(axis=0)).ravel()
    item_mean = np.divide(sums, counts, out=np.full(len(movie_ids), 3.0), where=counts > 0)
    return CollaborativeModels(
        movie_ids=movie_ids,
        id_to_idx=id_to_idx,
        user_ids=user_ids,
        user_to_idx=user_to_idx,
        neighbor_index=neighbor_index,
        neighbor_similarity=neighbor_similarity,
        svd=svd,
        item_factors=svd.components_.T.copy(),
        train_matrix=matrix,
        item_mean=item_mean,
        popularity=counts,
    )
