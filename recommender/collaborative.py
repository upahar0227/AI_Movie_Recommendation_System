from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

from .config import N_LATENT, N_NEIGHBORS
from .data import MovieLensData, user_item_matrix


class ItemKNNRecommender:
    """Item-item kNN collaborative filter (Sarwar et al., 2001 style)."""

    name = "item_knn"

    def __init__(self, n_neighbors: int = N_NEIGHBORS):
        self.n_neighbors = n_neighbors
        self.data: MovieLensData | None = None
        self.matrix = None
        self.item_sim = None
        self.train_ratings = None

    def fit(self, data: MovieLensData, ratings=None):
        self.data = data
        self.train_ratings = data.ratings if ratings is None else ratings
        self.matrix = user_item_matrix(data, self.train_ratings)
        item_vectors = csr_matrix(self.matrix.T)
        dense_sim = cosine_similarity(item_vectors)
        np.fill_diagonal(dense_sim, 0.0)
        k = min(self.n_neighbors, dense_sim.shape[0] - 1)
        rows, cols, vals = [], [], []
        for i in range(dense_sim.shape[0]):
            neighbors = np.argpartition(-dense_sim[i], k)[:k]
            rows.extend([i] * k)
            cols.extend(neighbors.tolist())
            vals.extend(dense_sim[i, neighbors].tolist())
        self.item_sim = csr_matrix(
            (vals, (rows, cols)), shape=dense_sim.shape, dtype=np.float32
        )
        return self

    def score_user(self, user_id: int) -> np.ndarray:
        user_ratings = self.matrix[int(user_id)]
        scores = self.item_sim @ user_ratings
        scores[user_ratings > 0] = 0.0
        return np.asarray(scores, dtype=np.float32)


class SVDRecommender:
    """Latent-factor collaborative filter via truncated SVD (Koren et al., 2009 family)."""

    name = "svd"

    def __init__(self, n_latent: int = N_LATENT):
        self.n_latent = n_latent
        self.data: MovieLensData | None = None
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0.0
        self.matrix = None
        self.train_ratings = None

    def fit(self, data: MovieLensData, ratings=None):
        self.data = data
        self.train_ratings = data.ratings if ratings is None else ratings
        self.matrix = user_item_matrix(data, self.train_ratings)
        mask = self.matrix > 0
        self.global_mean = float(self.matrix[mask].mean()) if mask.any() else 0.0
        centered = np.where(mask, self.matrix - self.global_mean, 0.0)
        k = min(self.n_latent, min(centered.shape) - 1)
        u, s, vt = svds(csr_matrix(centered), k=k)
        order = np.argsort(s)[::-1]
        self.user_factors = u[:, order] * s[order]
        self.item_factors = vt[order].T
        return self

    def score_user(self, user_id: int) -> np.ndarray:
        uid = int(user_id)
        if uid >= self.user_factors.shape[0]:
            return np.zeros(self.item_factors.shape[0], dtype=np.float32)
        scores = self.user_factors[uid] @ self.item_factors.T + self.global_mean
        rated = self.matrix[uid] > 0
        scores[rated] = 0.0
        return scores.astype(np.float32)

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        idx = self.data.movie_id_to_idx.get(int(movie_id))
        if idx is None or int(user_id) >= self.user_factors.shape[0]:
            return self.global_mean
        return float(self.user_factors[int(user_id)] @ self.item_factors[idx] + self.global_mean)
