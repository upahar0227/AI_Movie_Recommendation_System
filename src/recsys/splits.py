"""Per-user folds. A movie a user rated is never in both train and test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import N_FOLDS, SEED


def per_user_folds(ratings: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Split each user's ratings into ``n_folds`` disjoint test folds.

    A movie a user rated lands in only one test fold. Other users' rows for that
    movie can still sit in train.

    Args:
        ratings: Rows with userId, movieId, and rating.
        n_folds: Folds per user. Each rating is assigned position modulo this count.
        seed: Shuffle seed, applied independently inside each user group.

    Returns:
        ``n_folds`` pairs of (train, test) frames. Test rows of one fold are dropped
        from that fold's train.
    """
    rng = np.random.default_rng(seed)
    assignments: list[pd.Series] = []
    for _user_id, group in ratings.groupby("userId"):
        indexes = group.index.to_numpy().copy()
        rng.shuffle(indexes)
        labels = np.empty(len(indexes), dtype=int)
        for position, index in enumerate(indexes):
            labels[position] = position % n_folds
        # ``labels`` follows the shuffled order, so map them back onto the index.
        fold = pd.Series(labels, index=indexes)
        assignments.append(fold)
    fold_of_row = pd.concat(assignments).sort_index()
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold_id in range(n_folds):
        test_index = fold_of_row.index[fold_of_row.to_numpy() == fold_id]
        test = ratings.loc[test_index]
        train = ratings.drop(index=test_index)
        folds.append((train, test))
    return folds
