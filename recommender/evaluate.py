from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

from .collaborative import ItemKNNRecommender, SVDRecommender
from .config import K, MIN_RATING_RELEVANT, PROCESSED_DIR, SEED, TEST_FRACTION
from .content_based import ContentBasedRecommender
from .data import load_movielens
from .hybrid import make_hybrid, make_hybrid_sentiment
from .metrics import coverage, f1_at_k, mae, ndcg_at_k, precision_at_k, recall_at_k, rmse
from .ranking import seen_movie_ids, top_k_from_scores
from .sentiment import evaluate_sentiment, fit_production_sentiment, movie_sentiment_scores


def per_user_holdout(ratings: pd.DataFrame, test_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for user_id, group in ratings.groupby("userId"):
        group = group.sort_values("timestamp")
        if len(group) < 5:
            train_parts.append(group)
            continue
        n_test = max(1, int(round(len(group) * test_frac)))
        # shuffle within user so the split is not only recency-based
        shuffled = group.iloc[rng.permutation(len(group))]
        test_parts.append(shuffled.iloc[:n_test])
        train_parts.append(shuffled.iloc[n_test:])
    return pd.concat(train_parts), pd.concat(test_parts)


def relevant_items(test_ratings: pd.DataFrame, user_id: int) -> set[int]:
    rows = test_ratings[test_ratings["userId"] == user_id]
    return set(rows.loc[rows["rating"] >= MIN_RATING_RELEVANT, "movieId"].astype(int))


def evaluate_fold(data, train, test, k: int, sentiment_index):
    content = ContentBasedRecommender()
    knn = ItemKNNRecommender()
    svd = SVDRecommender()
    content.fit(data, train)
    knn.fit(data, train)
    svd.fit(data, train)
    hybrid = make_hybrid(content, svd)
    hybrid.data = data
    hybrid_s = make_hybrid_sentiment(content, svd, sentiment_index)
    hybrid_s.data = data

    models = {
        "content": content,
        "item_knn": knn,
        "svd": svd,
        "hybrid": hybrid,
        "hybrid_sentiment": hybrid_s,
    }

    test_counts = train.groupby("movieId").size()
    cold_movies = set(test_counts[test_counts < 5].index.astype(int))
    train_counts_user = train.groupby("userId").size()
    cold_users = set(train_counts_user[train_counts_user <= 5].index.astype(int))

    per_user = {name: defaultdict(list) for name in models}
    recommended_pool = {name: [] for name in models}
    rating_true, rating_pred = [], []

    users = test["userId"].unique()
    for user_id in users:
        relevant = relevant_items(test, user_id)
        if not relevant:
            continue
        exclude = seen_movie_ids(train, user_id)
        is_cold_user = int(user_id) in cold_users
        for name, model in models.items():
            scores = model.score_user(int(user_id))
            ranked = top_k_from_scores(data, scores, k, exclude)
            rec_ids = [mid for mid, _ in ranked]
            recommended_pool[name].extend(rec_ids)
            metrics = {
                "precision": precision_at_k(rec_ids, relevant, k),
                "recall": recall_at_k(rec_ids, relevant, k),
                "f1": f1_at_k(rec_ids, relevant, k),
                "ndcg": ndcg_at_k(rec_ids, relevant, k),
            }
            if cold_movies:
                rec_cold = [mid for mid in rec_ids if mid in cold_movies]
                rel_cold = relevant & cold_movies
                metrics["cold_movie_hit"] = 1.0 if rec_cold and rel_cold else 0.0
            if is_cold_user:
                metrics["cold_user_ndcg"] = metrics["ndcg"]
            for key, value in metrics.items():
                per_user[name][key].append(value)

    test_pairs = test.sample(n=min(4000, len(test)), random_state=SEED)
    for user_id, movie_id, rating in test_pairs[["userId", "movieId", "rating"]].itertuples(
        index=False
    ):
        rating_true.append(float(rating))
        rating_pred.append(svd.predict_rating(int(user_id), int(movie_id)))

    summary = {}
    for name, buckets in per_user.items():
        summary[name] = {
            key: float(np.mean(vals)) if vals else 0.0 for key, vals in buckets.items()
        }
        summary[name]["coverage"] = coverage(recommended_pool[name], len(data.movie_ids))
        summary[name]["n_users"] = len(buckets.get("ndcg", []))
    summary["svd"]["rmse"] = rmse(np.array(rating_true), np.array(rating_pred))
    summary["svd"]["mae"] = mae(np.array(rating_true), np.array(rating_pred))

    ndcg_vectors = {name: np.array(per_user[name]["ndcg"]) for name in models if per_user[name]["ndcg"]}
    tests = {}
    baseline = ndcg_vectors.get("content")
    if baseline is not None:
        for name, vector in ndcg_vectors.items():
            if name == "content" or len(vector) != len(baseline):
                continue
            stat, pvalue = stats.ttest_rel(vector, baseline)
            tests[f"{name}_vs_content_ndcg"] = {
                "t": float(stat),
                "p": float(pvalue),
                "significant_0.05": bool(pvalue < 0.05),
            }
    return summary, tests, ndcg_vectors


def run_benchmark(folds: int = 1, k: int = K, include_sentiment_lab: bool = True) -> dict:
    data = load_movielens()
    sentiment_model = fit_production_sentiment()
    sentiment_index = movie_sentiment_scores(data, sentiment_model)

    fold_summaries = []
    fold_tests = []
    for fold in range(folds):
        train, test = per_user_holdout(data.ratings, TEST_FRACTION, SEED + fold)
        summary, tests, _ = evaluate_fold(data, train, test, k, sentiment_index)
        fold_summaries.append(summary)
        fold_tests.append(tests)

    model_names = ["content", "item_knn", "svd", "hybrid", "hybrid_sentiment"]
    averaged = {}
    for name in model_names:
        keys = fold_summaries[0][name].keys()
        averaged[name] = {
            key: float(np.mean([fold[name].get(key, 0.0) for fold in fold_summaries]))
            for key in keys
        }

    payload = {
        "k": k,
        "folds": folds,
        "models": averaged,
        "significance_last_fold": fold_tests[-1],
    }
    if include_sentiment_lab:
        lab = evaluate_sentiment()
        payload["sentiment_classifiers"] = lab["metrics"]
        payload["best_sentiment_model"] = lab["best"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "benchmark.json"
    serializable = json.loads(json.dumps(payload))
    out.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Offline recommender benchmark")
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--skip-sentiment-lab", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(
        folds=args.folds, k=args.k, include_sentiment_lab=not args.skip_sentiment_lab
    )
    print(json.dumps({k: v for k, v in result.items() if k != "sentiment_classifiers"}, indent=2))
    if "sentiment_classifiers" in result:
        print("\nSentiment classifiers")
        print(json.dumps(result["sentiment_classifiers"], indent=2))


if __name__ == "__main__":
    main()
