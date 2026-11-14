from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.config import K, PROCESSED_DIR  # noqa: E402
from src.app.settings import APP_NAME  # noqa: E402
from recommender.service import build_suite  # noqa: E402

METHODS = {
    "hybrid_sentiment": "Hybrid + review sentiment (proposed)",
    "hybrid": "Hybrid (content + SVD)",
    "content": "Content-based cosine / TF-IDF",
    "item_knn": "Collaborative item-kNN",
    "svd": "Collaborative SVD",
}


@st.cache_resource(show_spinner="Fitting content, collaborative, hybrid and sentiment models…")
def suite():
    return build_suite()


def main():
    st.set_page_config(page_title=APP_NAME, layout="wide")
    st.markdown(
        """
        <style>
          .stApp { background: #0f1117; color: #eeeae3; }
          h1, h2, h3 { font-family: Georgia, serif; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("AI-Based Movie Recommendation System")
    st.caption(
        "Hybrid, sentiment-aware recommender benchmarked against content-based and "
        "collaborative baselines — MovieLens metadata + NLTK polar reviews."
    )

    recs = suite()
    users = sorted(recs.data.ratings["userId"].unique().tolist())

    tab_rec, tab_cold, tab_eval = st.tabs(
        ["Recommend for a viewer", "Cold-start (favorites)", "Offline evaluation"]
    )

    with tab_rec:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            method = st.selectbox(
                "Model",
                list(METHODS.keys()),
                format_func=lambda key: METHODS[key],
            )
        with col2:
            user_id = st.selectbox("MovieLens user", users, index=0)
        with col3:
            k = st.slider("k", 5, 20, K)

        history = recs.data.ratings[recs.data.ratings["userId"] == user_id].merge(
            recs.data.movies, on="movieId"
        ).sort_values("rating", ascending=False)
        st.subheader("Highly rated by this viewer")
        st.dataframe(
            history.head(8)[["title", "rating", "genres"]],
            hide_index=True,
            use_container_width=True,
        )
        results = recs.recommend(method, int(user_id), k=k)
        st.subheader("Recommendations")
        st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)

    with tab_cold:
        st.write(
            "RQ3-style cold start: no rating history. Describe films you like; "
            "the content + sentiment model ranks the catalog."
        )
        query = st.text_input(
            "Favorite movies (comma-separated titles)",
            "Toy Story, Heat, Pulp Fiction",
        )
        k_cold = st.slider("Results", 5, 20, 10, key="cold_k")
        if st.button("Recommend from favorites"):
            titles = [part.strip() for part in query.split(",")]
            cold = recs.recommend_from_titles(titles, k=k_cold)
            if not cold:
                st.warning("No titles matched the MovieLens catalog. Try a more complete name.")
            else:
                st.dataframe(pd.DataFrame(cold), hide_index=True, use_container_width=True)

    with tab_eval:
        path = PROCESSED_DIR / "benchmark.json"
        st.write(
            "Metrics from the proposal: precision@k, recall@k, NDCG, coverage, "
            "RMSE/MAE for SVD, and paired tests vs the content baseline."
        )
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            table = pd.DataFrame(payload["models"]).T
            st.dataframe(table, use_container_width=True)
            st.json(payload.get("significance_last_fold", {}))
            if "sentiment_classifiers" in payload:
                st.subheader("Sentiment classifiers (NB / logistic regression / random forest)")
                st.dataframe(pd.DataFrame(payload["sentiment_classifiers"]).T, use_container_width=True)
        else:
            st.info(
                "No cached benchmark yet. From the project folder run:\n\n"
                "`py -3 -m recommender.evaluate --folds 1 --skip-sentiment-lab`"
            )
            if st.button("Run a 1-fold benchmark now"):
                from recommender.evaluate import run_benchmark

                with st.spinner("Evaluating models on a per-user hold-out…"):
                    payload = run_benchmark(folds=1, include_sentiment_lab=False)
                st.dataframe(pd.DataFrame(payload["models"]).T, use_container_width=True)


if __name__ == "__main__":
    main()
