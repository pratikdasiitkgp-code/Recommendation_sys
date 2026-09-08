"""
Streamlit UI - Recommendation Demo
=====================================================
A simple, demo-able frontend for the two-stage recommendation pipeline.
Reads directly from the parquet files produced by the pipeline (no need
to run the FastAPI service alongside this) - good for a live portfolio
demo or deploying to Streamlit Community Cloud.
 
Run:
    streamlit run streamlit_app.py
"""
 
import os
 
import pandas as pd
import streamlit as st
 
DATA_DIR = os.environ.get("DATA_DIR", "./data")
 
st.set_page_config(page_title="Recommendation System Demo", page_icon="🛍️", layout="wide")
 
 
@st.cache_data
def load_data():
    ranked = pd.read_parquet(os.path.join(DATA_DIR, "ranked_candidates.parquet"))
    meta = pd.read_parquet(os.path.join(DATA_DIR, "item_metadata.parquet"))
    eval_path = os.path.join(DATA_DIR, "eval_results.json")
    eval_results = None
    if os.path.exists(eval_path):
        import json
        with open(eval_path) as f:
            eval_results = json.load(f)
    return ranked, meta, eval_results
 
 
ranked_df, meta_df, eval_results = load_data()
 
st.title("🛍️ Two-Stage Product Recommendation System")
st.caption("ALS candidate retrieval + LightGBM ranking, trained on Amazon Reviews 2023")
 
# --- Sidebar: model performance ---
with st.sidebar:
    st.header("Model performance")
    if eval_results:
        retrieval = eval_results.get("retrieval_only (ALS)", {})
        full = eval_results.get("retrieval_plus_ranking (ALS + LightGBM)", {})
        st.metric(
            "NDCG@10 (retrieval only)",
            f"{retrieval.get('ndcg@10', 0):.4f}",
        )
        st.metric(
            "NDCG@10 (retrieval + ranking)",
            f"{full.get('ndcg@10', 0):.4f}",
            delta=f"{full.get('ndcg@10', 0) - retrieval.get('ndcg@10', 0):+.4f}",
        )
        st.metric("Precision@10 (full pipeline)", f"{full.get('precision@10', 0):.4f}")
    else:
        st.info("Run 04_evaluation.py to see metrics here.")
 
    st.divider()
    st.header("Architecture")
    st.markdown(
        """
        1. **Retrieval** — implicit ALS generates ~100 candidates/user
        2. **Ranking** — LightGBM re-ranks with recency, popularity,
           price, and ALS score as features
        """
    )
 
# --- Main: pick a user, show recommendations ---
user_ids = sorted(ranked_df["user_id"].unique().tolist())
selected_user = st.selectbox(
    f"Choose a user ({len(user_ids)} available)", user_ids
)
 
top_k = st.slider("Number of recommendations", min_value=3, max_value=20, value=10)
 
user_recs = (
    ranked_df[ranked_df["user_id"] == selected_user]
    .sort_values("rank_score", ascending=False)
    .head(top_k)
    .merge(meta_df[["item_id", "title"]], on="item_id", how="left")
)
# price and average_rating already exist in ranked_df (used as ranking
# features in 03_ranking.py) - only pulling `title` from meta_df here
# to avoid duplicate/suffixed columns on re-merge.
 
st.subheader(f"Top {top_k} recommendations")
 
cols = st.columns(3)
for i, row in enumerate(user_recs.itertuples()):
    with cols[i % 3]:
        with st.container(border=True):
            title = getattr(row, "title", None) or row.item_id
            st.markdown(f"**{title[:80]}**" if isinstance(title, str) else f"**{row.item_id}**")
            price = getattr(row, "price", None)
            avg_rating = getattr(row, "average_rating", None)
            if price and str(price) not in ("None", "nan"):
                st.write(f"💲 {price}")
            if avg_rating:
                st.write(f"⭐ {avg_rating}")
            st.caption(f"rank score: {row.rank_score:.4f}")
 
if user_recs.empty:
    st.warning("No recommendations found for this user.")
 
st.divider()
with st.expander("Raw recommendation table"):
    st.dataframe(
        user_recs[["item_id", "title", "price", "average_rating", "rank_score"]],
        use_container_width=True,
    )