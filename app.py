"""
API - Serve recommendations from the trained pipeline
=======================================================
Wraps the offline pipeline output (ranked_candidates.parquet) in a
FastAPI service so the project can be deployed and demoed live -
this is what turns "I trained a model" into "here's a working
endpoint", which is a much stronger thing to show in an interview.

Local run:
    uvicorn app:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API docs.
"""

import os
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DATA_DIR = os.environ.get("DATA_DIR", "./data")

app = FastAPI(
    title="Sequential Recommendation API",
    description="Two-stage (ALS retrieval + LightGBM ranking) product recommendation service",
    version="1.0.0",
)


class RecommendationItem(BaseModel):
    item_id: str
    rank_score: float


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendationItem]


@lru_cache(maxsize=1)
def load_ranked_candidates() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "ranked_candidates.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run the full pipeline (01-04) first to generate it."
        )
    return pd.read_parquet(path)


@app.get("/health")
def health():
    """Basic health check for deployment platforms (Render/HF Spaces ping this)."""
    return {"status": "ok"}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: str, k: int = 10):
    """Return the top-k ranked recommendations for a given user_id."""
    try:
        df = load_ranked_candidates()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    user_rows = df[df["user_id"] == user_id]
    if user_rows.empty:
        raise HTTPException(status_code=404, detail=f"No recommendations found for user_id={user_id}")

    top_k = (
        user_rows.sort_values("rank_score", ascending=False)
        .head(k)[["item_id", "rank_score"]]
        .to_dict(orient="records")
    )

    return RecommendationResponse(user_id=user_id, recommendations=top_k)


@app.get("/sample_users")
def sample_users(n: int = 10):
    """Convenience endpoint: returns a handful of valid user_ids to try in /recommend."""
    try:
        df = load_ranked_candidates()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"user_ids": df["user_id"].drop_duplicates().head(n).tolist()}
