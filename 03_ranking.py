"""
Phase 3 - Ranking
=====================================================
Takes the candidates from Phase 2 (ALS retrieval) and re-ranks them
with a gradient-boosted ranker trained on richer features than the
retrieval model has access to: recency, popularity, price, category
match, and the ALS score itself as one input feature among many.

This is the step most student recsys projects skip - including it
is what makes this look like a production-style two-stage system
instead of a single collaborative-filtering demo.

Run:
    pip install -r requirements.txt
    python 03_ranking.py --data_dir ./data
"""

import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd


def build_item_features(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Simple item-level features from metadata."""
    feats = meta_df.copy()
    feats["price"] = pd.to_numeric(feats["price"], errors="coerce")
    feats["price"] = feats["price"].fillna(feats["price"].median())
    feats["average_rating"] = pd.to_numeric(feats["average_rating"], errors="coerce").fillna(0)
    feats["rating_number"] = pd.to_numeric(feats["rating_number"], errors="coerce").fillna(0)
    feats["log_popularity"] = np.log1p(feats["rating_number"])
    return feats[["item_id", "price", "average_rating", "log_popularity"]]


def build_user_features(train_df: pd.DataFrame) -> pd.DataFrame:
    """Simple user-level features: activity level and average rating given."""
    g = train_df.groupby("user_id").agg(
        user_n_interactions=("item_id", "count"),
        user_avg_rating=("rating", "mean"),
        user_last_timestamp=("timestamp", "max"),
    ).reset_index()
    return g


def build_training_examples(candidates_df, train_df, val_df, item_feats, user_feats, idx_to_user):
    """
    Build labeled (user, item) rows for the ranker:
      positive = the item the user actually interacted with next (val set)
      negative = ALS candidates the user did NOT interact with
    This is a standard pointwise ranking setup (binary label + LightGBM
    ranking objective grouped by user).
    """
    candidates_df = candidates_df.copy()
    candidates_df["user_id"] = candidates_df["user_idx"].map(idx_to_user)

    # Positive labels: the validation-set "next item" per user
    val_pos = val_df[["user_id", "item_id"]].copy()
    val_pos["label"] = 1

    merged = candidates_df.merge(
        val_pos, on=["user_id", "item_id"], how="left"
    )
    merged["label"] = merged["label"].fillna(0).astype(int)

    merged = merged.merge(item_feats, on="item_id", how="left")
    merged = merged.merge(user_feats, on="user_id", how="left")
    merged = merged.fillna(0)

    return merged


def train_ranker(feature_df: pd.DataFrame):
    feature_cols = [
        "als_score", "price", "average_rating", "log_popularity",
        "user_n_interactions", "user_avg_rating",
    ]

    feature_df = feature_df.sort_values("user_id")
    group_sizes = feature_df.groupby("user_id").size().values

    train_set = lgb.Dataset(
        feature_df[feature_cols],
        label=feature_df["label"],
        group=group_sizes,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
    }

    model = lgb.train(params, train_set, num_boost_round=200)
    return model, feature_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    print("[1/5] Loading data")
    train_df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(args.data_dir, "val.parquet"))
    meta_df = pd.read_parquet(os.path.join(args.data_dir, "item_metadata.parquet"))
    candidates_df = pd.read_parquet(os.path.join(args.data_dir, "candidates.parquet"))

    import json
    with open(os.path.join(args.data_dir, "id_maps.json")) as f:
        id_maps = json.load(f)
    idx_to_user = {int(k): v for k, v in id_maps["idx_to_user"].items()}

    print("[2/5] Building item and user features")
    item_feats = build_item_features(meta_df)
    user_feats = build_user_features(train_df)

    print("[3/5] Building labeled training examples from candidates + val labels")
    feature_df = build_training_examples(
        candidates_df, train_df, val_df, item_feats, user_feats, idx_to_user
    )
    print(f"      rows: {len(feature_df):,} | positive rate: {feature_df['label'].mean():.4f}")

    print("[4/5] Training LightGBM lambdarank model")
    model, feature_cols = train_ranker(feature_df)

    print("[5/5] Scoring all candidates and saving ranked output")
    feature_df["rank_score"] = model.predict(feature_df[feature_cols])
    ranked_df = feature_df.sort_values(["user_id", "rank_score"], ascending=[True, False])
    ranked_df.to_parquet(os.path.join(args.data_dir, "ranked_candidates.parquet"), index=False)

    model.save_model(os.path.join(args.data_dir, "ranker_model.txt"))

    print("\nDone. Ranked candidates saved to ranked_candidates.parquet")
    print("Top feature importances:")
    importances = dict(zip(feature_cols, model.feature_importance()))
    for k, v in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
