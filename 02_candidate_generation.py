"""
Phase 2 - Candidate Generation (ALS collaborative filtering)
===============================================================
Takes the train split from Phase 1 and trains an implicit-feedback
ALS model to generate top-K candidate items per user. This is the
"retrieval" stage of a two-stage recommender: fast, approximate,
casts a wide net. Precision comes later from the ranking stage.

Run:
    pip install -r requirements.txt
    python 02_candidate_generation.py --data_dir ./data --factors 64 --top_k 100
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_id_maps(train_df: pd.DataFrame):
    """Map raw user_id / item_id strings to contiguous integer indices."""
    user_ids = train_df["user_id"].unique()
    item_ids = train_df["item_id"].unique()

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {it: i for i, it in enumerate(item_ids)}
    idx_to_item = {i: it for it, i in item_to_idx.items()}

    return user_to_idx, item_to_idx, idx_to_item


def build_interaction_matrix(train_df: pd.DataFrame, user_to_idx, item_to_idx):
    """
    Build a sparse user-item matrix of implicit "confidence" values.
    We use a simple confidence = 1 + alpha * rating scheme, standard
    for implicit-feedback ALS (Hu, Koren, Volinsky 2008).
    """
    alpha = 2.0
    rows = train_df["user_id"].map(user_to_idx).values
    cols = train_df["item_id"].map(item_to_idx).values
    conf = 1 + alpha * train_df["rating"].clip(lower=1).values

    mat = csr_matrix(
        (conf, (rows, cols)),
        shape=(len(user_to_idx), len(item_to_idx)),
    )
    return mat


def train_als(interaction_matrix: csr_matrix, factors: int, iterations: int, reg: float):
    from implicit.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=reg,
        iterations=iterations,
        random_state=42,
    )
    # implicit expects (user, item) matrix for .fit
    model.fit(interaction_matrix)
    return model


def generate_candidates(model, interaction_matrix, idx_to_item, top_k: int):
    """
    For every user, retrieve top_k candidate items via the ALS model's
    approximate nearest-neighbor recommend() call (already excludes
    items the user has already interacted with).
    """
    n_users = interaction_matrix.shape[0]
    results = []

    ids, scores = model.recommend(
        userid=np.arange(n_users),
        user_items=interaction_matrix,
        N=top_k,
        filter_already_liked_items=True,
    )

    for u in range(n_users):
        for item_idx, score in zip(ids[u], scores[u]):
            if item_idx == -1:
                continue
            results.append((u, idx_to_item[item_idx], float(score)))

    return pd.DataFrame(results, columns=["user_idx", "item_id", "als_score"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument("--top_k", type=int, default=100,
                         help="Number of candidates to generate per user (this is the retrieval width)")
    args = parser.parse_args()

    print("[1/5] Loading train split")
    train_df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))

    print("[2/5] Building ID maps and sparse interaction matrix")
    user_to_idx, item_to_idx, idx_to_item = build_id_maps(train_df)
    interaction_matrix = build_interaction_matrix(train_df, user_to_idx, item_to_idx)
    print(f"      matrix shape: {interaction_matrix.shape}, nnz: {interaction_matrix.nnz:,}")

    print(f"[3/5] Training ALS (factors={args.factors}, iterations={args.iterations})")
    model = train_als(interaction_matrix, args.factors, args.iterations, args.reg)

    print(f"[4/5] Generating top-{args.top_k} candidates per user")
    candidates_df = generate_candidates(model, interaction_matrix, idx_to_item, args.top_k)

    print("[5/5] Saving candidates and ID maps")
    candidates_df.to_parquet(os.path.join(args.data_dir, "candidates.parquet"), index=False)

    idx_to_user = {i: u for u, i in user_to_idx.items()}
    with open(os.path.join(args.data_dir, "id_maps.json"), "w") as f:
        json.dump(
            {"idx_to_user": idx_to_user, "idx_to_item": idx_to_item},
            f,
        )

    print(f"\nDone. {len(candidates_df):,} (user, candidate) pairs saved.")
    print(f"Avg candidates per user: {len(candidates_df) / len(user_to_idx):.1f}")


if __name__ == "__main__":
    main()
