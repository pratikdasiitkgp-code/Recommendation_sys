"""
Phase 4 - Evaluation
=====================================================
Computes ranking metrics (Precision@K, Recall@K, NDCG@K) - NOT RMSE,
which is the wrong metric for a ranking/retrieval task. Compares:
  (a) ALS candidates only (retrieval-stage ordering)
  (b) ALS + LightGBM ranking (full two-stage pipeline)
so you can show, with numbers, that the ranking stage adds value.
This comparison is exactly the kind of thing to put in your write-up.

Run:
    python 04_evaluation.py --data_dir ./data --k 10
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def precision_at_k(ranked_items, true_item, k):
    return 1.0 if true_item in ranked_items[:k] else 0.0


def recall_at_k(ranked_items, true_item, k):
    # single true item per user here -> recall@k == precision@k,
    # kept separate for clarity / to generalize to multi-positive later
    return 1.0 if true_item in ranked_items[:k] else 0.0


def ndcg_at_k(ranked_items, true_item, k):
    if true_item not in ranked_items[:k]:
        return 0.0
    rank = ranked_items[:k].index(true_item)
    return 1.0 / np.log2(rank + 2)  # +2 because rank is 0-indexed


def evaluate(ranked_df: pd.DataFrame, test_df: pd.DataFrame, score_col: str, k: int):
    test_lookup = test_df.set_index("user_id")["item_id"].to_dict()

    precisions, recalls, ndcgs = [], [], []

    for user_id, group in ranked_df.groupby("user_id"):
        if user_id not in test_lookup:
            continue
        true_item = test_lookup[user_id]
        ranked_items = (
            group.sort_values(score_col, ascending=False)["item_id"].tolist()
        )

        precisions.append(precision_at_k(ranked_items, true_item, k))
        recalls.append(recall_at_k(ranked_items, true_item, k))
        ndcgs.append(ndcg_at_k(ranked_items, true_item, k))

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_users_evaluated": len(precisions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    print("[1/3] Loading test split and ranked candidates")
    test_df = pd.read_parquet(os.path.join(args.data_dir, "test.parquet"))
    ranked_df = pd.read_parquet(os.path.join(args.data_dir, "ranked_candidates.parquet"))

    print(f"[2/3] Evaluating retrieval-only ordering (ALS score) at K={args.k}")
    retrieval_metrics = evaluate(ranked_df, test_df, score_col="als_score", k=args.k)

    print(f"[3/3] Evaluating full pipeline ordering (ranker score) at K={args.k}")
    ranked_metrics = evaluate(ranked_df, test_df, score_col="rank_score", k=args.k)

    results = {
        "retrieval_only (ALS)": retrieval_metrics,
        "retrieval_plus_ranking (ALS + LightGBM)": ranked_metrics,
    }

    print("\n=== Results ===")
    print(json.dumps(results, indent=2))

    with open(os.path.join(args.data_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    lift = (
        ranked_metrics[f"ndcg@{args.k}"] - retrieval_metrics[f"ndcg@{args.k}"]
    )
    print(f"\nNDCG@{args.k} lift from ranking stage: {lift:+.4f}")


if __name__ == "__main__":
    main()
