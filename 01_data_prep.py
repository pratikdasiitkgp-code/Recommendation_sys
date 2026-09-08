"""
Phase 1 - Data loading & time-based train/val/test split
==========================================================
Recommendation system project (session/sequential recsys, Amazon-style)

What this script does:
1. Loads a single category from the Amazon Reviews 2023 dataset
   (reviews = interactions, meta = item metadata)
2. Filters out users/items with too few interactions (standard k-core filtering)
3. Builds a per-user interaction sequence sorted by timestamp
4. Splits each user's sequence using "leave-last-out":
     - last item      -> test
     - second-last     -> validation
     - everything else -> train
   This is the standard split for sequential/session recommendation -
   a RANDOM split would leak future interactions into training, which is
   a common mistake interviewers will ask you about.

Run this on your own machine (needs internet access to Hugging Face).
    pip install -r requirements.txt
    python 01_data_prep.py --category All_Beauty --min_interactions 5
"""

import argparse
import json
import os

import pandas as pd


def _download_jsonl(repo_path: str, cache_dir: str = "./hf_cache") -> str:
    """
    Download a single raw .jsonl file from the Amazon-Reviews-2023 dataset
    repo directly (bypassing `datasets.load_dataset`, since the dataset
    still relies on a loading script that `datasets>=4.0` no longer
    supports - see https://github.com/hyp1231/AmazonReviews2023/issues/41).
    """
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        repo_type="dataset",
        filename=repo_path,
        cache_dir=cache_dir,
    )
    return local_path


def load_reviews(category: str) -> pd.DataFrame:
    """Load the raw review (interaction) data for one category."""
    local_path = _download_jsonl(f"raw/review_categories/{category}.jsonl")
    df = pd.read_json(local_path, lines=True)
    df = df[["user_id", "parent_asin", "rating", "timestamp"]].rename(
        columns={"parent_asin": "item_id"}
    )
    df = df.dropna(subset=["user_id", "item_id", "timestamp"])
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    return df


def load_item_metadata(category: str) -> pd.DataFrame:
    """Load item metadata (title, price, category) for feature building later."""
    local_path = _download_jsonl(f"raw/meta_categories/meta_{category}.jsonl")
    meta = pd.read_json(local_path, lines=True)
    meta = meta.rename(columns={"parent_asin": "item_id"})
    for col in ["average_rating", "rating_number", "price"]:
        if col not in meta.columns:
            meta[col] = None
    return meta[["item_id", "title", "average_rating", "rating_number", "price"]]


def k_core_filter(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    """
    Iteratively drop users/items with fewer than `min_interactions`
    interactions until the dataset stabilizes. Standard preprocessing
    step for recsys - keeps the graph dense enough to learn from.
    """
    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["item_id"].value_counts()

        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index

        before = len(df)
        df = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
        after = len(df)

        if after == before:
            break

    return df.reset_index(drop=True)


def build_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Sort each user's interactions by time and assign a position index."""
    df = df.sort_values(["user_id", "timestamp"])
    df["position"] = df.groupby("user_id").cumcount()
    df["seq_len"] = df.groupby("user_id")["item_id"].transform("count")
    return df


def leave_last_out_split(df: pd.DataFrame):
    """
    Time-based split per user:
      last interaction        -> test
      second-to-last          -> validation
      everything before that  -> train
    """
    df = df.copy()
    df["rank_from_end"] = df.groupby("user_id").cumcount(ascending=False)

    test_mask = df["rank_from_end"] == 0
    val_mask = df["rank_from_end"] == 1
    train_mask = df["rank_from_end"] >= 2

    train_df = df[train_mask].drop(columns=["rank_from_end"])
    val_df = df[val_mask].drop(columns=["rank_from_end"])
    test_df = df[test_mask].drop(columns=["rank_from_end"])

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="All_Beauty")
    parser.add_argument("--min_interactions", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="./data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/5] Loading reviews for category: {args.category}")
    reviews = load_reviews(args.category)
    print(f"      raw interactions: {len(reviews):,}")

    print(f"[2/5] Loading item metadata for category: {args.category}")
    meta = load_item_metadata(args.category)
    print(f"      items with metadata: {len(meta):,}")

    print(f"[3/5] Applying k-core filter (min_interactions={args.min_interactions})")
    filtered = k_core_filter(reviews, args.min_interactions)
    print(f"      interactions after filtering: {len(filtered):,}")
    print(f"      users: {filtered['user_id'].nunique():,} | items: {filtered['item_id'].nunique():,}")

    print("[4/5] Building per-user sequences")
    seq_df = build_sequences(filtered)

    print("[5/5] Creating time-based leave-last-out train/val/test split")
    train_df, val_df, test_df = leave_last_out_split(seq_df)
    print(f"      train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}")

    train_df.to_parquet(os.path.join(args.out_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(args.out_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.out_dir, "test.parquet"), index=False)
    meta.to_parquet(os.path.join(args.out_dir, "item_metadata.parquet"), index=False)

    stats = {
        "category": args.category,
        "min_interactions": args.min_interactions,
        "n_users": int(filtered["user_id"].nunique()),
        "n_items": int(filtered["item_id"].nunique()),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\nDone. Splits saved to:", args.out_dir)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
