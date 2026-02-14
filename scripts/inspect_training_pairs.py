#!/usr/bin/env python3
"""Inspect training pairs generated for diner embedding model.

This script randomly samples positive pairs and displays them with
diner names and categories to help understand the data quality.

Usage:
    poetry run python scripts/inspect_training_pairs.py [--num_samples 10]
    poetry run python scripts/inspect_training_pairs.py --show_negatives --num_samples 5
"""

import argparse
import random

import pandas as pd
import torch


def load_data(
    pairs_path: str = "data/processed/training_pairs.parquet",
    diner_csv_path: str = "data/diner.csv",
    category_csv_path: str = "data/diner_category.csv",
):
    """Load pairs and diner metadata."""
    pairs_df = pd.read_parquet(pairs_path)
    diner_df = pd.read_csv(diner_csv_path, low_memory=False)
    category_df = pd.read_csv(category_csv_path)

    # Merge category into diner
    diner_df = diner_df.merge(
        category_df[
            [
                "diner_idx",
                "diner_category_large",
                "diner_category_middle",
                "diner_category_small",
            ]
        ],
        on="diner_idx",
        how="left",
    )

    return pairs_df, diner_df


def get_diner_info(diner_idx: int, diner_df: pd.DataFrame) -> dict:
    """Get diner name and category info."""
    row = diner_df[diner_df["diner_idx"] == diner_idx]
    if len(row) == 0:
        return {"name": "Unknown", "category": "Unknown"}

    row = row.iloc[0]
    return {
        "name": row.get("diner_name", "Unknown"),
        "large": row.get("diner_category_large", "Unknown"),
        "middle": row.get("diner_category_middle", "Unknown"),
        "small": row.get("diner_category_small", "Unknown"),
    }


def print_pair(idx: int, row: pd.Series, diner_df: pd.DataFrame):
    """Print a single pair with details."""
    anchor_info = get_diner_info(row["anchor_idx"], diner_df)
    positive_info = get_diner_info(row["positive_idx"], diner_df)
    pair_type = row.get("pair_type", "unknown")

    # Check category match
    same_large = anchor_info["large"] == positive_info["large"]
    same_middle = anchor_info["middle"] == positive_info["middle"]

    print(f"\n{'=' * 70}")
    print(f"Pair #{idx + 1} | Type: {pair_type}")
    print(f"{'=' * 70}")
    print(f"ANCHOR:   {anchor_info['name']}")
    print(
        f"          {anchor_info['large']} > {anchor_info['middle']} > {anchor_info['small']}"
    )
    print("")
    print(f"POSITIVE: {positive_info['name']}")
    print(
        f"          {positive_info['large']} > {positive_info['middle']} > {positive_info['small']}"
    )
    print("")
    print(
        f"Category Match: Large={'Yes' if same_large else 'No'}, Middle={'Yes' if same_middle else 'No'}"
    )


def print_triplet(
    idx: int, sample: dict, diner_df: pd.DataFrame, position_to_diner_idx: dict
):
    """Print a triplet with anchor, positive, and negatives."""
    anchor_pos = sample["anchor_idx"].item()
    positive_pos = sample["positive_idx"].item()
    negative_positions = sample["negative_indices"].tolist()

    anchor_idx = position_to_diner_idx.get(anchor_pos, anchor_pos)
    positive_idx = position_to_diner_idx.get(positive_pos, positive_pos)
    negative_indices = [position_to_diner_idx.get(p, p) for p in negative_positions]

    anchor_info = get_diner_info(anchor_idx, diner_df)
    positive_info = get_diner_info(positive_idx, diner_df)

    print(f"\n{'=' * 70}")
    print(f"Triplet #{idx + 1}")
    print(f"{'=' * 70}")
    print(f"ANCHOR:   {anchor_info['name']}")
    print(
        f"          {anchor_info['large']} > {anchor_info['middle']} > {anchor_info['small']}"
    )
    print()
    print(f"POSITIVE: {positive_info['name']}")
    print(
        f"          {positive_info['large']} > {positive_info['middle']} > {positive_info['small']}"
    )
    print()
    print(f"NEGATIVES ({len(negative_indices)}):")
    for i, neg_idx in enumerate(negative_indices):
        neg_info = get_diner_info(neg_idx, diner_df)
        same_large = anchor_info["large"] == neg_info["large"]
        same_middle = anchor_info["middle"] == neg_info["middle"]
        hardness = "HARD" if same_middle else ("SEMI" if same_large else "EASY")
        print(f"  [{hardness:4}] {neg_info['name']}")
        print(
            f"          {neg_info['large']} > {neg_info['middle']} > {neg_info['small']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Inspect training pairs")
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of pairs to sample"
    )
    parser.add_argument(
        "--pairs_path", type=str, default="data/processed/training_pairs.parquet"
    )
    parser.add_argument(
        "--pair_type",
        type=str,
        default=None,
        help="Filter by pair type: co_review or category",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--show_negatives",
        action="store_true",
        help="Show negative samples (requires loading dataset)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading data...")
    pairs_df, diner_df = load_data(pairs_path=args.pairs_path)

    print(f"\nTotal pairs: {len(pairs_df)}")
    if "pair_type" in pairs_df.columns:
        print("Pair type distribution:")
        print(pairs_df["pair_type"].value_counts().to_string())

    # Filter by pair type if specified
    if args.pair_type:
        pairs_df = pairs_df[pairs_df["pair_type"] == args.pair_type]
        print(f"\nFiltered to {args.pair_type}: {len(pairs_df)} pairs")

    if args.show_negatives:
        # Load the full dataset to get negative samples
        from yamyam_lab.data.diner_embedding import DinerEmbeddingDataset

        print("\nLoading dataset with negative mining...")
        dataset = DinerEmbeddingDataset(
            pairs_path=args.pairs_path,
            features_path="data/processed/diner_features.parquet",
            category_mapping_path="data/processed/category_mapping.parquet",
        )

        # Build reverse mapping: position -> diner_idx
        position_to_diner_idx = {v: k for k, v in dataset.diner_idx_to_position.items()}

        # Sample random indices
        n_samples = min(args.num_samples, len(dataset))
        sample_indices = random.sample(range(len(dataset)), n_samples)

        print(f"\n{'#' * 70}")
        print(f"# Randomly Sampled {n_samples} Triplets (with negatives)")
        print(f"{'#' * 70}")

        for i, idx in enumerate(sample_indices):
            sample = dataset[idx]
            print_triplet(i, sample, diner_df, position_to_diner_idx)
    else:
        # Sample pairs
        n_samples = min(args.num_samples, len(pairs_df))
        sampled = pairs_df.sample(n=n_samples, random_state=args.seed)

        print(f"\n{'#' * 70}")
        print(f"# Randomly Sampled {n_samples} Pairs")
        print(f"{'#' * 70}")

        for idx, (_, row) in enumerate(sampled.iterrows()):
            print_pair(idx, row, diner_df)

    print(f"\n{'=' * 70}")
    print("Done!")


if __name__ == "__main__":
    main()
