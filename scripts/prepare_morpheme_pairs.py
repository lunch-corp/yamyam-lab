#!/usr/bin/env python3
"""Preprocessing script for morpheme-based positive pair generation.

Extracts morphemes from diner reviews, builds a binary morpheme matrix,
computes pairwise Jaccard similarity within the same middle category,
and outputs train/val/test pair parquet files.

Usage:
    poetry run python scripts/prepare_morpheme_pairs.py \
        --output_dir data/processed \
        [--jaccard_threshold 0.3] \
        [--test]
"""

import argparse
from pathlib import Path

import pandas as pd

from scripts.prepare_diner_embedding_data import split_pairs
from yamyam_lab.features.morpheme_cooccurrence import (
    analyze_threshold,
    build_morpheme_matrix,
    compute_jaccard_pairs,
    tokenize_reviews_parallel,
)
from yamyam_lab.tools.google_drive import check_data_and_return_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate morpheme-based positive pairs for embedding training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed pair files",
    )
    parser.add_argument(
        "--jaccard_threshold",
        type=float,
        default=0.3,
        help="Minimum Jaccard similarity for positive pairs",
    )
    parser.add_argument(
        "--min_df",
        type=int,
        default=3,
        help="Minimum document frequency for morpheme vocabulary",
    )
    parser.add_argument(
        "--max_df",
        type=float,
        default=0.5,
        help="Maximum document frequency (fraction) for morpheme vocabulary",
    )
    parser.add_argument(
        "--included_tags",
        nargs="+",
        default=["NNG", "NNP", "VA"],
        help="POS tags to keep",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio for validation set",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Ratio for test set",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=None,
        help="Number of worker processes for morpheme extraction",
    )
    parser.add_argument(
        "--local_data_dir",
        type=str,
        default=None,
        help="Local directory containing CSV files (if not using Google Drive)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with subset of data",
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only run threshold analysis, do not save pairs",
    )
    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        help="Skip threshold analysis and directly compute pairs",
    )
    return parser.parse_args()


def load_data(args) -> dict:
    """Load review, diner, and category DataFrames."""
    if args.local_data_dir:
        data_dir = Path(args.local_data_dir)
        review_df = pd.read_csv(data_dir / "review.csv")
        diner_df = pd.read_csv(data_dir / "diner.csv", low_memory=False)
        category_df = pd.read_csv(data_dir / "diner_category_filled.csv")
    else:
        data_paths = check_data_and_return_paths()
        review_df = pd.read_csv(data_paths["review"])
        diner_df = pd.read_csv(data_paths["diner"], low_memory=False)
        category_df = pd.read_csv(data_paths["category"])

    if args.test:
        print("Running in test mode — using subset of data")
        subset_diners = diner_df["diner_idx"].unique()[:100]
        review_df = review_df[review_df["diner_idx"].isin(subset_diners)]
        diner_df = diner_df[diner_df["diner_idx"].isin(subset_diners)]
        category_df = category_df[category_df["diner_idx"].isin(subset_diners)]

    return {"review": review_df, "diner": diner_df, "category": category_df}


def main():
    args = parse_args()

    print("=" * 60)
    print("Morpheme-Based Positive Pair Generation")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("\n[1/5] Loading data...")
    data = load_data(args)
    print(f"  Reviews: {len(data['review'])}")
    print(f"  Diners: {len(data['diner'])}")

    # 2. Extract morphemes
    print("\n[2/5] Extracting morphemes (parallel)...")
    diner_morphemes = tokenize_reviews_parallel(
        review_df=data["review"],
        included_tags=args.included_tags,
        n_processes=args.n_processes,
    )
    print(f"  Diners with morphemes: {len(diner_morphemes)}")

    # 3. Build morpheme matrix
    print("\n[3/5] Building morpheme matrix...")
    morpheme_matrix, vocab, diner_idx_to_pos = build_morpheme_matrix(
        diner_morphemes=diner_morphemes,
        min_df=args.min_df,
        max_df=args.max_df,
        save_dir=str(output_dir),
    )
    print(f"  Matrix shape: {morpheme_matrix.shape}")
    print(f"  Vocabulary size: {len(vocab)}")

    # 4. Prepare category mapping with middle_category_id
    # The category_df may have raw names; we need numeric IDs
    cat_df = data["category"].copy()
    if "middle_category_id" not in cat_df.columns:
        if "diner_category_middle" in cat_df.columns:
            cat_df["middle_category_id"] = (
                cat_df["diner_category_middle"].astype("category").cat.codes
            )
        else:
            raise ValueError(
                "category_df must have 'middle_category_id' or 'diner_category_middle'"
            )

    # 5. Threshold analysis
    if not args.skip_analysis:
        print("\n[4/5] Threshold analysis...")
        analyze_threshold(
            morpheme_matrix=morpheme_matrix,
            category_df=cat_df,
            diner_idx_to_pos=diner_idx_to_pos,
            diner_df=data["diner"],
        )

    if args.analyze_only:
        print("\nAnalysis-only mode. Exiting.")
        return

    # 6. Compute pairs and split
    print(f"\n[5/5] Computing pairs (threshold={args.jaccard_threshold})...")
    pairs_df = compute_jaccard_pairs(
        morpheme_matrix=morpheme_matrix,
        category_df=cat_df,
        diner_idx_to_pos=diner_idx_to_pos,
        threshold=args.jaccard_threshold,
    )
    print(f"  Total pairs (bidirectional): {len(pairs_df)}")

    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs_df=pairs_df,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
    )

    train_pairs.to_parquet(output_dir / "training_pairs.parquet", index=False)
    val_pairs.to_parquet(output_dir / "val_pairs.parquet", index=False)
    test_pairs.to_parquet(output_dir / "test_pairs.parquet", index=False)

    print(f"\n  Training pairs: {len(train_pairs)}")
    print(f"  Validation pairs: {len(val_pairs)}")
    print(f"  Test pairs: {len(test_pairs)}")

    print("\n" + "=" * 60)
    print("Done! Pair files saved to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
