#!/usr/bin/env python3
"""Data preparation script for diner embedding model.

This script reads raw CSV data files and outputs preprocessed parquet files
for training the diner embedding model.

Usage:
    poetry run python scripts/prepare_diner_embedding_data.py \
        --output_dir data/processed \
        [--test]

Input files (from data/ directory or Google Drive):
    - review.csv: Review data with columns [diner_idx, reviewer_id, reviewer_review_score, ...]
    - diner.csv: Diner data with columns [diner_idx, ...]
    - menu.csv: Menu data with columns [diner_idx, menu_name, menu_price]
    - diner_category_filled.csv: Category data with ML-filled missing values [diner_idx, diner_category_large, ...]

Output files:
    - diner_features.parquet: Preprocessed features for all diners
    - training_pairs.parquet: Positive training pairs
    - val_pairs.parquet: Validation pairs
    - test_pairs.parquet: Test pairs
    - category_mapping.parquet: Category mapping for hard negative mining
"""

import argparse
from pathlib import Path

import pandas as pd

from yamyam_lab.features.diner_embedding_features import (
    generate_training_pairs,
    prepare_diner_features,
)
from yamyam_lab.tools.google_drive import check_data_and_return_paths


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare data for diner embedding model training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data files",
    )
    parser.add_argument(
        "--kobert_model_name",
        type=str,
        default="klue/bert-base",
        help="HuggingFace model name for Korean BERT",
    )
    parser.add_argument(
        "--normalize_numerical",
        action="store_true",
        default=True,
        help="Whether to normalize numerical features",
    )
    parser.add_argument(
        "--max_menu_length",
        type=int,
        default=512,
        help="Maximum token length for menu text",
    )
    parser.add_argument(
        "--min_co_reviews",
        type=int,
        default=2,
        help="Minimum co-review count for training pairs",
    )
    parser.add_argument(
        "--max_pairs_per_category",
        type=int,
        default=10000,
        help="Maximum pairs per category",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of pairs to use for validation",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Ratio of pairs to use for testing",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with subset of data",
    )
    parser.add_argument(
        "--local_data_dir",
        type=str,
        default=None,
        help="Local directory containing CSV files (if not using Google Drive)",
    )

    return parser.parse_args()


def load_data(args) -> dict:
    """Load raw data files.

    Args:
        args: Command line arguments.

    Returns:
        Dictionary containing DataFrames for review, diner, menu, category.
    """
    if args.local_data_dir:
        # Load from local directory
        data_dir = Path(args.local_data_dir)
        review_df = pd.read_csv(data_dir / "review.csv")
        diner_df = pd.read_csv(data_dir / "diner.csv", low_memory=False)
        category_df = pd.read_csv(data_dir / "diner_category_filled.csv")

        # Menu might not exist
        menu_path = data_dir / "menu_df.csv"
        if menu_path.exists():
            menu_df = pd.read_csv(menu_path)
        else:
            print("Warning: menu.csv not found. Using empty menu data.")
            menu_df = pd.DataFrame(columns=["diner_idx", "menu_name", "menu_price"])
    else:
        # Load from Google Drive
        data_paths = check_data_and_return_paths()
        review_df = pd.read_csv(data_paths["review"])
        diner_df = pd.read_csv(data_paths["diner"], low_memory=False)
        category_df = pd.read_csv(data_paths["category"])

        # Menu might not exist in Google Drive paths
        if "menu" in data_paths:
            menu_df = pd.read_csv(data_paths["menu"])
        else:
            print(
                "Warning: menu data not found in Google Drive. Using empty menu data."
            )
            menu_df = pd.DataFrame(columns=["diner_idx", "menu_name", "menu_price"])

    # Merge reviewer data if available
    if args.local_data_dir:
        reviewer_path = Path(args.local_data_dir) / "reviewer.csv"
        if reviewer_path.exists():
            reviewer_df = pd.read_csv(reviewer_path)
            review_df = pd.merge(review_df, reviewer_df, on="reviewer_id", how="left")
    else:
        if "reviewer" in data_paths:
            reviewer_df = pd.read_csv(data_paths["reviewer"])
            review_df = pd.merge(review_df, reviewer_df, on="reviewer_id", how="left")

    # Filter for test mode
    if args.test:
        print("Running in test mode - using subset of data")
        # Get a small subset of diners
        yongsan_diners = diner_df[
            diner_df["diner_road_address"].str.startswith("서울 용산구", na=False)
        ]["diner_idx"].unique()[:100]

        review_df = review_df[review_df["diner_idx"].isin(yongsan_diners)]
        diner_df = diner_df[diner_df["diner_idx"].isin(yongsan_diners)]
        category_df = category_df[category_df["diner_idx"].isin(yongsan_diners)]
        menu_df = menu_df[menu_df["diner_idx"].isin(yongsan_diners)]

    return {
        "review": review_df,
        "diner": diner_df,
        "menu": menu_df,
        "category": category_df,
    }


def split_pairs(
    pairs_df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple:
    """Split pairs into train, validation, and test sets.

    Args:
        pairs_df: DataFrame with training pairs.
        val_ratio: Ratio for validation set.
        test_ratio: Ratio for test set.
        random_seed: Random seed.

    Returns:
        Tuple of (train_pairs, val_pairs, test_pairs) DataFrames.
    """
    # Shuffle pairs
    pairs_df = pairs_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    n_total = len(pairs_df)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)

    test_pairs = pairs_df.iloc[:n_test]
    val_pairs = pairs_df.iloc[n_test : n_test + n_val]
    train_pairs = pairs_df.iloc[n_test + n_val :]

    return train_pairs, val_pairs, test_pairs


def main():
    """Main function to prepare diner embedding data."""
    args = parse_args()

    print("=" * 60)
    print("Diner Embedding Data Preparation")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/4] Loading raw data...")
    data = load_data(args)

    print(f"  - Reviews: {len(data['review'])} rows")
    print(f"  - Diners: {len(data['diner'])} rows")
    print(f"  - Menu items: {len(data['menu'])} rows")
    print(f"  - Categories: {len(data['category'])} rows")

    # Prepare features
    print("\n[2/4] Preparing diner features...")
    features_df, category_mapping_df, metadata = prepare_diner_features(
        review_df=data["review"],
        diner_df=data["diner"],
        menu_df=data["menu"],
        category_df=data["category"],
        output_dir=str(output_dir),
        kobert_model_name=args.kobert_model_name,
        normalize_numerical=args.normalize_numerical,
        max_menu_length=args.max_menu_length,
    )

    print(f"  - Features prepared for {len(features_df)} diners")
    print(f"  - Large categories: {metadata['num_large_categories']}")
    print(f"  - Middle categories: {metadata['num_middle_categories']}")
    print(f"  - Small categories: {metadata['num_small_categories']}")

    # Generate training pairs
    print("\n[3/4] Generating training pairs...")
    pairs_df = generate_training_pairs(
        review_df=data["review"],
        category_df=category_mapping_df,
        output_dir=str(output_dir),
        min_co_reviews=args.min_co_reviews,
        max_pairs_per_category=args.max_pairs_per_category,
    )

    print(f"  - Total pairs: {len(pairs_df)}")

    # Split into train/val/test
    print("\n[4/4] Splitting pairs into train/val/test...")
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs_df=pairs_df,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
    )

    # Save split pairs
    train_pairs.to_parquet(output_dir / "training_pairs.parquet", index=False)
    val_pairs.to_parquet(output_dir / "val_pairs.parquet", index=False)
    test_pairs.to_parquet(output_dir / "test_pairs.parquet", index=False)

    print(f"  - Training pairs: {len(train_pairs)}")
    print(f"  - Validation pairs: {len(val_pairs)}")
    print(f"  - Test pairs: {len(test_pairs)}")

    # Summary
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nOutput files saved to: {output_dir}")
    print("  - diner_features.parquet")
    print("  - category_mapping.parquet")
    print("  - training_pairs.parquet")
    print("  - val_pairs.parquet")
    print("  - test_pairs.parquet")

    print("\nTo train the model, run:")
    print(
        f"  poetry run python -m yamyam_lab.train --model diner_embedding "
        f"--features_path {output_dir}/diner_features.parquet "
        f"--pairs_path {output_dir}/training_pairs.parquet"
    )


if __name__ == "__main__":
    main()
