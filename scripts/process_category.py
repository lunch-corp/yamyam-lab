"""Script to process diner category data using CategoryProcessor and MiddleCategorySimplifier."""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import pandas as pd

from yamyam_lab.preprocess.diner_transform import (
    CategoryProcessor,
    MiddleCategorySimplifier,
)

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_CONFIG_PATH = os.path.join(ROOT_PATH, "config")
DEFAULT_DATA_PATH = os.path.join(ROOT_PATH, "src/data")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process diner category CSV using CategoryProcessor and MiddleCategorySimplifier"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input diner_category.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output diner_category_processed.csv",
    )
    parser.add_argument(
        "--config-root-path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Root path for config files (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to data directory for MiddleCategorySimplifier (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--skip-simplifier",
        action="store_true",
        help="Skip MiddleCategorySimplifier step",
    )
    return parser.parse_args()


def main(args):
    # Load input CSV
    print(f"Loading input file: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")

    # Step 1: Run CategoryProcessor
    print("Running CategoryProcessor...")
    processor = CategoryProcessor(
        df=df,
        config_root_path=args.config_root_path,
    )
    processor.process_all()
    processed_df = processor.category_preprocessed_diners
    print("CategoryProcessor completed")

    # Step 2: Run MiddleCategorySimplifier (optional)
    if not args.skip_simplifier:
        print("Running MiddleCategorySimplifier...")
        simplifier = MiddleCategorySimplifier(
            config_root_path=args.config_root_path,
            data_path=args.data_path,
        )
        processed_df = simplifier.process(processed_df)
        print("MiddleCategorySimplifier completed")

    # Save output CSV
    print(f"Saving output file: {args.output}")
    processed_df.to_csv(args.output, index=False)
    print(f"Saved {len(processed_df)} rows")


if __name__ == "__main__":
    args = parse_args()
    main(args)
