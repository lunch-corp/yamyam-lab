#!/usr/bin/env python3
"""Qualitative evaluation script for diner embedding model.

This script allows you to inspect top-N similar diners for a given anchor diner,
displaying diner names, categories, and similarity scores.

Usage:
    poetry run python scripts/evaluate_diner_similarity.py \
        --model_path result/untest/diner_embedding/<timestamp>/weight.pt \
        --diner_idx 12345 \
        --top_n 10
"""

import argparse
from typing import Optional

import pandas as pd
import torch

from yamyam_lab.model.graph.diner_embedding import Model


def load_model_and_data(
    model_path: str,
    features_path: str = "data/processed/diner_features.parquet",
    diner_csv_path: str = "data/diner.csv",
    category_csv_path: str = "data/diner_category.csv",
    device: str = "cpu",
):
    """Load trained model and diner metadata.

    Args:
        model_path: Path to trained model weights (.pt file).
        features_path: Path to preprocessed features parquet.
        diner_csv_path: Path to diner.csv with diner names.
        category_csv_path: Path to diner_category.csv with categories.
        device: Device to load model on.

    Returns:
        Tuple of (model, features_df, diner_df, category_df, diner_idx_to_pos).
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    config.device = device

    # Create and load model
    model = Model(config=config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load features
    features_df = pd.read_parquet(features_path)

    # Build diner_idx to position mapping
    diner_idx_to_pos = {
        diner_idx: pos for pos, diner_idx in enumerate(features_df["diner_idx"].values)
    }

    # Build feature tensors for embedding computation
    all_features = _build_feature_tensors(features_df, device)

    # Compute and store embeddings
    model.compute_and_store_embeddings(all_features, batch_size=256)

    # Load diner metadata
    diner_df = pd.read_csv(diner_csv_path, low_memory=False)
    category_df = pd.read_csv(category_csv_path)

    return model, features_df, diner_df, category_df, diner_idx_to_pos


def _build_feature_tensors(features_df: pd.DataFrame, device: str):
    """Build feature tensors from dataframe."""
    # Category features
    large_category_ids = torch.tensor(
        features_df["large_category_id"].values, dtype=torch.long
    )
    middle_category_ids = torch.tensor(
        features_df["middle_category_id"].values, dtype=torch.long
    )
    small_category_ids = torch.tensor(
        features_df["small_category_id"].values, dtype=torch.long
    )

    # Menu embeddings
    menu_cols = [col for col in features_df.columns if col.startswith("menu_")]
    if menu_cols:
        menu_embeddings = torch.tensor(
            features_df[menu_cols].values, dtype=torch.float32
        )
    else:
        menu_embeddings = torch.zeros(len(features_df), 768, dtype=torch.float32)

    # Diner name embeddings
    name_cols = [col for col in features_df.columns if col.startswith("name_")]
    if name_cols:
        diner_name_embeddings = torch.tensor(
            features_df[name_cols].values, dtype=torch.float32
        )
    else:
        diner_name_embeddings = torch.zeros(len(features_df), 768, dtype=torch.float32)

    # Price features
    price_cols = ["avg_price", "min_price", "max_price"]
    if all(col in features_df.columns for col in price_cols):
        price_features = torch.tensor(
            features_df[price_cols].values, dtype=torch.float32
        )
    else:
        price_features = torch.zeros(len(features_df), 3, dtype=torch.float32)

    return {
        "large_category_ids": large_category_ids,
        "middle_category_ids": middle_category_ids,
        "small_category_ids": small_category_ids,
        "menu_embeddings": menu_embeddings,
        "diner_name_embeddings": diner_name_embeddings,
        "price_features": price_features,
    }


def show_similar_diners(
    anchor_diner_idx: int,
    model: Model,
    features_df: pd.DataFrame,
    diner_df: pd.DataFrame,
    category_df: pd.DataFrame,
    diner_idx_to_pos: dict,
    top_n: int = 10,
) -> Optional[pd.DataFrame]:
    """Show top-N similar diners for a given anchor diner.

    Args:
        anchor_diner_idx: The diner_idx of the anchor diner.
        model: Trained diner embedding model.
        features_df: DataFrame with preprocessed features.
        diner_df: DataFrame with diner metadata (names, etc.).
        category_df: DataFrame with category information.
        diner_idx_to_pos: Mapping from diner_idx to position index.
        top_n: Number of similar diners to return.

    Returns:
        DataFrame with similar diners or None if anchor not found.
    """
    # Check if anchor exists
    if anchor_diner_idx not in diner_idx_to_pos:
        print(f"Error: diner_idx {anchor_diner_idx} not found in features.")
        return None

    anchor_pos = diner_idx_to_pos[anchor_diner_idx]

    # Get anchor info
    anchor_diner = diner_df[diner_df["diner_idx"] == anchor_diner_idx]
    anchor_category = category_df[category_df["diner_idx"] == anchor_diner_idx]

    if len(anchor_diner) == 0:
        print(f"Error: diner_idx {anchor_diner_idx} not found in diner.csv")
        return None

    anchor_name = anchor_diner["diner_name"].values[0]
    anchor_large = (
        anchor_category["diner_category_large"].values[0]
        if len(anchor_category) > 0
        else "Unknown"
    )
    anchor_middle = (
        anchor_category["diner_category_middle"].values[0]
        if len(anchor_category) > 0
        else "Unknown"
    )
    anchor_small = (
        anchor_category["diner_category_small"].values[0]
        if len(anchor_category) > 0
        else "Unknown"
    )

    print("=" * 80)
    print(f"Anchor Diner (idx={anchor_diner_idx}):")
    print(f"  Name: {anchor_name}")
    print(f"  Category: {anchor_large} > {anchor_middle} > {anchor_small}")
    print("=" * 80)

    # Get anchor embedding
    anchor_emb = model._embedding[anchor_pos : anchor_pos + 1]

    # Compute similarities
    similarities = model.similarity(anchor_emb, model._embedding).squeeze(0)

    # Exclude anchor itself
    similarities[anchor_pos] = -float("inf")

    # Get top-N
    top_k_result = torch.topk(similarities, k=top_n)
    top_positions = top_k_result.indices.cpu().numpy()
    top_scores = top_k_result.values.cpu().numpy()

    # Map positions back to diner_idx
    pos_to_diner_idx = {pos: idx for idx, pos in diner_idx_to_pos.items()}

    # Collect results
    results = []
    print(f"\nTop-{top_n} Similar Diners:")
    print("-" * 80)

    for rank, (pos, score) in enumerate(zip(top_positions, top_scores), 1):
        similar_diner_idx = pos_to_diner_idx.get(pos)
        if similar_diner_idx is None:
            continue

        # Get diner info
        similar_diner = diner_df[diner_df["diner_idx"] == similar_diner_idx]
        similar_category = category_df[category_df["diner_idx"] == similar_diner_idx]

        if len(similar_diner) == 0:
            continue

        name = similar_diner["diner_name"].values[0]
        large = (
            similar_category["diner_category_large"].values[0]
            if len(similar_category) > 0
            else "Unknown"
        )
        middle = (
            similar_category["diner_category_middle"].values[0]
            if len(similar_category) > 0
            else "Unknown"
        )
        small = (
            similar_category["diner_category_small"].values[0]
            if len(similar_category) > 0
            else "Unknown"
        )

        # Check if same category as anchor
        same_large = "✓" if large == anchor_large else ""
        same_middle = "✓" if middle == anchor_middle else ""

        print(f"{rank:2d}. [{score:.4f}] {name}")
        print(f"    Category: {large} > {middle} > {small} {same_large}{same_middle}")

        results.append(
            {
                "rank": rank,
                "diner_idx": similar_diner_idx,
                "diner_name": name,
                "category_large": large,
                "category_middle": middle,
                "category_small": small,
                "similarity_score": score,
                "same_large_category": large == anchor_large,
                "same_middle_category": middle == anchor_middle,
            }
        )

    print("-" * 80)

    return pd.DataFrame(results)


def search_diner_by_name(
    query: str,
    diner_df: pd.DataFrame,
    diner_idx_to_pos: dict,
    max_results: int = 10,
) -> pd.DataFrame:
    """Search for diners by name substring.

    Args:
        query: Search query (case-insensitive substring match).
        diner_df: DataFrame with diner metadata.
        diner_idx_to_pos: Mapping of valid diner indices.
        max_results: Maximum number of results to return.

    Returns:
        DataFrame with matching diners.
    """
    # Filter to only diners in our feature set
    valid_diners = diner_df[diner_df["diner_idx"].isin(diner_idx_to_pos.keys())]

    # Search by name
    matches = valid_diners[
        valid_diners["diner_name"].str.contains(query, case=False, na=False)
    ]

    if len(matches) == 0:
        print(f"No diners found matching '{query}'")
        return pd.DataFrame()

    results = matches[["diner_idx", "diner_name"]].head(max_results)
    print(f"Found {len(matches)} diners matching '{query}':")
    print(results.to_string(index=False))

    return results


def main():
    """Main function for qualitative evaluation."""
    parser = argparse.ArgumentParser(
        description="Qualitative evaluation of diner embedding model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model weights (.pt file)",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default="data/processed/diner_features.parquet",
        help="Path to preprocessed features parquet",
    )
    parser.add_argument(
        "--diner_csv",
        type=str,
        default="data/diner.csv",
        help="Path to diner.csv",
    )
    parser.add_argument(
        "--category_csv",
        type=str,
        default="data/diner_category.csv",
        help="Path to diner_category.csv",
    )
    parser.add_argument(
        "--diner_idx",
        type=int,
        default=None,
        help="Anchor diner index to query",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search for diners by name",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of similar diners to show",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )

    args = parser.parse_args()

    print("Loading model and data...")
    model, features_df, diner_df, category_df, diner_idx_to_pos = load_model_and_data(
        model_path=args.model_path,
        features_path=args.features_path,
        diner_csv_path=args.diner_csv,
        category_csv_path=args.category_csv,
        device=args.device,
    )
    print(f"Loaded {len(diner_idx_to_pos)} diners with embeddings.\n")

    if args.search:
        search_diner_by_name(args.search, diner_df, diner_idx_to_pos)

    if args.diner_idx is not None:
        show_similar_diners(
            anchor_diner_idx=args.diner_idx,
            model=model,
            features_df=features_df,
            diner_df=diner_df,
            category_df=category_df,
            diner_idx_to_pos=diner_idx_to_pos,
            top_n=args.top_n,
        )

    if args.interactive:
        print("\nInteractive mode. Commands:")
        print("  search <query>  - Search diners by name")
        print("  show <idx>      - Show similar diners for diner_idx")
        print("  quit            - Exit")
        print()

        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue

            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "quit" or cmd == "exit":
                break
            elif cmd == "search" and len(parts) > 1:
                search_diner_by_name(parts[1], diner_df, diner_idx_to_pos)
            elif cmd == "show" and len(parts) > 1:
                try:
                    diner_idx = int(parts[1])
                    show_similar_diners(
                        anchor_diner_idx=diner_idx,
                        model=model,
                        features_df=features_df,
                        diner_df=diner_df,
                        category_df=category_df,
                        diner_idx_to_pos=diner_idx_to_pos,
                        top_n=args.top_n,
                    )
                except ValueError:
                    print("Invalid diner_idx. Use 'show <number>'")
            else:
                print("Unknown command. Use 'search <query>', 'show <idx>', or 'quit'")


if __name__ == "__main__":
    main()
