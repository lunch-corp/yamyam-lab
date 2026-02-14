"""Feature preprocessing for diner embedding model.

This module provides functions to prepare diner features for the embedding model:
- Aggregate menu text per diner
- Compute diner name embeddings
- Compute price statistics (avg, min, max)
- Generate training pairs (co-review pairs, category pairs)
- Normalize numerical features
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import AutoModel, AutoTokenizer


def prepare_diner_features(
    review_df: pd.DataFrame,
    diner_df: pd.DataFrame,
    menu_df: pd.DataFrame,
    category_df: pd.DataFrame,
    output_dir: str,
    kobert_model_name: str = "klue/bert-base",
    normalize_numerical: bool = True,
    max_menu_length: int = 512,
    max_name_length: int = 64,
    max_review_length: int = 512,
    max_reviews_per_diner: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Prepare diner features for embedding model training.

    This function aggregates and processes features for each diner:
    1. Category features (large, middle, small)
    2. Menu text aggregation and KoBERT embedding
    3. Diner name KoBERT embedding
    4. Price statistics (avg, min, max)
    5. Review text aggregation and KoBERT embedding

    Args:
        review_df: DataFrame with columns [diner_idx, reviewer_id, reviewer_review_score].
        diner_df: DataFrame with columns [diner_idx, diner_name, ...].
        menu_df: DataFrame with columns [diner_idx, menu_name, menu_price].
        category_df: DataFrame with columns [diner_idx, diner_category_large,
            diner_category_middle, diner_category_small].
        output_dir: Directory to save processed features.
        kobert_model_name: HuggingFace model name for KoBERT. Default: "klue/bert-base".
        normalize_numerical: Whether to normalize numerical features. Default: True.
        max_menu_length: Maximum token length for menu text. Default: 512.
        max_name_length: Maximum token length for diner name. Default: 64.
        max_review_length: Maximum token length for review text. Default: 512.
        max_reviews_per_diner: Max reviews per diner before subsampling. Default: 30.

    Returns:
        Tuple of (features_df, category_mapping_df, metadata_dict).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get unique diner IDs
    all_diner_ids = diner_df["diner_idx"].unique()

    # 1. Process category features
    category_features, category_encoders = _process_category_features(
        diner_df=diner_df,
        category_df=category_df,
        all_diner_ids=all_diner_ids,
    )

    # 2. Process price features
    price_features = _process_price_features(
        menu_df=menu_df,
        all_diner_ids=all_diner_ids,
        normalize=normalize_numerical,
    )

    # 3. Process menu embeddings (KoBERT)
    menu_embeddings = _process_menu_embeddings(
        menu_df=menu_df,
        all_diner_ids=all_diner_ids,
        kobert_model_name=kobert_model_name,
        max_length=max_menu_length,
    )

    # 4. Process diner name embeddings (KoBERT)
    diner_name_embeddings = _process_diner_name_embeddings(
        diner_df=diner_df,
        all_diner_ids=all_diner_ids,
        kobert_model_name=kobert_model_name,
        max_length=max_name_length,
    )

    # 5. Process review text embeddings (KoBERT)
    review_text_embeddings = _process_review_text_embeddings(
        review_df=review_df,
        all_diner_ids=all_diner_ids,
        kobert_model_name=kobert_model_name,
        max_length=max_review_length,
        max_reviews_per_diner=max_reviews_per_diner,
    )

    # Combine all features
    features_df = category_features.copy()
    features_df = features_df.merge(price_features, on="diner_idx", how="left")
    features_df = features_df.merge(menu_embeddings, on="diner_idx", how="left")
    features_df = features_df.merge(diner_name_embeddings, on="diner_idx", how="left")
    features_df = features_df.merge(review_text_embeddings, on="diner_idx", how="left")

    # Fill NaN values
    features_df = features_df.fillna(0)

    # Sort by diner_idx to ensure consistent ordering
    features_df = features_df.sort_values("diner_idx").reset_index(drop=True)

    # Create category mapping for hard negative mining
    category_mapping_df = features_df[
        ["diner_idx", "large_category_id", "middle_category_id", "small_category_id"]
    ].copy()

    # Metadata for model configuration
    metadata = {
        "num_diners": len(all_diner_ids),
        "num_large_categories": len(category_encoders["large"].classes_),
        "num_middle_categories": len(category_encoders["middle"].classes_),
        "num_small_categories": len(category_encoders["small"].classes_),
        "category_encoders": category_encoders,
    }

    # Save to parquet
    features_df.to_parquet(output_path / "diner_features.parquet", index=False)
    category_mapping_df.to_parquet(
        output_path / "category_mapping.parquet", index=False
    )

    return features_df, category_mapping_df, metadata


def _process_category_features(
    diner_df: pd.DataFrame,
    category_df: pd.DataFrame,
    all_diner_ids: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Process hierarchical category features.

    Args:
        diner_df: DataFrame with diner information.
        category_df: DataFrame with category columns.
        all_diner_ids: Array of all diner IDs.

    Returns:
        Tuple of (features_df, category_encoders_dict).
    """
    # Merge diner with category info
    if "diner_category_large" in category_df.columns:
        category_cols = [
            "diner_idx",
            "diner_category_large",
            "diner_category_middle",
            "diner_category_small",
        ]
        cat_data = category_df[
            [c for c in category_cols if c in category_df.columns]
        ].copy()
    else:
        # Use diner_df if category columns are there
        category_cols = ["diner_idx"]
        for col in [
            "diner_category_large",
            "diner_category_middle",
            "diner_category_small",
        ]:
            if col in diner_df.columns:
                category_cols.append(col)
        cat_data = diner_df[category_cols].copy()

    # Ensure all diners are present
    all_diners_df = pd.DataFrame({"diner_idx": all_diner_ids})
    cat_data = all_diners_df.merge(cat_data, on="diner_idx", how="left")

    # Fill missing categories with "unknown"
    for col in [
        "diner_category_large",
        "diner_category_middle",
        "diner_category_small",
    ]:
        if col in cat_data.columns:
            cat_data[col] = cat_data[col].fillna("unknown")
        else:
            cat_data[col] = "unknown"

    # Encode categories
    encoders = {}
    result_df = pd.DataFrame({"diner_idx": all_diner_ids})

    for level, col in [
        ("large", "diner_category_large"),
        ("middle", "diner_category_middle"),
        ("small", "diner_category_small"),
    ]:
        encoder = LabelEncoder()
        result_df[f"{level}_category_id"] = encoder.fit_transform(cat_data[col])
        encoders[level] = encoder

    return result_df, encoders


def _process_price_features(
    menu_df: pd.DataFrame,
    all_diner_ids: np.ndarray,
    normalize: bool = True,
) -> pd.DataFrame:
    """Process price statistics per diner.

    Args:
        menu_df: DataFrame with menu data.
        all_diner_ids: Array of all diner IDs.
        normalize: Whether to normalize features.

    Returns:
        DataFrame with price features.
    """
    if "menu_price" not in menu_df.columns:
        # Return placeholder if no price data
        return pd.DataFrame(
            {
                "diner_idx": all_diner_ids,
                "avg_price": 0.0,
                "min_price": 0.0,
                "max_price": 0.0,
            }
        )

    # Filter valid prices
    valid_menu = menu_df[menu_df["menu_price"] > 0].copy()

    # Aggregate price statistics
    price_agg = (
        valid_menu.groupby("diner_idx")["menu_price"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    price_agg.columns = ["diner_idx", "avg_price", "min_price", "max_price"]

    # Ensure all diners are present
    all_diners_df = pd.DataFrame({"diner_idx": all_diner_ids})
    price_features = all_diners_df.merge(price_agg, on="diner_idx", how="left")

    # Fill NaN for diners with no menu prices
    for col in ["avg_price", "min_price", "max_price"]:
        price_features[col] = price_features[col].fillna(price_agg[col].median())

    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        cols_to_normalize = ["avg_price", "min_price", "max_price"]
        price_features[cols_to_normalize] = scaler.fit_transform(
            price_features[cols_to_normalize]
        )

    return price_features


def _process_menu_embeddings(
    menu_df: pd.DataFrame,
    all_diner_ids: np.ndarray,
    kobert_model_name: str = "monologg/kobert",
    max_length: int = 512,
) -> pd.DataFrame:
    """Process menu text into KoBERT embeddings.

    Aggregates all menu names per diner and encodes using KoBERT.

    Args:
        menu_df: DataFrame with menu data.
        all_diner_ids: Array of all diner IDs.
        kobert_model_name: HuggingFace model name.
        max_length: Maximum token length.

    Returns:
        DataFrame with menu embeddings (768 dimensions).
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(kobert_model_name)
    model = AutoModel.from_pretrained(kobert_model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Aggregate menu names per diner
    if "menu_name" in menu_df.columns:
        menu_text = (
            menu_df.groupby("diner_idx")["menu_name"]
            .apply(lambda x: " ".join(x.dropna().astype(str)))
            .reset_index()
        )
        menu_text.columns = ["diner_idx", "menu_text"]
    else:
        menu_text = pd.DataFrame({"diner_idx": all_diner_ids, "menu_text": ""})

    # Ensure all diners are present
    all_diners_df = pd.DataFrame({"diner_idx": all_diner_ids})
    menu_text = all_diners_df.merge(menu_text, on="diner_idx", how="left")
    menu_text["menu_text"] = menu_text["menu_text"].fillna("")

    # Generate embeddings
    embeddings = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(menu_text), batch_size):
            batch_texts = menu_text["menu_text"].iloc[i : i + batch_size].tolist()

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embeddings
            outputs = model(**inputs)
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = (sum_hidden / sum_mask).cpu().numpy()

            embeddings.append(pooled)

    embeddings = np.vstack(embeddings)

    # Create DataFrame with embedding columns
    embedding_cols = [f"menu_{i}" for i in range(768)]
    result = pd.DataFrame({"diner_idx": all_diner_ids})
    for i, col in enumerate(embedding_cols):
        result[col] = embeddings[:, i]

    return result


def _process_diner_name_embeddings(
    diner_df: pd.DataFrame,
    all_diner_ids: np.ndarray,
    kobert_model_name: str = "klue/bert-base",
    max_length: int = 64,
) -> pd.DataFrame:
    """Process diner names into KoBERT embeddings.

    Encodes each diner's name using KoBERT.

    Args:
        diner_df: DataFrame with diner data including diner_name column.
        all_diner_ids: Array of all diner IDs.
        kobert_model_name: HuggingFace model name.
        max_length: Maximum token length.

    Returns:
        DataFrame with diner name embeddings (768 dimensions).
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(kobert_model_name)
    model = AutoModel.from_pretrained(kobert_model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get diner names
    if "diner_name" in diner_df.columns:
        name_col = "diner_name"
    elif "name" in diner_df.columns:
        name_col = "name"
    else:
        # Return placeholder if no name column
        print("Warning: No diner_name column found. Returning zero embeddings.")
        embedding_cols = [f"name_{i}" for i in range(768)]
        result = pd.DataFrame({"diner_idx": all_diner_ids})
        for col in embedding_cols:
            result[col] = 0.0
        return result

    # Prepare diner name data
    diner_names = diner_df[["diner_idx", name_col]].copy()
    diner_names.columns = ["diner_idx", "diner_name"]

    # Ensure all diners are present
    all_diners_df = pd.DataFrame({"diner_idx": all_diner_ids})
    diner_names = all_diners_df.merge(diner_names, on="diner_idx", how="left")
    diner_names["diner_name"] = diner_names["diner_name"].fillna("")

    # Generate embeddings
    embeddings = []
    batch_size = 64  # Diner names are shorter, can use larger batch

    with torch.no_grad():
        for i in range(0, len(diner_names), batch_size):
            batch_texts = diner_names["diner_name"].iloc[i : i + batch_size].tolist()

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embeddings
            outputs = model(**inputs)
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = (sum_hidden / sum_mask).cpu().numpy()

            embeddings.append(pooled)

    embeddings = np.vstack(embeddings)

    # Create DataFrame with embedding columns
    embedding_cols = [f"name_{i}" for i in range(768)]
    result = pd.DataFrame({"diner_idx": all_diner_ids})
    for i, col in enumerate(embedding_cols):
        result[col] = embeddings[:, i]

    return result


def _process_review_text_embeddings(
    review_df: pd.DataFrame,
    all_diner_ids: np.ndarray,
    kobert_model_name: str = "klue/bert-base",
    max_length: int = 512,
    max_reviews_per_diner: int = 30,
) -> pd.DataFrame:
    """Process review text into KoBERT embeddings.

    Aggregates review texts per diner (subsampled if needed) and encodes using KoBERT.

    Args:
        review_df: DataFrame with review data including reviewer_review column.
        all_diner_ids: Array of all diner IDs.
        kobert_model_name: HuggingFace model name.
        max_length: Maximum token length.
        max_reviews_per_diner: Maximum number of reviews to use per diner.
            Diners with more reviews will be subsampled. Default: 30.

    Returns:
        DataFrame with review text embeddings (768 dimensions).
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(kobert_model_name)
    model = AutoModel.from_pretrained(kobert_model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Aggregate review texts per diner (subsample if exceeding max)
    if "reviewer_review" in review_df.columns:
        filtered = review_df[["diner_idx", "reviewer_review"]].dropna(
            subset=["reviewer_review"]
        )
        if max_reviews_per_diner > 0:
            filtered = (
                filtered.groupby("diner_idx")
                .apply(
                    lambda x: x.sample(
                        n=min(len(x), max_reviews_per_diner), random_state=42
                    ),
                    include_groups=False,
                )
                .reset_index(level=0)
            )
        review_text = (
            filtered.groupby("diner_idx")["reviewer_review"]
            .apply(lambda x: " ".join(x.astype(str)))
            .reset_index()
        )
        review_text.columns = ["diner_idx", "review_text"]
    else:
        print("Warning: No reviewer_review column found. Returning zero embeddings.")
        embedding_cols = [f"review_{i}" for i in range(768)]
        result = pd.DataFrame({"diner_idx": all_diner_ids})
        for col in embedding_cols:
            result[col] = 0.0
        return result

    # Ensure all diners are present
    all_diners_df = pd.DataFrame({"diner_idx": all_diner_ids})
    review_text = all_diners_df.merge(review_text, on="diner_idx", how="left")
    review_text["review_text"] = review_text["review_text"].fillna("")

    # Generate embeddings
    embeddings = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(review_text), batch_size):
            batch_texts = review_text["review_text"].iloc[i : i + batch_size].tolist()

            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embeddings
            outputs = model(**inputs)
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = (sum_hidden / sum_mask).cpu().numpy()

            embeddings.append(pooled)

    embeddings = np.vstack(embeddings)

    # Create DataFrame with embedding columns
    embedding_cols = [f"review_{i}" for i in range(768)]
    result = pd.DataFrame({"diner_idx": all_diner_ids})
    for i, col in enumerate(embedding_cols):
        result[col] = embeddings[:, i]

    return result


def generate_training_pairs(
    review_df: pd.DataFrame,
    category_df: pd.DataFrame,
    output_dir: str,
    min_co_reviews: int = 2,
    max_pairs_per_category: int = 10000,
) -> pd.DataFrame:
    """Generate training pairs from co-review and category relationships.

    Positive pairs are generated from:
    1. Co-review pairs: Diners reviewed by the same user with high scores
    2. Category pairs: Diners in the same small category

    Args:
        review_df: DataFrame with columns [diner_idx, reviewer_id, reviewer_review_score].
        category_df: DataFrame with category columns.
        output_dir: Directory to save pairs.
        min_co_reviews: Minimum co-review count to consider as pair. Default: 2.
        max_pairs_per_category: Maximum pairs per category. Default: 10000.

    Returns:
        DataFrame with columns [anchor_idx, positive_idx, pair_type].
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pairs = []

    # 1. Generate co-review pairs
    co_review_pairs = _generate_co_review_pairs(
        review_df=review_df,
        min_co_reviews=min_co_reviews,
    )
    pairs.extend(co_review_pairs)

    # 2. Generate category pairs
    category_pairs = _generate_category_pairs(
        category_df=category_df,
        max_pairs_per_category=max_pairs_per_category,
    )
    pairs.extend(category_pairs)

    # Create DataFrame
    pairs_df = pd.DataFrame(pairs)

    # Remove duplicates
    pairs_df = pairs_df.drop_duplicates(subset=["anchor_idx", "positive_idx"])

    # Shuffle
    pairs_df = pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    pairs_df.to_parquet(output_path / "training_pairs.parquet", index=False)

    print(f"Generated {len(pairs_df)} training pairs:")
    print(pairs_df["pair_type"].value_counts())

    return pairs_df


def _generate_co_review_pairs(
    review_df: pd.DataFrame,
    min_co_reviews: int = 2,
    score_threshold: float = 3.5,
) -> List[Dict[str, Any]]:
    """Generate pairs from co-reviews (same user reviewed both diners highly).

    Args:
        review_df: Review DataFrame.
        min_co_reviews: Minimum number of common reviewers.
        score_threshold: Minimum review score to consider.

    Returns:
        List of pair dictionaries.
    """
    # Filter high-score reviews
    high_score_reviews = review_df[
        review_df["reviewer_review_score"] >= score_threshold
    ].copy()

    # Group by reviewer
    reviewer_diners = (
        high_score_reviews.groupby("reviewer_id")["diner_idx"].apply(list).to_dict()
    )

    # Build co-review counts
    co_review_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for reviewer_id, diners in reviewer_diners.items():
        if len(diners) < 2:
            continue
        # Generate pairs
        for i in range(len(diners)):
            for j in range(i + 1, len(diners)):
                pair = tuple(sorted([diners[i], diners[j]]))
                co_review_counts[pair] += 1

    # Filter by minimum co-reviews
    pairs = []
    for (diner_a, diner_b), count in co_review_counts.items():
        if count >= min_co_reviews:
            pairs.append(
                {
                    "anchor_idx": diner_a,
                    "positive_idx": diner_b,
                    "pair_type": "co_review",
                }
            )
            # Add reverse pair
            pairs.append(
                {
                    "anchor_idx": diner_b,
                    "positive_idx": diner_a,
                    "pair_type": "co_review",
                }
            )

    return pairs


def _generate_category_pairs(
    category_df: pd.DataFrame,
    max_pairs_per_category: int = 10000,
) -> List[Dict[str, Any]]:
    """Generate pairs from same middle category.

    Uses middle category instead of small category because small category
    often has too many missing values (nan/unknown).

    Args:
        category_df: Category DataFrame.
        max_pairs_per_category: Maximum pairs per category.

    Returns:
        List of pair dictionaries.
    """
    pairs = []

    # Group by middle category (more reliable than small category)
    if "diner_category_middle" in category_df.columns:
        category_col = "diner_category_middle"
    elif "middle_category_id" in category_df.columns:
        category_col = "middle_category_id"
    elif "diner_category_small" in category_df.columns:
        category_col = "diner_category_small"
    elif "small_category_id" in category_df.columns:
        category_col = "small_category_id"
    else:
        return pairs

    # Filter out NaN categories first
    filtered_df = category_df[category_df[category_col].notna()].copy()

    # Filter out categories with more than 50% of total diners (likely "unknown")
    category_counts = filtered_df[category_col].value_counts()
    max_category_size = len(category_df) * 0.5
    valid_categories = category_counts[
        category_counts <= max_category_size
    ].index.tolist()

    filtered_df = filtered_df[filtered_df[category_col].isin(valid_categories)]
    category_groups = filtered_df.groupby(category_col)["diner_idx"].apply(list)

    for category, diners in category_groups.items():
        if len(diners) < 2:
            continue

        # Sample pairs if too many
        all_possible_pairs = []
        for i in range(len(diners)):
            for j in range(i + 1, len(diners)):
                all_possible_pairs.append((diners[i], diners[j]))

        if len(all_possible_pairs) > max_pairs_per_category:
            import random

            all_possible_pairs = random.sample(
                all_possible_pairs, max_pairs_per_category
            )

        for diner_a, diner_b in all_possible_pairs:
            pairs.append(
                {
                    "anchor_idx": diner_a,
                    "positive_idx": diner_b,
                    "pair_type": "category",
                }
            )

    return pairs


def normalize_features(
    features_df: pd.DataFrame,
    columns: List[str],
    method: str = "standard",
) -> Tuple[pd.DataFrame, Any]:
    """Normalize numerical features.

    Args:
        features_df: DataFrame with features.
        columns: Columns to normalize.
        method: Normalization method ("standard" or "minmax"). Default: "standard".

    Returns:
        Tuple of (normalized_df, scaler).
    """
    df = features_df.copy()

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    df[columns] = scaler.fit_transform(df[columns])

    return df, scaler
