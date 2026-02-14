"""Dataset and DataLoader for multimodal triplet embedding model.

This module provides the dataset class and data loading utilities for
training the multimodal triplet embedding model with triplet loss.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class MultimodalTripletDataset(Dataset):
    """Dataset for multimodal triplet embedding training with triplet loss.

    This dataset generates triplets (anchor, positive, negative) for training.
    Positive pairs are diners that are semantically similar (e.g., co-reviewed
    by same users, same category). Negatives are sampled using hard negative
    mining strategy.

    Hard negative mining strategy:
    - 5 same-category negatives (hard)
    - 3 nearby-category negatives (semi-hard)
    - 2 random negatives (easy)

    Args:
        features_path: Path to preprocessed features parquet file.
        pairs_path: Path to training pairs parquet file.
        category_mapping_path: Path to category mapping parquet file.
        num_hard_negatives: Number of same-category hard negatives. Default: 5.
        num_nearby_negatives: Number of nearby-category negatives. Default: 3.
        num_random_negatives: Number of random negatives. Default: 2.
        random_seed: Random seed for reproducibility. Default: 42.
    """

    def __init__(
        self,
        features_path: str,
        pairs_path: str,
        category_mapping_path: str,
        num_hard_negatives: int = 5,
        num_nearby_negatives: int = 3,
        num_random_negatives: int = 2,
        random_seed: int = 42,
    ):
        self.features_path = Path(features_path)
        self.pairs_path = Path(pairs_path)
        self.category_mapping_path = Path(category_mapping_path)
        self.num_hard_negatives = num_hard_negatives
        self.num_nearby_negatives = num_nearby_negatives
        self.num_random_negatives = num_random_negatives
        self.total_negatives = (
            num_hard_negatives + num_nearby_negatives + num_random_negatives
        )

        random.seed(random_seed)
        np.random.seed(random_seed)

        # Load data
        self._load_data()

    def _load_data(self) -> None:
        """Load preprocessed features and training pairs."""
        # Load features
        self.features_df = pd.read_parquet(self.features_path)
        self.num_diners = len(self.features_df)

        # Build mapping from diner_idx to position index
        self.diner_idx_to_position: Dict[int, int] = {
            diner_idx: pos
            for pos, diner_idx in enumerate(self.features_df["diner_idx"].values)
        }
        self.all_diner_indices = list(self.diner_idx_to_position.keys())

        # Load positive pairs
        self.pairs_df = pd.read_parquet(self.pairs_path)

        # Filter pairs to only include diners we have features for
        valid_diners = set(self.diner_idx_to_position.keys())
        self.pairs_df = self.pairs_df[
            self.pairs_df["anchor_idx"].isin(valid_diners)
            & self.pairs_df["positive_idx"].isin(valid_diners)
        ].reset_index(drop=True)

        # Load category mapping for hard negative mining
        self.category_df = pd.read_parquet(self.category_mapping_path)

        # Build category indices for fast lookup
        self._build_category_indices()

        # Build feature tensors
        self._build_feature_tensors()

    def _build_category_indices(self) -> None:
        """Build mapping from category to diner indices for hard negative mining."""
        # Group diners by large category (only include diners we have features for)
        valid_diners = set(self.diner_idx_to_position.keys())
        self.category_to_diners: Dict[int, List[int]] = {}
        for idx, row in self.category_df.iterrows():
            cat_id = row["large_category_id"]
            diner_idx = row["diner_idx"]
            if diner_idx not in valid_diners:
                continue
            if cat_id not in self.category_to_diners:
                self.category_to_diners[cat_id] = []
            self.category_to_diners[cat_id].append(diner_idx)

        # Build nearby category mapping (for semi-hard negatives)
        # Categories are considered "nearby" if they share similar characteristics
        # For simplicity, we use the middle category as a proxy
        self.category_to_nearby: Dict[int, List[int]] = {}
        if "middle_category_id" in self.category_df.columns:
            middle_to_large = self.category_df.groupby("middle_category_id")[
                "large_category_id"
            ].unique()
            for large_cat in self.category_to_diners.keys():
                nearby = set()
                for middle_cat, large_cats in middle_to_large.items():
                    if large_cat in large_cats:
                        nearby.update(large_cats)
                # Remove self
                nearby.discard(large_cat)
                self.category_to_nearby[large_cat] = list(nearby)
        else:
            # Default: all other categories are nearby
            all_cats = list(self.category_to_diners.keys())
            for cat in all_cats:
                self.category_to_nearby[cat] = [c for c in all_cats if c != cat]

        # Create diner to category mapping
        self.diner_to_category: Dict[int, int] = dict(
            zip(self.category_df["diner_idx"], self.category_df["large_category_id"])
        )

    def _build_feature_tensors(self) -> None:
        """Build feature tensors from dataframe."""
        # Category features
        self.large_category_ids = torch.tensor(
            self.features_df["large_category_id"].values, dtype=torch.long
        )
        self.middle_category_ids = torch.tensor(
            self.features_df["middle_category_id"].values, dtype=torch.long
        )
        self.small_category_ids = torch.tensor(
            self.features_df["small_category_id"].values, dtype=torch.long
        )

        # Menu embeddings (precomputed KoBERT)
        menu_cols = [col for col in self.features_df.columns if col.startswith("menu_")]
        if menu_cols:
            self.menu_embeddings = torch.tensor(
                self.features_df[menu_cols].values, dtype=torch.float32
            )
        else:
            # Placeholder if menu embeddings not available
            self.menu_embeddings = torch.zeros(
                self.num_diners, 768, dtype=torch.float32
            )

        # Diner name embeddings (precomputed KoBERT)
        name_cols = [col for col in self.features_df.columns if col.startswith("name_")]
        if name_cols:
            self.diner_name_embeddings = torch.tensor(
                self.features_df[name_cols].values, dtype=torch.float32
            )
        else:
            # Placeholder if diner name embeddings not available
            self.diner_name_embeddings = torch.zeros(
                self.num_diners, 768, dtype=torch.float32
            )

        # Price features
        price_cols = ["avg_price", "min_price", "max_price"]
        if all(col in self.features_df.columns for col in price_cols):
            self.price_features = torch.tensor(
                self.features_df[price_cols].values, dtype=torch.float32
            )
        else:
            self.price_features = torch.zeros(self.num_diners, 3, dtype=torch.float32)

        # Review text embeddings (precomputed KoBERT)
        review_cols = [
            col for col in self.features_df.columns if col.startswith("review_")
        ]
        if review_cols:
            self.review_text_embeddings = torch.tensor(
                self.features_df[review_cols].values, dtype=torch.float32
            )
        else:
            # Placeholder if review text embeddings not available
            self.review_text_embeddings = torch.zeros(
                self.num_diners, 768, dtype=torch.float32
            )

    def __len__(self) -> int:
        """Return number of positive pairs (each pair generates one triplet)."""
        return len(self.pairs_df)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Get a training triplet.

        Args:
            idx: Index of the positive pair.

        Returns:
            Dictionary containing:
                - anchor_idx: Anchor diner index
                - positive_idx: Positive diner index
                - negative_indices: List of negative diner indices
                - anchor_category: Anchor's large category ID
                - positive_category: Positive's large category ID
                - negative_categories: List of negative category IDs
        """
        row = self.pairs_df.iloc[idx]
        anchor_diner_idx = int(row["anchor_idx"])
        positive_diner_idx = int(row["positive_idx"])

        # Convert diner_idx to position indices
        anchor_pos = self.diner_idx_to_position.get(anchor_diner_idx, 0)
        positive_pos = self.diner_idx_to_position.get(positive_diner_idx, 0)

        # Get anchor category
        anchor_category = self.diner_to_category.get(anchor_diner_idx, 0)

        # Sample negatives using hard negative mining (returns diner_idx values)
        negative_diner_indices = self._sample_negatives(
            anchor_diner_idx, positive_diner_idx
        )
        # Convert to position indices
        negative_positions = [
            self.diner_idx_to_position.get(neg_idx, 0)
            for neg_idx in negative_diner_indices
        ]
        negative_categories = [
            self.diner_to_category.get(neg_idx, 0) for neg_idx in negative_diner_indices
        ]

        return {
            "anchor_idx": torch.tensor(anchor_pos, dtype=torch.long),
            "positive_idx": torch.tensor(positive_pos, dtype=torch.long),
            "negative_indices": torch.tensor(negative_positions, dtype=torch.long),
            "anchor_category": torch.tensor(anchor_category, dtype=torch.long),
            "positive_category": torch.tensor(
                self.diner_to_category.get(positive_diner_idx, 0), dtype=torch.long
            ),
            "negative_categories": torch.tensor(negative_categories, dtype=torch.long),
        }

    def _sample_negatives(self, anchor_idx: int, positive_idx: int) -> List[int]:
        """Sample negative examples using hard negative mining strategy.

        Strategy:
        - 5 same-category negatives (hard)
        - 3 nearby-category negatives (semi-hard)
        - 2 random negatives (easy)

        Args:
            anchor_idx: Anchor diner index.
            positive_idx: Positive diner index.

        Returns:
            List of negative diner indices.
        """
        negatives = []
        excluded = {anchor_idx, positive_idx}
        anchor_category = self.diner_to_category.get(anchor_idx, 0)

        # 1. Same-category hard negatives
        same_cat_diners = [
            d
            for d in self.category_to_diners.get(anchor_category, [])
            if d not in excluded
        ]
        if same_cat_diners:
            k = min(self.num_hard_negatives, len(same_cat_diners))
            hard_negs = random.sample(same_cat_diners, k)
            negatives.extend(hard_negs)
            excluded.update(hard_negs)

        # 2. Nearby-category semi-hard negatives
        nearby_cats = self.category_to_nearby.get(anchor_category, [])
        nearby_diners = []
        for cat in nearby_cats:
            nearby_diners.extend(
                [d for d in self.category_to_diners.get(cat, []) if d not in excluded]
            )
        if nearby_diners:
            k = min(self.num_nearby_negatives, len(nearby_diners))
            nearby_negs = random.sample(nearby_diners, k)
            negatives.extend(nearby_negs)
            excluded.update(nearby_negs)

        # 3. Random easy negatives (use all valid diner_idx values)
        random_pool = [d for d in self.all_diner_indices if d not in excluded]
        if random_pool:
            k = min(self.num_random_negatives, len(random_pool))
            random_negs = random.sample(random_pool, k)
            negatives.extend(random_negs)
            excluded.update(random_negs)

        # Pad with random if not enough negatives
        while len(negatives) < self.total_negatives:
            remaining_pool = [d for d in self.all_diner_indices if d not in excluded]
            if remaining_pool:
                neg = random.choice(remaining_pool)
                negatives.append(neg)
                excluded.add(neg)
                excluded.add(neg)
            else:
                # Fallback: repeat existing negatives
                if negatives:
                    negatives.append(random.choice(negatives))
                else:
                    negatives.append(0)

        return negatives[: self.total_negatives]

    def get_all_features(self) -> Dict[str, Tensor]:
        """Get all feature tensors for embedding computation.

        Returns:
            Dictionary containing all feature tensors.
        """
        return {
            "large_category_ids": self.large_category_ids,
            "middle_category_ids": self.middle_category_ids,
            "small_category_ids": self.small_category_ids,
            "menu_embeddings": self.menu_embeddings,
            "diner_name_embeddings": self.diner_name_embeddings,
            "price_features": self.price_features,
            "review_text_embeddings": self.review_text_embeddings,
        }

    def get_features_by_indices(self, indices: Tensor) -> Dict[str, Tensor]:
        """Get feature tensors for specific diner indices.

        Args:
            indices: Tensor of diner indices.

        Returns:
            Dictionary containing feature tensors for the specified indices.
        """
        return {
            "large_category_ids": self.large_category_ids[indices],
            "middle_category_ids": self.middle_category_ids[indices],
            "small_category_ids": self.small_category_ids[indices],
            "menu_embeddings": self.menu_embeddings[indices],
            "diner_name_embeddings": self.diner_name_embeddings[indices],
            "price_features": self.price_features[indices],
            "review_text_embeddings": self.review_text_embeddings[indices],
        }


def multimodal_triplet_collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Collate function for MultimodalTripletDataset.

    Combines individual triplet samples into a batch.

    Args:
        batch: List of samples from MultimodalTripletDataset.

    Returns:
        Dictionary containing batched tensors.
    """
    anchor_indices = torch.stack([sample["anchor_idx"] for sample in batch])
    positive_indices = torch.stack([sample["positive_idx"] for sample in batch])
    negative_indices = torch.stack([sample["negative_indices"] for sample in batch])
    anchor_categories = torch.stack([sample["anchor_category"] for sample in batch])
    positive_categories = torch.stack([sample["positive_category"] for sample in batch])
    negative_categories = torch.stack(
        [sample["negative_categories"] for sample in batch]
    )

    return {
        "anchor_indices": anchor_indices,
        "positive_indices": positive_indices,
        "negative_indices": negative_indices,
        "anchor_categories": anchor_categories,
        "positive_categories": positive_categories,
        "negative_categories": negative_categories,
    }


def create_multimodal_triplet_dataloader(
    features_path: str,
    pairs_path: str,
    category_mapping_path: str,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    num_hard_negatives: int = 5,
    num_nearby_negatives: int = 3,
    num_random_negatives: int = 2,
    random_seed: int = 42,
) -> Tuple[DataLoader, MultimodalTripletDataset]:
    """Create DataLoader for multimodal triplet embedding training.

    Args:
        features_path: Path to preprocessed features parquet file.
        pairs_path: Path to training pairs parquet file.
        category_mapping_path: Path to category mapping parquet file.
        batch_size: Batch size for training. Default: 256.
        shuffle: Whether to shuffle data. Default: True.
        num_workers: Number of data loading workers. Default: 4.
        num_hard_negatives: Number of same-category hard negatives. Default: 5.
        num_nearby_negatives: Number of nearby-category negatives. Default: 3.
        num_random_negatives: Number of random negatives. Default: 2.
        random_seed: Random seed for reproducibility. Default: 42.

    Returns:
        Tuple of (DataLoader, MultimodalTripletDataset).
    """
    dataset = MultimodalTripletDataset(
        features_path=features_path,
        pairs_path=pairs_path,
        category_mapping_path=category_mapping_path,
        num_hard_negatives=num_hard_negatives,
        num_nearby_negatives=num_nearby_negatives,
        num_random_negatives=num_random_negatives,
        random_seed=random_seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=multimodal_triplet_collate_fn,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return dataloader, dataset
