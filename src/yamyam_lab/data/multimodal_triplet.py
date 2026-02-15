"""Dataset and DataLoader for multimodal triplet embedding model.

This module provides the dataset class and data loading utilities for
training the multimodal triplet embedding model with in-batch InfoNCE loss.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class MultimodalTripletDataset(Dataset):
    """Dataset for multimodal triplet embedding training with in-batch InfoNCE.

    Returns anchor/positive pairs. Negatives are handled in-batch by the
    trainer using the B x B similarity matrix.

    Args:
        features_path: Path to preprocessed features parquet file.
        pairs_path: Path to training pairs parquet file.
        category_mapping_path: Path to category mapping parquet file.
        random_seed: Random seed for reproducibility. Default: 42.
    """

    def __init__(
        self,
        features_path: str,
        pairs_path: str,
        category_mapping_path: str,
        random_seed: int = 42,
    ):
        self.features_path = Path(features_path)
        self.pairs_path = Path(pairs_path)
        self.category_mapping_path = Path(category_mapping_path)

        np.random.seed(random_seed)

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

        # Load category mapping (kept for compatibility with evaluation)
        self.category_df = pd.read_parquet(self.category_mapping_path)

        # Build positive pair index for masking
        self._build_positive_pairs_index()

        # Build feature tensors
        self._build_feature_tensors()

    def _build_positive_pairs_index(self) -> None:
        """Build mapping from diner_idx to its known positive diner indices."""
        self.diner_to_positives: Dict[int, Set[int]] = {}
        for _, row in self.pairs_df.iterrows():
            a, p = int(row["anchor_idx"]), int(row["positive_idx"])
            self.diner_to_positives.setdefault(a, set()).add(p)
            self.diner_to_positives.setdefault(p, set()).add(a)

    def _build_feature_tensors(self) -> None:
        """Build feature tensors from dataframe."""
        # Category features
        self.large_category_ids = torch.tensor(
            self.features_df["large_category_id"].values, dtype=torch.long
        )
        self.middle_category_ids = torch.tensor(
            self.features_df["middle_category_id"].values, dtype=torch.long
        )
        # Menu embeddings (precomputed KoBERT)
        menu_cols = [col for col in self.features_df.columns if col.startswith("menu_")]
        if menu_cols:
            self.menu_embeddings = torch.tensor(
                self.features_df[menu_cols].values, dtype=torch.float32
            )
        else:
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
            self.review_text_embeddings = torch.zeros(
                self.num_diners, 768, dtype=torch.float32
            )

    def __len__(self) -> int:
        """Return number of positive pairs."""
        return len(self.pairs_df)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Get a training pair (anchor, positive).

        Args:
            idx: Index of the positive pair.

        Returns:
            Dictionary containing:
                - anchor_idx: Anchor position index (in feature tensors)
                - positive_idx: Positive position index (in feature tensors)
                - anchor_diner_idx: Anchor diner ID (for positive mask building)
                - positive_diner_idx: Positive diner ID (for positive mask building)
        """
        row = self.pairs_df.iloc[idx]
        anchor_diner_idx = int(row["anchor_idx"])
        positive_diner_idx = int(row["positive_idx"])

        anchor_pos = self.diner_idx_to_position.get(anchor_diner_idx, 0)
        positive_pos = self.diner_idx_to_position.get(positive_diner_idx, 0)

        return {
            "anchor_idx": torch.tensor(anchor_pos, dtype=torch.long),
            "positive_idx": torch.tensor(positive_pos, dtype=torch.long),
            "anchor_diner_idx": torch.tensor(anchor_diner_idx, dtype=torch.long),
            "positive_diner_idx": torch.tensor(positive_diner_idx, dtype=torch.long),
        }

    def get_all_features(self) -> Dict[str, Tensor]:
        """Get all feature tensors for embedding computation."""
        return {
            "large_category_ids": self.large_category_ids,
            "middle_category_ids": self.middle_category_ids,
            "menu_embeddings": self.menu_embeddings,
            "diner_name_embeddings": self.diner_name_embeddings,
            "price_features": self.price_features,
            "review_text_embeddings": self.review_text_embeddings,
        }

    def get_features_by_indices(self, indices: Tensor) -> Dict[str, Tensor]:
        """Get feature tensors for specific diner indices."""
        return {
            "large_category_ids": self.large_category_ids[indices],
            "middle_category_ids": self.middle_category_ids[indices],
            "menu_embeddings": self.menu_embeddings[indices],
            "diner_name_embeddings": self.diner_name_embeddings[indices],
            "price_features": self.price_features[indices],
            "review_text_embeddings": self.review_text_embeddings[indices],
        }


def multimodal_triplet_collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Collate function for MultimodalTripletDataset.

    Args:
        batch: List of samples from MultimodalTripletDataset.

    Returns:
        Dictionary containing batched tensors.
    """
    return {
        "anchor_indices": torch.stack([s["anchor_idx"] for s in batch]),
        "positive_indices": torch.stack([s["positive_idx"] for s in batch]),
        "anchor_diner_indices": torch.stack([s["anchor_diner_idx"] for s in batch]),
        "positive_diner_indices": torch.stack([s["positive_diner_idx"] for s in batch]),
    }


def create_multimodal_triplet_dataloader(
    features_path: str,
    pairs_path: str,
    category_mapping_path: str,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
    random_seed: int = 42,
) -> Tuple[DataLoader, MultimodalTripletDataset]:
    """Create DataLoader for multimodal triplet embedding training.

    Args:
        features_path: Path to preprocessed features parquet file.
        pairs_path: Path to training pairs parquet file.
        category_mapping_path: Path to category mapping parquet file.
        batch_size: Batch size for training. Default: 512.
        shuffle: Whether to shuffle data. Default: True.
        num_workers: Number of data loading workers. Default: 4.
        random_seed: Random seed for reproducibility. Default: 42.

    Returns:
        Tuple of (DataLoader, MultimodalTripletDataset).
    """
    dataset = MultimodalTripletDataset(
        features_path=features_path,
        pairs_path=pairs_path,
        category_mapping_path=category_mapping_path,
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
