"""Multimodal Triplet Embedding Model for restaurant recommendation.

This module implements a multi-modal embedding model for diners (restaurants)
that creates 128-dimensional embeddings where dot-product similarity returns
semantically similar restaurants.

The model uses 4 encoders:
- CategoryEncoder: Encodes hierarchical category features (128d)
- MenuEncoder: Encodes menu text using frozen KoBERT (256d)
- DinerNameEncoder: Encodes diner name using frozen KoBERT (64d)
- PriceEncoder: Encodes price statistics (32d)

These are fused using multi-head attention and projected to the final 128d embedding.
Training uses triplet loss with hard negative mining.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor

from yamyam_lab.constant.metric.metric import Metric
from yamyam_lab.model.embedding.encoders import (
    AttentionFusion,
    CategoryEncoder,
    DinerNameEncoder,
    FinalProjection,
    MenuEncoder,
    PriceEncoder,
    ReviewTextEncoder,
)


@dataclass
class MultimodalTripletConfig:
    """Configuration for MultimodalTripletModel.

    Attributes:
        num_large_categories: Number of unique large categories.
        num_middle_categories: Number of unique middle categories.
        embedding_dim: Final embedding dimension. Default: 128.
        category_dim: Category encoder output dimension. Default: 128.
        menu_dim: Menu encoder output dimension. Default: 256.
        diner_name_dim: Diner name encoder output dimension. Default: 64.
        price_dim: Price encoder output dimension. Default: 32.
        num_attention_heads: Number of attention heads in fusion layer. Default: 4.
        dropout: Dropout probability. Default: 0.1.
        kobert_model_name: HuggingFace model name for KoBERT. Default: "klue/bert-base".
        use_precomputed_menu_embeddings: Whether to use precomputed KoBERT embeddings.
        use_precomputed_name_embeddings: Whether to use precomputed KoBERT embeddings for names.
        device: Device to run model on. Default: "cpu".
        top_k_values: List of k values for evaluation metrics.
        diner_ids: Tensor of all diner IDs.
        recommend_batch_size: Batch size for recommendation. Default: 1000.
    """

    num_large_categories: int
    num_middle_categories: int
    embedding_dim: int = 128
    category_dim: int = 128
    menu_dim: int = 256
    diner_name_dim: int = 64
    price_dim: int = 32
    review_text_dim: int = 128
    num_attention_heads: int = 4
    dropout: float = 0.1
    kobert_model_name: str = "klue/bert-base"
    use_precomputed_menu_embeddings: bool = True
    use_precomputed_name_embeddings: bool = True
    use_precomputed_review_text_embeddings: bool = True
    device: str = "cpu"
    top_k_values: List[int] = None
    diner_ids: Tensor = None
    recommend_batch_size: int = 1000


class Model(nn.Module):
    """Multimodal Triplet Embedding Model for restaurant recommendation.

    Creates 128-dimensional embeddings for diners where dot-product similarity
    returns semantically similar restaurants. Trained using triplet loss with
    hard negative mining.

    The model architecture:
    1. CategoryEncoder: Hierarchical category embeddings -> 128d
    2. MenuEncoder: KoBERT-based menu text encoding -> 256d
    3. DinerNameEncoder: KoBERT-based diner name encoding -> 64d
    4. PriceEncoder: Price statistics encoding -> 32d
    5. AttentionFusion: Multi-head attention over 4 modalities -> 480d
    6. FinalProjection: MLP + L2 normalization -> 128d

    Args:
        config: MultimodalTripletConfig with model configuration.
    """

    def __init__(self, config: MultimodalTripletConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.device = config.device
        self.use_precomputed_menu_embeddings = config.use_precomputed_menu_embeddings
        self.use_precomputed_name_embeddings = config.use_precomputed_name_embeddings
        self.use_precomputed_review_text_embeddings = (
            config.use_precomputed_review_text_embeddings
        )
        self.review_text_dim = config.review_text_dim
        self.top_k_values = config.top_k_values or [3, 7, 10, 20, 100]
        self.diner_ids = config.diner_ids
        self.recommend_batch_size = config.recommend_batch_size

        # Initialize encoders
        self.category_encoder = CategoryEncoder(
            num_large_categories=config.num_large_categories,
            num_middle_categories=config.num_middle_categories,
            output_dim=config.category_dim,
            dropout=config.dropout,
        )

        self.menu_encoder = MenuEncoder(
            output_dim=config.menu_dim,
            dropout=config.dropout,
            kobert_model_name=config.kobert_model_name,
        )

        self.diner_name_encoder = DinerNameEncoder(
            output_dim=config.diner_name_dim,
            dropout=config.dropout,
            kobert_model_name=config.kobert_model_name,
        )

        self.price_encoder = PriceEncoder(
            output_dim=config.price_dim,
            dropout=config.dropout,
        )

        # Review text encoder (optional, enabled when review_text_dim > 0)
        if config.review_text_dim > 0:
            self.review_text_encoder = ReviewTextEncoder(
                output_dim=config.review_text_dim,
                dropout=config.dropout,
                kobert_model_name=config.kobert_model_name,
            )

        # Attention fusion layer
        self.attention_fusion = AttentionFusion(
            category_dim=config.category_dim,
            menu_dim=config.menu_dim,
            diner_name_dim=config.diner_name_dim,
            price_dim=config.price_dim,
            review_text_dim=config.review_text_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
        )

        # Final projection layer
        self.final_projection = FinalProjection(
            input_dim=self.attention_fusion.total_dim,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
        )

        # Store embeddings for all diners (computed during training)
        self._embedding = None

        # Training loss history
        self.tr_loss: List[float] = []

        # Metric tracking
        self.metric_at_k_total_epochs = {
            k: {
                Metric.MAP: [],
                Metric.NDCG: [],
                Metric.RECALL: [],
                Metric.COUNT: 0,
            }
            for k in self.top_k_values
        }

    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        """Forward pass to compute diner embeddings.

        Args:
            features: Dictionary containing:
                - large_category_ids: (batch_size,) large category indices
                - middle_category_ids: (batch_size,) middle category indices
                - menu_embeddings: (batch_size, 768) precomputed KoBERT embeddings
                  OR menu_input_ids + menu_attention_mask for raw text
                - diner_name_embeddings: (batch_size, 768) precomputed KoBERT embeddings
                  OR diner_name_input_ids + diner_name_attention_mask for raw text
                - price_features: (batch_size, 3) [avg_price, min_price, max_price]
                - review_text_embeddings: (batch_size, 768) precomputed KoBERT embeddings
                  (optional, used when review_text_dim > 0)

        Returns:
            Tensor of shape (batch_size, 128) with L2-normalized diner embeddings.
        """
        # Encode categories
        category_emb = self.category_encoder(
            large_category_ids=features["large_category_ids"],
            middle_category_ids=features["middle_category_ids"],
        )

        # Encode menu
        if self.use_precomputed_menu_embeddings:
            menu_emb = self.menu_encoder.forward_precomputed(
                menu_embeddings=features["menu_embeddings"]
            )
        else:
            menu_emb = self.menu_encoder(
                input_ids=features["menu_input_ids"],
                attention_mask=features["menu_attention_mask"],
            )

        # Encode diner name
        if self.use_precomputed_name_embeddings:
            diner_name_emb = self.diner_name_encoder.forward_precomputed(
                name_embeddings=features["diner_name_embeddings"]
            )
        else:
            diner_name_emb = self.diner_name_encoder(
                input_ids=features["diner_name_input_ids"],
                attention_mask=features["diner_name_attention_mask"],
            )

        # Encode price
        price_emb = self.price_encoder(features["price_features"])

        # Encode review text (optional)
        review_text_emb = None
        if self.review_text_dim > 0:
            if self.use_precomputed_review_text_embeddings:
                review_text_emb = self.review_text_encoder.forward_precomputed(
                    review_text_embeddings=features["review_text_embeddings"]
                )
            else:
                review_text_emb = self.review_text_encoder(
                    input_ids=features["review_text_input_ids"],
                    attention_mask=features["review_text_attention_mask"],
                )

        # Fuse modalities with attention
        fused = self.attention_fusion(
            category_emb=category_emb,
            menu_emb=menu_emb,
            diner_name_emb=diner_name_emb,
            price_emb=price_emb,
            review_text_emb=review_text_emb,
        )

        # Final projection with L2 normalization
        embeddings = self.final_projection(fused)

        return embeddings

    def compute_and_store_embeddings(
        self, all_features: Dict[str, Tensor], batch_size: int = 256
    ) -> None:
        """Compute and store embeddings for all diners.

        Args:
            all_features: Dictionary containing features for all diners.
            batch_size: Batch size for computing embeddings.
        """
        num_diners = all_features["large_category_ids"].size(0)
        self._embedding = torch.zeros(
            num_diners, self.embedding_dim, device=self.device
        )

        self.eval()
        with torch.no_grad():
            for start in range(0, num_diners, batch_size):
                end = min(start + batch_size, num_diners)

                # Extract batch features
                batch_features = {
                    key: value[start:end].to(self.device)
                    for key, value in all_features.items()
                }

                # Compute embeddings
                embeddings = self.forward(batch_features)
                self._embedding[start:end] = embeddings

        self.train()

    def get_embedding(self, diner_indices: Tensor) -> Tensor:
        """Get embeddings for specific diner indices.

        Args:
            diner_indices: Tensor of diner indices.

        Returns:
            Tensor of shape (len(diner_indices), embedding_dim) with diner embeddings.
        """
        if self._embedding is None:
            raise RuntimeError(
                "Embeddings not computed. Call compute_and_store_embeddings first."
            )
        return self._embedding[diner_indices]

    def similarity(self, anchor: Tensor, candidates: Tensor) -> Tensor:
        """Compute dot product similarity between anchor and candidates.

        Since embeddings are L2-normalized, dot product equals cosine similarity.

        Args:
            anchor: Tensor of shape (batch_size, embedding_dim).
            candidates: Tensor of shape (num_candidates, embedding_dim).

        Returns:
            Tensor of shape (batch_size, num_candidates) with similarity scores.
        """
        return torch.mm(anchor, candidates.t())

    def recommend(
        self,
        anchor_embedding: Tensor,
        exclude_indices: Optional[List[int]] = None,
        top_k: int = 10,
    ) -> Tuple[NDArray, NDArray]:
        """Get top-k recommendations for a diner embedding.

        Args:
            anchor_embedding: Tensor of shape (1, embedding_dim) for the anchor diner.
            exclude_indices: List of diner indices to exclude from recommendations.
            top_k: Number of recommendations to return.

        Returns:
            Tuple of (diner_indices, similarity_scores) arrays.
        """
        if self._embedding is None:
            raise RuntimeError(
                "Embeddings not computed. Call compute_and_store_embeddings first."
            )

        # Compute similarity with all diners
        similarities = self.similarity(anchor_embedding, self._embedding).squeeze(0)

        # Exclude specified indices
        if exclude_indices is not None:
            for idx in exclude_indices:
                similarities[idx] = -float("inf")

        # Get top-k
        top_k_result = torch.topk(similarities, k=top_k)
        indices = top_k_result.indices.detach().cpu().numpy()
        scores = top_k_result.values.detach().cpu().numpy()

        return indices, scores

    def generate_candidates_for_each_diner(self, top_k_value: int) -> pd.DataFrame:
        """Generate top-k similar diners for each diner.

        Args:
            top_k_value: Number of candidates to generate per diner.

        Returns:
            DataFrame with columns [diner_id, candidate_diner_id, score].
        """
        if self._embedding is None:
            raise RuntimeError(
                "Embeddings not computed. Call compute_and_store_embeddings first."
            )

        num_diners = self._embedding.size(0)
        results = []

        for start in range(0, num_diners, self.recommend_batch_size):
            end = min(start + self.recommend_batch_size, num_diners)
            batch_embeddings = self._embedding[start:end]

            # Compute similarities
            similarities = self.similarity(batch_embeddings, self._embedding)

            # Exclude self-similarity
            for i in range(end - start):
                similarities[i, start + i] = -float("inf")

            # Get top-k for each diner in batch
            top_k = torch.topk(similarities, k=top_k_value, dim=1)
            top_k_indices = top_k.indices.cpu().numpy()
            top_k_scores = top_k.values.cpu().numpy()

            # Collect results
            for i in range(end - start):
                diner_idx = start + i
                for j in range(top_k_value):
                    results.append(
                        {
                            "diner_id": diner_idx,
                            "candidate_diner_id": top_k_indices[i, j],
                            "score": top_k_scores[i, j],
                        }
                    )

        return pd.DataFrame(results)
