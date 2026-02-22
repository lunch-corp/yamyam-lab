"""Encoder modules for multimodal triplet embedding model.

This module contains the encoder components used to create diner embeddings:
- CategoryEncoder: Encodes hierarchical category features (large, middle)
- MenuEncoder: Encodes menu text using frozen KoBERT
- DinerNameEncoder: Encodes diner name using frozen KoBERT
- PriceEncoder: Encodes price statistics
- ReviewTextEncoder: Encodes aggregated review text using frozen KoBERT
- AttentionFusion: Multi-head attention fusion of modalities
- FinalProjection: Projects fused features to final embedding dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import BertModel


class CategoryEncoder(nn.Module):
    """Encodes hierarchical category features (large, middle) into a single embedding.

    The encoder uses separate embedding tables for each category level:
    - Large category: 32-dimensional
    - Middle category: 48-dimensional

    These are concatenated and passed through an MLP to produce 128-dimensional output.

    Args:
        num_large_categories: Number of unique large categories.
        num_middle_categories: Number of unique middle categories.
        output_dim: Output embedding dimension. Default: 128.
        dropout: Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        num_large_categories: int,
        num_middle_categories: int,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.large_dim = 32
        self.middle_dim = 48
        self.concat_dim = self.large_dim + self.middle_dim  # 80

        # Embedding tables for each category level
        self.large_embedding = nn.Embedding(num_large_categories, self.large_dim)
        self.middle_embedding = nn.Embedding(num_middle_categories, self.middle_dim)

        # MLP to project concatenated embeddings to output dimension
        self.mlp = nn.Sequential(
            nn.Linear(self.concat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embedding weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.large_embedding.weight)
        nn.init.xavier_uniform_(self.middle_embedding.weight)

    def forward(
        self,
        large_category_ids: Tensor,
        middle_category_ids: Tensor,
    ) -> Tensor:
        """Forward pass through category encoder.

        Args:
            large_category_ids: Tensor of shape (batch_size,) with large category indices.
            middle_category_ids: Tensor of shape (batch_size,) with middle category indices.

        Returns:
            Tensor of shape (batch_size, output_dim) with encoded category features.
        """
        large_emb = self.large_embedding(large_category_ids)  # (B, 32)
        middle_emb = self.middle_embedding(middle_category_ids)  # (B, 48)

        concat_emb = torch.cat([large_emb, middle_emb], dim=-1)  # (B, 80)
        return self.mlp(concat_emb)


class MenuEncoder(nn.Module):
    """Encodes menu text using frozen KoBERT with mean pooling.

    Uses the pretrained KoBERT model from HuggingFace (monologg/kobert)
    with frozen weights. The output is passed through an MLP to produce
    256-dimensional output.

    Args:
        output_dim: Output embedding dimension. Default: 256.
        dropout: Dropout probability. Default: 0.1.
        kobert_model_name: HuggingFace model name. Default: "monologg/kobert".
    """

    def __init__(
        self,
        output_dim: int = 256,
        dropout: float = 0.1,
        kobert_model_name: str = "monologg/kobert",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.kobert_model_name = kobert_model_name

        # Lazy initialization - KoBERT will be loaded on first forward pass
        self._kobert = None
        self._kobert_dim = 768  # KoBERT hidden size

        # MLP to project KoBERT output to desired dimension
        self.mlp = nn.Sequential(
            nn.Linear(self._kobert_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )

    def _load_kobert(self, device: torch.device) -> None:
        """Lazy load KoBERT model."""
        if self._kobert is None:
            self._kobert = BertModel.from_pretrained(self.kobert_model_name)
            self._kobert = self._kobert.to(device)
            # Freeze KoBERT weights
            for param in self._kobert.parameters():
                param.requires_grad = False
            self._kobert.eval()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Forward pass through menu encoder.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with tokenized menu text.
            attention_mask: Tensor of shape (batch_size, seq_len) with attention mask.

        Returns:
            Tensor of shape (batch_size, 256) with encoded menu features.
        """
        device = input_ids.device
        self._load_kobert(device)

        # Get KoBERT outputs (frozen)
        with torch.no_grad():
            outputs = self._kobert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Mean pooling over sequence dimension
            hidden_states = outputs.last_hidden_state  # (B, seq_len, 768)
            # Mask padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask  # (B, 768)

        # Project through MLP (trainable)
        return self.mlp(pooled)  # (B, 256)

    def forward_precomputed(self, menu_embeddings: Tensor) -> Tensor:
        """Forward pass with precomputed KoBERT embeddings.

        Use this method when KoBERT embeddings are precomputed during data preprocessing.

        Args:
            menu_embeddings: Tensor of shape (batch_size, 768) with precomputed KoBERT embeddings.

        Returns:
            Tensor of shape (batch_size, 256) with encoded menu features.
        """
        return self.mlp(menu_embeddings)


class DinerNameEncoder(nn.Module):
    """Encodes diner name using frozen KoBERT with mean pooling.

    Uses the pretrained Korean BERT model from HuggingFace
    with frozen weights. The output is passed through an MLP to produce
    64-dimensional output.

    Args:
        output_dim: Output embedding dimension. Default: 64.
        dropout: Dropout probability. Default: 0.1.
        kobert_model_name: HuggingFace model name. Default: "klue/bert-base".
    """

    def __init__(
        self,
        output_dim: int = 64,
        dropout: float = 0.1,
        kobert_model_name: str = "klue/bert-base",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.kobert_model_name = kobert_model_name

        # Lazy initialization - KoBERT will be loaded on first forward pass
        self._kobert = None
        self._kobert_dim = 768  # KoBERT hidden size

        # MLP to project KoBERT output to desired dimension
        self.mlp = nn.Sequential(
            nn.Linear(self._kobert_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.ReLU(),
        )

    def _load_kobert(self, device: torch.device) -> None:
        """Lazy load KoBERT model."""
        if self._kobert is None:
            self._kobert = BertModel.from_pretrained(self.kobert_model_name)
            self._kobert = self._kobert.to(device)
            # Freeze KoBERT weights
            for param in self._kobert.parameters():
                param.requires_grad = False
            self._kobert.eval()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Forward pass through diner name encoder.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with tokenized diner name.
            attention_mask: Tensor of shape (batch_size, seq_len) with attention mask.

        Returns:
            Tensor of shape (batch_size, 64) with encoded diner name features.
        """
        device = input_ids.device
        self._load_kobert(device)

        # Get KoBERT outputs (frozen)
        with torch.no_grad():
            outputs = self._kobert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Mean pooling over sequence dimension
            hidden_states = outputs.last_hidden_state  # (B, seq_len, 768)
            # Mask padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask  # (B, 768)

        # Project through MLP (trainable)
        return self.mlp(pooled)  # (B, 64)

    def forward_precomputed(self, name_embeddings: Tensor) -> Tensor:
        """Forward pass with precomputed KoBERT embeddings.

        Use this method when KoBERT embeddings are precomputed during data preprocessing.

        Args:
            name_embeddings: Tensor of shape (batch_size, 768) with precomputed KoBERT embeddings.

        Returns:
            Tensor of shape (batch_size, 64) with encoded diner name features.
        """
        return self.mlp(name_embeddings)


class PriceEncoder(nn.Module):
    """Encodes price statistics into embeddings.

    Takes 3 price features:
    - Average menu price
    - Min menu price
    - Max menu price

    Args:
        output_dim: Output embedding dimension. Default: 32.
        dropout: Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        output_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = 3  # avg_price, min_price, max_price

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, output_dim),
            nn.ReLU(),
        )

    def forward(self, price_features: Tensor) -> Tensor:
        """Forward pass through price encoder.

        Args:
            price_features: Tensor of shape (batch_size, 3) containing
                [avg_price, min_price, max_price] for each diner.

        Returns:
            Tensor of shape (batch_size, 32) with encoded price features.
        """
        return self.mlp(price_features)


class ReviewTextEncoder(nn.Module):
    """Encodes aggregated review text using frozen KoBERT with mean pooling.

    Uses the pretrained Korean BERT model from HuggingFace
    with frozen weights. The output is passed through an MLP to produce
    128-dimensional output.

    Args:
        output_dim: Output embedding dimension. Default: 128.
        dropout: Dropout probability. Default: 0.1.
        kobert_model_name: HuggingFace model name. Default: "klue/bert-base".
    """

    def __init__(
        self,
        output_dim: int = 128,
        dropout: float = 0.1,
        kobert_model_name: str = "klue/bert-base",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.kobert_model_name = kobert_model_name

        # Lazy initialization - KoBERT will be loaded on first forward pass
        self._kobert = None
        self._kobert_dim = 768  # KoBERT hidden size

        # MLP to project KoBERT output to desired dimension
        self.mlp = nn.Sequential(
            nn.Linear(self._kobert_dim, 384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(384, output_dim),
            nn.ReLU(),
        )

    def _load_kobert(self, device: torch.device) -> None:
        """Lazy load KoBERT model."""
        if self._kobert is None:
            self._kobert = BertModel.from_pretrained(self.kobert_model_name)
            self._kobert = self._kobert.to(device)
            # Freeze KoBERT weights
            for param in self._kobert.parameters():
                param.requires_grad = False
            self._kobert.eval()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Forward pass through review text encoder.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with tokenized review text.
            attention_mask: Tensor of shape (batch_size, seq_len) with attention mask.

        Returns:
            Tensor of shape (batch_size, output_dim) with encoded review text features.
        """
        device = input_ids.device
        self._load_kobert(device)

        # Get KoBERT outputs (frozen)
        with torch.no_grad():
            outputs = self._kobert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            # Mean pooling over sequence dimension
            hidden_states = outputs.last_hidden_state  # (B, seq_len, 768)
            # Mask padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask  # (B, 768)

        # Project through MLP (trainable)
        return self.mlp(pooled)

    def forward_precomputed(self, review_text_embeddings: Tensor) -> Tensor:
        """Forward pass with precomputed KoBERT embeddings.

        Use this method when KoBERT embeddings are precomputed during data preprocessing.

        Args:
            review_text_embeddings: Tensor of shape (batch_size, 768) with precomputed KoBERT embeddings.

        Returns:
            Tensor of shape (batch_size, output_dim) with encoded review text features.
        """
        return self.mlp(review_text_embeddings)


class AttentionFusion(nn.Module):
    """Multi-head attention fusion layer for combining modality embeddings.

    Takes embeddings from multiple modalities and fuses them using
    multi-head self-attention. Supports 4 base modalities (category, menu,
    diner_name, price) plus an optional review_text modality.

    Args:
        category_dim: Dimension of category embeddings. Default: 128.
        menu_dim: Dimension of menu embeddings. Default: 256.
        diner_name_dim: Dimension of diner name embeddings. Default: 64.
        price_dim: Dimension of price embeddings. Default: 32.
        review_text_dim: Dimension of review text embeddings. Default: 0 (disabled).
        num_heads: Number of attention heads. Default: 4.
        dropout: Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        category_dim: int = 128,
        menu_dim: int = 256,
        diner_name_dim: int = 64,
        price_dim: int = 32,
        review_text_dim: int = 0,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.category_dim = category_dim
        self.menu_dim = menu_dim
        self.diner_name_dim = diner_name_dim
        self.price_dim = price_dim
        self.review_text_dim = review_text_dim
        self.total_dim = (
            category_dim + menu_dim + diner_name_dim + price_dim + review_text_dim
        )
        self.num_modalities = 4 + (1 if review_text_dim > 0 else 0)
        self.num_heads = num_heads

        # Project all modalities to same dimension for attention
        self.attention_dim = 128  # Common dimension for attention
        self.category_proj = nn.Linear(category_dim, self.attention_dim)
        self.menu_proj = nn.Linear(menu_dim, self.attention_dim)
        self.diner_name_proj = nn.Linear(diner_name_dim, self.attention_dim)
        self.price_proj = nn.Linear(price_dim, self.attention_dim)

        if review_text_dim > 0:
            self.review_text_proj = nn.Linear(review_text_dim, self.attention_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # LayerNorm for attention output
        self.layer_norm = nn.LayerNorm(self.attention_dim)

        # Output projection to concatenate attention-weighted features
        self.output_proj = nn.Linear(
            self.attention_dim * self.num_modalities, self.total_dim
        )

    def forward(
        self,
        category_emb: Tensor,
        menu_emb: Tensor,
        diner_name_emb: Tensor,
        price_emb: Tensor,
        review_text_emb: Tensor = None,
    ) -> Tensor:
        """Forward pass through attention fusion layer.

        Args:
            category_emb: Tensor of shape (batch_size, category_dim).
            menu_emb: Tensor of shape (batch_size, menu_dim).
            diner_name_emb: Tensor of shape (batch_size, diner_name_dim).
            price_emb: Tensor of shape (batch_size, price_dim).
            review_text_emb: Optional tensor of shape (batch_size, review_text_dim).

        Returns:
            Tensor of shape (batch_size, total_dim) with fused features.
        """
        batch_size = category_emb.size(0)

        # Project each modality to attention dimension
        projections = [
            self.category_proj(category_emb),  # (B, 128)
            self.menu_proj(menu_emb),  # (B, 128)
            self.diner_name_proj(diner_name_emb),  # (B, 128)
            self.price_proj(price_emb),  # (B, 128)
        ]

        if self.review_text_dim > 0 and review_text_emb is not None:
            projections.append(self.review_text_proj(review_text_emb))  # (B, 128)

        # Stack as sequence for attention: (B, num_modalities, 128)
        modalities = torch.stack(projections, dim=1)

        # Apply self-attention
        attn_output, _ = self.attention(modalities, modalities, modalities)
        attn_output = self.layer_norm(attn_output + modalities)  # Residual connection

        # Flatten attention output: (B, num_modalities * 128)
        attn_flat = attn_output.view(batch_size, -1)

        # Project to output dimension: (B, total_dim)
        output = self.output_proj(attn_flat)

        return output


class FinalProjection(nn.Module):
    """Final projection layer that maps fused features to L2-normalized embeddings.

    Args:
        input_dim: Input dimension from attention fusion. Default: 480.
        output_dim: Output embedding dimension. Default: 128.
        dropout: Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        input_dim: int = 480,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, fused_features: Tensor) -> Tensor:
        """Forward pass through final projection layer.

        Args:
            fused_features: Tensor of shape (batch_size, 480) from attention fusion.

        Returns:
            Tensor of shape (batch_size, 128) with L2-normalized embeddings.
        """
        embeddings = self.mlp(fused_features)  # (B, 128)
        # L2 normalize for dot product similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
