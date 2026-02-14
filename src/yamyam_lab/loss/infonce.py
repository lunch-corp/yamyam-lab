"""InfoNCE (Noise Contrastive Estimation) loss for embedding learning.

This module provides InfoNCE loss functions for learning embeddings where
similar items should have high dot-product similarity. InfoNCE uses a
softmax-based formulation that provides better gradient signal than
triplet loss and naturally prevents embedding collapse.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def infonce_loss_with_multiple_negatives(
    anchor: Tensor,
    positive: Tensor,
    negatives: Tensor,
    anchor_category: Tensor,
    positive_category: Tensor,
    negative_categories: Tensor,
    temperature: float = 0.07,
    category_weight: float = 0.1,
) -> Tensor:
    """Compute InfoNCE loss with multiple negatives per anchor.

    InfoNCE treats the problem as (1+N)-way classification: the model must
    identify the positive among N negatives. The softmax denominator provides
    a natural repulsion force that prevents embedding collapse.

    Loss = -log(exp(sim(a,p)/t) / (exp(sim(a,p)/t) + sum(exp(sim(a,n_i)/t))))

    Args:
        anchor: Tensor of shape (batch_size, embedding_dim) with L2-normalized anchor embeddings.
        positive: Tensor of shape (batch_size, embedding_dim) with L2-normalized positive embeddings.
        negatives: Tensor of shape (batch_size, num_negatives, embedding_dim) with L2-normalized negative embeddings.
        anchor_category: Tensor of shape (batch_size,) with anchor category IDs.
        positive_category: Tensor of shape (batch_size,) with positive category IDs.
        negative_categories: Tensor of shape (batch_size, num_negatives) with negative category IDs.
        temperature: Temperature scaling parameter. Lower values make the distribution sharper. Default: 0.07.
        category_weight: Weight for category regularization. Default: 0.1.

    Returns:
        Scalar tensor with combined loss.
    """
    # anchor: (B, D), positive: (B, D), negatives: (B, N, D)

    # Compute positive similarity: (B,)
    pos_similarity = torch.sum(anchor * positive, dim=-1)

    # Compute negative similarities: (B, N)
    neg_similarities = torch.sum(anchor.unsqueeze(1) * negatives, dim=-1)

    # Scale by temperature
    pos_logits = pos_similarity / temperature  # (B,)
    neg_logits = neg_similarities / temperature  # (B, N)

    # Concatenate: positive at index 0, negatives at indices 1..N
    # logits: (B, 1+N)
    logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)

    # Labels: positive is always at index 0
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

    # Cross-entropy loss
    base_loss = F.cross_entropy(logits, labels)

    # Category-aware regularization (same as triplet loss variant)
    same_category_mask = (anchor_category == positive_category).float()
    category_loss = same_category_mask * (1 - pos_similarity)
    category_loss = category_loss.mean()

    return base_loss + category_weight * category_loss
