"""Triplet margin loss for embedding learning.

This module provides triplet loss functions for learning embeddings where
similar items should have high dot-product similarity.
"""

import torch
from torch import Tensor

EPS = 1e-15


def triplet_margin_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 0.5,
) -> Tensor:
    """Compute triplet margin loss using dot product similarity.

    The loss encourages the dot product between anchor and positive to be
    higher than anchor and negative by at least the margin.

    For L2-normalized embeddings, dot product equals cosine similarity,
    and similarity scores range from -1 to 1.

    Loss = max(0, margin - (sim(anchor, positive) - sim(anchor, negative)))

    Args:
        anchor: Tensor of shape (batch_size, embedding_dim) with L2-normalized anchor embeddings.
        positive: Tensor of shape (batch_size, embedding_dim) with L2-normalized positive embeddings.
        negative: Tensor of shape (batch_size, embedding_dim) with L2-normalized negative embeddings.
        margin: Margin for triplet loss. Default: 0.5.

    Returns:
        Scalar tensor with mean triplet loss.

    Example:
        >>> anchor = F.normalize(torch.randn(32, 128), p=2, dim=-1)
        >>> positive = F.normalize(torch.randn(32, 128), p=2, dim=-1)
        >>> negative = F.normalize(torch.randn(32, 128), p=2, dim=-1)
        >>> loss = triplet_margin_loss(anchor, positive, negative, margin=0.5)
    """
    # Compute dot product similarities (equivalent to cosine for normalized vectors)
    pos_similarity = torch.sum(anchor * positive, dim=-1)  # (batch_size,)
    neg_similarity = torch.sum(anchor * negative, dim=-1)  # (batch_size,)

    # Triplet margin loss: max(0, margin - (pos_sim - neg_sim))
    loss = torch.clamp(margin - (pos_similarity - neg_similarity), min=0)

    return loss.mean()


def triplet_margin_loss_with_category(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    anchor_category: Tensor,
    positive_category: Tensor,
    negative_category: Tensor,
    margin: float = 0.5,
    category_weight: float = 0.1,
) -> Tensor:
    """Compute triplet margin loss with category-aware regularization.

    Adds a category regularization term that encourages same-category items
    to have similar embeddings. This helps with hard negative mining.

    Total Loss = triplet_loss + category_weight * category_loss

    Category loss encourages:
    - Same category pairs to have higher similarity
    - Different category pairs to have lower similarity

    Args:
        anchor: Tensor of shape (batch_size, embedding_dim) with L2-normalized anchor embeddings.
        positive: Tensor of shape (batch_size, embedding_dim) with L2-normalized positive embeddings.
        negative: Tensor of shape (batch_size, embedding_dim) with L2-normalized negative embeddings.
        anchor_category: Tensor of shape (batch_size,) with anchor category IDs.
        positive_category: Tensor of shape (batch_size,) with positive category IDs.
        negative_category: Tensor of shape (batch_size,) with negative category IDs.
        margin: Margin for triplet loss. Default: 0.5.
        category_weight: Weight for category regularization. Default: 0.1.

    Returns:
        Scalar tensor with combined loss.
    """
    # Base triplet loss
    base_loss = triplet_margin_loss(anchor, positive, negative, margin)

    # Category-aware regularization
    # Encourage anchor-positive similarity when same category
    pos_similarity = torch.sum(anchor * positive, dim=-1)
    same_category_mask = (anchor_category == positive_category).float()

    # Loss: for same category, maximize similarity (minimize 1 - sim)
    # For different category, no additional penalty (handled by triplet loss)
    category_loss = same_category_mask * (1 - pos_similarity)
    category_loss = category_loss.mean()

    return base_loss + category_weight * category_loss


def triplet_margin_loss_with_multiple_negatives(
    anchor: Tensor,
    positive: Tensor,
    negatives: Tensor,
    anchor_category: Tensor,
    positive_category: Tensor,
    negative_categories: Tensor,
    margin: float = 0.5,
    category_weight: float = 0.1,
) -> Tensor:
    """Compute triplet margin loss with multiple negatives per anchor.

    This is a batched version that processes all negatives in one pass,
    avoiding the need to loop over negatives individually.

    Args:
        anchor: Tensor of shape (batch_size, embedding_dim) with L2-normalized anchor embeddings.
        positive: Tensor of shape (batch_size, embedding_dim) with L2-normalized positive embeddings.
        negatives: Tensor of shape (batch_size, num_negatives, embedding_dim) with L2-normalized negative embeddings.
        anchor_category: Tensor of shape (batch_size,) with anchor category IDs.
        positive_category: Tensor of shape (batch_size,) with positive category IDs.
        negative_categories: Tensor of shape (batch_size, num_negatives) with negative category IDs.
        margin: Margin for triplet loss. Default: 0.5.
        category_weight: Weight for category regularization. Default: 0.1.

    Returns:
        Scalar tensor with combined loss averaged over all negatives.
    """
    # anchor: (B, D), positive: (B, D), negatives: (B, N, D)
    # Compute positive similarity: (B,)
    pos_similarity = torch.sum(anchor * positive, dim=-1)

    # Compute negative similarities: (B, N)
    # anchor.unsqueeze(1): (B, 1, D), negatives: (B, N, D)
    neg_similarities = torch.sum(anchor.unsqueeze(1) * negatives, dim=-1)

    # Triplet loss for each negative: (B, N)
    # margin - (pos_sim - neg_sim) for each negative
    triplet_losses = torch.clamp(
        margin - (pos_similarity.unsqueeze(1) - neg_similarities), min=0
    )

    # Base triplet loss: mean over all (batch, negatives)
    base_loss = triplet_losses.mean()

    # Category-aware regularization (same as single-negative version)
    same_category_mask = (anchor_category == positive_category).float()
    category_loss = same_category_mask * (1 - pos_similarity)
    category_loss = category_loss.mean()

    return base_loss + category_weight * category_loss


def batch_hard_triplet_loss(
    embeddings: Tensor,
    labels: Tensor,
    margin: float = 0.5,
) -> Tensor:
    """Compute batch hard triplet loss.

    For each anchor, selects the hardest positive (smallest similarity) and
    hardest negative (largest similarity) within the batch.

    Args:
        embeddings: Tensor of shape (batch_size, embedding_dim) with L2-normalized embeddings.
        labels: Tensor of shape (batch_size,) with category/group labels.
        margin: Margin for triplet loss. Default: 0.5.

    Returns:
        Scalar tensor with batch hard triplet loss.
    """
    batch_size = embeddings.size(0)
    device = embeddings.device

    # Compute pairwise dot product similarities
    similarities = torch.mm(embeddings, embeddings.t())  # (batch_size, batch_size)

    # Create masks for positive and negative pairs
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    labels_not_equal = ~labels_equal

    # Diagonal should not be considered (self-similarity)
    eye_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

    # Mask for valid positives: same label, not self
    positive_mask = labels_equal & ~eye_mask  # (B, B)

    # Mask for valid negatives: different label
    negative_mask = labels_not_equal  # (B, B)

    # For each anchor, find hardest positive (minimum similarity among positives)
    # Set non-positive similarities to large value so they're not selected as min
    pos_similarities = similarities.clone()
    pos_similarities[~positive_mask] = float("inf")
    hardest_pos_sim, _ = pos_similarities.min(dim=1)  # (B,)

    # For each anchor, find hardest negative (maximum similarity among negatives)
    # Set non-negative similarities to small value so they're not selected as max
    neg_similarities = similarities.clone()
    neg_similarities[~negative_mask] = -float("inf")
    hardest_neg_sim, _ = neg_similarities.max(dim=1)  # (B,)

    # Handle cases where no valid positive or negative exists
    # Check if anchor has at least one positive and one negative
    has_positive = positive_mask.sum(dim=1) > 0
    has_negative = negative_mask.sum(dim=1) > 0
    valid_anchor = has_positive & has_negative

    if not valid_anchor.any():
        # No valid triplets in batch
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Compute triplet loss only for valid anchors
    loss = torch.clamp(
        margin - (hardest_pos_sim[valid_anchor] - hardest_neg_sim[valid_anchor]),
        min=0,
    )

    return loss.mean()
