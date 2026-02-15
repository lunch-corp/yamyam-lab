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


def build_positive_mask(
    anchor_diner_indices: Tensor,
    positive_diner_indices: Tensor,
    diner_to_positives: dict,
) -> Tensor:
    """Build a boolean mask of known positive pairs for in-batch InfoNCE.

    mask[i][j] is True when positive_diner_indices[j] is a known positive
    of anchor_diner_indices[i] AND i != j (diagonal is never masked so the
    paired positive stays in the denominator).

    Args:
        anchor_diner_indices: (B,) tensor of anchor diner indices.
        positive_diner_indices: (B,) tensor of positive diner indices.
        diner_to_positives: Mapping from diner_idx to set of positive diner indices.

    Returns:
        Boolean tensor of shape (B, B).
    """
    B = anchor_diner_indices.size(0)
    mask = torch.zeros(B, B, dtype=torch.bool, device=anchor_diner_indices.device)

    anchor_list = anchor_diner_indices.tolist()
    positive_list = positive_diner_indices.tolist()

    for i, a_idx in enumerate(anchor_list):
        pos_set = diner_to_positives.get(a_idx, set())
        for j, p_idx in enumerate(positive_list):
            if i != j and p_idx in pos_set:
                mask[i, j] = True

    return mask


def infonce_in_batch_loss(
    anchor_emb: Tensor,
    positive_emb: Tensor,
    positive_mask: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """CLIP-style in-batch InfoNCE loss.

    Computes B x B similarity matrix between anchors and positives.
    Known positive pairs (other than the diagonal) are masked out
    so they don't act as false negatives.

    Args:
        anchor_emb: (B, D) L2-normalized anchor embeddings.
        positive_emb: (B, D) L2-normalized positive embeddings.
        positive_mask: (B, B) boolean mask — True entries are set to -inf.
        temperature: Temperature scaling parameter.

    Returns:
        Scalar loss tensor.
    """
    # (B, B) similarity matrix
    sim_matrix = anchor_emb @ positive_emb.T / temperature

    # Mask known positives (set to -inf so they don't contribute to denominator)
    sim_matrix = sim_matrix.masked_fill(positive_mask, float("-inf"))

    # Labels: diagonal (i-th anchor matches i-th positive)
    labels = torch.arange(anchor_emb.size(0), device=anchor_emb.device)

    return F.cross_entropy(sim_matrix, labels)
