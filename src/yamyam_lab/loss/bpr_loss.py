import torch
from torch import Tensor

EPS = 1e-15


def bpr_loss(pred_pos: Tensor, pred_neg: Tensor) -> Tensor:
    """
    Calculate Bayesian Personalized Ranking (BPR) pairwise loss.

    BPR loss encourages the model to rank positive items higher than negative items.
    Loss = -log(sigmoid(score_pos - score_neg))

    Args:
        pred_pos (Tensor): Predicted scores for positive samples. Shape: (batch_size,)
        pred_neg (Tensor): Predicted scores for negative samples. Shape: (batch_size,)

    Returns (Tensor):
        Calculated BPR loss (scalar).
    """
    # Calculate pairwise difference
    diff = pred_pos - pred_neg

    # BPR loss: -log(sigmoid(diff))
    loss = -torch.log(torch.sigmoid(diff) + EPS).mean()

    return loss


def bpr_loss_sampled(
    scores: Tensor,
    targets: Tensor,
    user_ids: Tensor | None = None,
    sample_negatives: int = 20,  # Increased from 10 for better gradient signal
    debug: bool = False,
) -> Tensor:
    """
    Calculate BPR loss from a batch containing both positive and negative samples.

    This implementation uses a sampling strategy to handle imbalanced pos/neg ratios:
    - For each positive sample, randomly sample a subset of negatives for comparison
    - This ensures stable gradients even when positive:negative ratio is very skewed (e.g., 1:100)

    Args:
        scores (Tensor): Predicted scores for all samples (positive + negative). Shape: (batch_size,)
        targets (Tensor): Binary labels (1 for positive, 0 for negative). Shape: (batch_size,)
        user_ids (Tensor | None): User IDs for each sample. If provided, enforces user-wise comparison.
        sample_negatives (int): Number of negatives to sample per positive. Default: 10.
        debug (bool): If True, print debug information.

    Returns (Tensor):
        Calculated BPR loss (scalar).
    """
    # Separate positive and negative samples
    pos_mask = targets == 1
    neg_mask = targets == 0

    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]

    # Edge case: no pairs can be formed; return zero while keeping grad_fn
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return (scores * 0).sum()

    # If user_ids are provided, compute user-wise BPR loss with sampling
    if user_ids is not None:
        pos_user_ids = user_ids[pos_mask]
        neg_user_ids = user_ids[neg_mask]

        # Get unique users in the batch
        unique_users = torch.unique(pos_user_ids)

        losses = []
        users_with_loss = 0
        users_with_cross_neg = 0

        for user_id in unique_users:
            # Get positive and negative scores for this user
            user_pos_mask = pos_user_ids == user_id
            user_neg_mask = neg_user_ids == user_id

            user_pos_scores = pos_scores[user_pos_mask]
            user_neg_scores = neg_scores[user_neg_mask]

            # Skip if this user has no positive samples
            if user_pos_scores.numel() == 0:
                continue

            # If this user has no negative samples in the batch,
            # use ALL negatives from the batch (cross-user negatives)
            if user_neg_scores.numel() == 0:
                user_neg_scores = neg_scores
                users_with_cross_neg += 1

            # Sample negatives for efficiency
            num_negs = min(sample_negatives, user_neg_scores.numel())
            if num_negs < user_neg_scores.numel():
                # Randomly sample negatives
                neg_indices = torch.randperm(user_neg_scores.numel())[:num_negs]
                sampled_neg_scores = user_neg_scores[neg_indices]
            else:
                sampled_neg_scores = user_neg_scores

            # Compute pairwise differences for this user
            user_pos_scores = user_pos_scores.unsqueeze(1)  # (num_pos, 1)
            sampled_neg_scores = sampled_neg_scores.unsqueeze(0)  # (1, num_sampled_neg)

            diff = user_pos_scores - sampled_neg_scores  # (num_pos, num_sampled_neg)
            user_loss = -torch.log(torch.sigmoid(diff) + EPS).mean()
            losses.append(user_loss)
            users_with_loss += 1

        if len(losses) == 0:
            # Fallback: compute global loss if no user-specific loss could be computed
            pos_scores = pos_scores.unsqueeze(1)

            # Sample negatives globally
            num_negs = min(sample_negatives, neg_scores.numel())
            if num_negs < neg_scores.numel():
                neg_indices = torch.randperm(neg_scores.numel())[:num_negs]
                neg_scores = neg_scores[neg_indices]

            neg_scores = neg_scores.unsqueeze(0)
            diff = pos_scores - neg_scores
            return -torch.log(torch.sigmoid(diff) + EPS).mean()

        # Average across users
        return torch.stack(losses).mean()

    # If no user_ids, compute global BPR loss with sampling
    # Sample negatives for efficiency
    num_negs = min(sample_negatives, neg_scores.numel())
    if num_negs < neg_scores.numel():
        neg_indices = torch.randperm(neg_scores.numel())[:num_negs]
        sampled_neg_scores = neg_scores[neg_indices]
    else:
        sampled_neg_scores = neg_scores

    # Expand dimensions for broadcasting
    pos_scores = pos_scores.unsqueeze(1)  # (num_pos, 1)
    sampled_neg_scores = sampled_neg_scores.unsqueeze(0)  # (1, num_sampled_neg)

    # Compute pairwise differences (num_pos, num_sampled_neg)
    diff = pos_scores - sampled_neg_scores

    # BPR loss: -log(sigmoid(diff))
    loss = -torch.log(torch.sigmoid(diff) + EPS).mean()

    return loss
