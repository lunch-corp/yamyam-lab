from typing import Dict

import numpy as np
from numpy.typing import NDArray

from yamyam_lab.constant.metric.metric import Metric
from yamyam_lab.tools.utils import safe_divide


def ranked_precision(liked_item: int, reco_items: NDArray) -> float:
    K = len(reco_items)
    for i, item in enumerate(reco_items):
        if liked_item == item:
            return (K - i) / K
    return 0


def ranking_metrics_at_k(
    liked_items: NDArray,
    reco_items: NDArray,
) -> Dict[str, float]:
    r"""
    References
    - https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
    - https://gist.github.com/bwhite/3726239

    Calculates ndcg, average precision (aP), hit, and recall for `one user` **in binary setting**.
    If you want to derive ndcg, map for n users, you should average them over n.

    For average precision, we use following definition.

    AP = \dfrac{1}{m} \sum_{i=1}^{K} P(i) r(i)
    where m is number of items user liked and K is number of recommended items,
    P(i) is precision at i and r(i) is indicator variable 1 if ith item is hit else 0.
    Here, precision@i = # of hit items until ith ranked item / K

    Note that for some variations of AP, \dfrac{1}{ min(m, K) } is used instead of \dfrac{1}{m} to prevent dividing
    large value when K < m for some special case. If defined denominator as min(m, K), map may not increase as K is
    getting larger.

    For normalized discounted cumulative gain, we use following definition.

    NDCG = \dfrac{DCG}{IDCG}
    where DCG = \sum_{i=1}^{K} \dfrac{1}{\log2{i+1}} * r(i) and
    IDCG = \sum_{i=1}^{M} \dfrac{1}{\log2{i+1}}.
    Here, r(i) is indicator variable 1 if ith item is hit else 0,
    K is number of recommended items, and M is number of items that users liked.

    Maximum value of NDCG is IDCG by their definitions, therefore, 0 <= NDCG <= 1.

    Note that NDCG does NOT guarantee increasing value as K is getting larger.

    Args:
        liked_items (NDArray): Item ids selected by one user in test dataset.
        reco_items (NDArray): Item ids recommended for one user.

    Returns (Dict[str, float]):
        Calculated metric for one user which is recall, map and ndcg.
    """
    # number of recommended items
    K = len(reco_items)

    ap = 0
    cg = 1.0 / np.log2(np.arange(2, K + 2))
    idcg = cg[: len(liked_items)].sum()
    ndcg = 0
    hit = 0

    for i in range(K):
        # reco_item that is not in liked_items will not contribute to metric
        if reco_items[i] in liked_items:
            hit += 1
            ap += hit / (i + 1)
            ndcg += cg[i] / idcg
    ap = safe_divide(ap, len(liked_items))

    # Calculate recall
    recall = safe_divide(hit, len(liked_items))

    return {Metric.AP: ap, Metric.NDCG: ndcg, Metric.RECALL: recall}


def fully_vectorized_ranking_metrics_at_k(
    liked_items: NDArray,  # Shape: [n_users, n_liked_items]
    reco_items: NDArray,  # Shape: [n_users, K]
) -> Dict[str, NDArray]:
    """
    Fully vectorized implementation to calculate ranking metrics for multiple users.
    This implementation uses pure NumPy operations without any loops.

    Args:
        liked_items (NDArray): 2D array of shape [n_users, n_liked_items] where each row contains liked items for one user
        reco_items (NDArray): 2D array of shape [n_users, K] where each row contains recommended items for one user

    Returns (Dict[str, NDArray]):
        Dictionary of metrics (AP, NDCG, RECALL) where each value is an array
        with one entry per user
    """
    n_users, K = reco_items.shape
    n_liked_items = liked_items.shape[1]

    # Create a mask of shape [n_users, K, n_liked_items] to check if each recommendation is in liked items
    # First, reshape arrays to enable broadcasting
    reco_expanded = reco_items.reshape(n_users, K, 1)  # [n_users, K, 1]
    liked_expanded = liked_items.reshape(
        n_users, 1, n_liked_items
    )  # [n_users, 1, n_liked_items]

    # Check for matches (True where reco_item == liked_item)
    is_hit = reco_expanded == liked_expanded  # [n_users, K, n_liked_items]

    # For each position and user, determine if it's a hit (any match in liked_items)
    hit_at_k = np.any(is_hit, axis=2)  # [n_users, K]

    # Calculate cumulative hits at each position
    cumulative_hits = np.cumsum(hit_at_k, axis=1)  # [n_users, K]

    # Calculate position indices (1-based) for precision calculation
    positions = np.arange(1, K + 1).reshape(1, K)  # [1, K]

    # Calculate precision at each position: cumulative_hits / position
    precision_at_k = cumulative_hits / positions  # [n_users, K]

    # Precision values at hit positions contribute to AP
    ap_contributions = precision_at_k * hit_at_k  # [n_users, K]

    # Sum the contributions for each user
    ap_per_user_unnormalized = np.sum(ap_contributions, axis=1)  # [n_users]

    # Calculate discount factors for NDCG
    discount_factors = 1.0 / np.log2(np.arange(2, K + 2))  # [K]

    # Calculate discounted gain for each position (only if it's a hit)
    discounted_gain = hit_at_k * discount_factors.reshape(1, K)  # [n_users, K]

    # Sum discounted gain for each user
    dcg_per_user = np.sum(discounted_gain, axis=1)  # [n_users]

    # Calculate IDCG based on number of liked items
    # First, create array of positions for each user's IDCG calculation
    idcg_positions = np.minimum(n_liked_items, K)  # Use min(n_liked_items, K) positions

    # Calculate IDCG for each user
    idcg_per_user = np.zeros(n_users)
    for i in range(1, np.max(idcg_positions) + 1):
        idcg_per_user += (idcg_positions >= i) * (1.0 / np.log2(i + 1))

    # Count total hits per user
    total_hits_per_user = np.sum(hit_at_k, axis=1)  # [n_users]

    # Calculate final metrics with safe division
    ap_values = ap_per_user_unnormalized / n_liked_items
    ndcg_values = dcg_per_user / idcg_per_user
    recall_values = total_hits_per_user / n_liked_items

    return {
        Metric.AP: ap_values,
        Metric.NDCG: ndcg_values,
        Metric.RECALL: recall_values,
    }
