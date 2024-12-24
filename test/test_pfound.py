import numpy as np
from numpy.typing import NDArray


def ranking_metrics_at_k(
    liked_items: NDArray,
    reco_items: NDArray,
    liked_items_score: NDArray = None,
    gamma: float = 0.85,
) -> dict[str, float]:
    """
    Calculates ndcg, average precision (AP), recall, and PFound for one user.
    If you want to derive ndcg, map, recall, and PFound for multiple users, average them over n users.

    Parameters
    ----------
    liked_items : NDArray
        Item IDs liked by one user in the test dataset.
    reco_items : NDArray
        Item IDs recommended for one user.
    liked_items_score : NDArray, optional
        Relevance scores associated with liked_items (e.g., ratings or binary indicators).
        If None, binary relevance is assumed.
    gamma : float, optional
        Decay factor for PFound, by default `0.85`.

    Returns
    -------
    dict[str, float]
        A dictionary with keys "ap", "ndcg", "recall", and "pfound".
    """
    # When target y is binary, set all scores to 1
    if liked_items_score is None:
        liked_items_score = np.ones(len(liked_items))
    else:
        assert liked_items.shape == liked_items_score.shape

    # Number of recommended items to evaluate
    K = len(reco_items)
    # Limit K to the number of liked items if it's smaller
    K = min(len(liked_items), K)
    reco_items = reco_items[:K]

    # Sort liked items by descending relevance
    idx = np.argsort(liked_items_score)[::-1]
    liked_items = liked_items[idx]
    liked_items_score = liked_items_score[idx]

    # Ideal DCG calculation (sorted by relevance)
    idcg = (liked_items_score[:K] / np.log2(np.arange(2, K + 2))).sum()

    # Initialize metrics
    ap = 0
    ndcg = 0
    hits = 0
    pfound = 0
    p_last = 1  # Initial probability of continuing exploration

    # Track seen items
    seen_items = set()

    for i, item in enumerate(reco_items):
        if item in liked_items and item not in seen_items:
            seen_items.add(item)
            hits += 1
            relevance = liked_items_score[np.where(liked_items == item)[0][0]]
            ap += hits / (i + 1)
            ndcg += relevance / np.log2(i + 2)

            # PFound calculation
            click_prob = relevance
            pfound += p_last * click_prob / (i + 1)
            p_last *= (1 - click_prob) * gamma  # Update p_last with decay

    ap /= K
    ndcg /= idcg if idcg > 0 else 1  # Avoid division by zero
    recall = hits / len(liked_items) if len(liked_items) > 0 else 0

    return {"ap": ap, "ndcg": ndcg, "recall": recall, "pfound": pfound}


liked_items = np.array([100, 10, 50, 0, 11, 22, 33, 44, 55, 66])

# case 1: hit at item_id 100, 10
reco_items = np.array([2, 3, 4, 100, 10])


# Call the function
metrics = ranking_metrics_at_k(liked_items, reco_items)
print(metrics)
