import numpy as np
from numpy.typing import NDArray


def ranking_metrics_at_k(
    liked_items: NDArray, reco_items: NDArray, liked_items_score: NDArray = None
) -> dict[str, float]:
    """
    Calculates ndcg, average precision (aP), hit, and recall for `one user`.
    If you want to derive ndcg, map for n users, you should average them over n.

    liked_items:
        item ids selected by one user in test dataset.
    reco_items:
        item ids recommended for one user.
    liked_items_score:
        `relevance` associated with liked_items. Could be ratings or indicator values depending on target y.
        Default value is None when binary classification. If target y is ratings, scores would be np.array([3,5,4.5]).
        This score is used when calculating ndcg.
    """
    # when target y is rating
    if isinstance(liked_items_score, np.ndarray):
        assert liked_items.shape == liked_items_score.shape
    # when target y is binary
    else:
        liked_items_score = np.array([1] * len(liked_items))

    # number of recommended items
    K = len(reco_items)
    # in case user liked items less than K
    K = min(len(liked_items), K)
    reco_items = reco_items[:K]

    # sort liked_items by descending order
    idx = np.argsort(liked_items_score)[::-1]
    liked_items_score = liked_items_score[idx]
    liked_items = liked_items[idx]

    liked_items2score = {}
    for item_id, score in zip(liked_items, liked_items_score):
        liked_items2score[item_id] = score

    ap = 0
    idcg = (liked_items_score[:K] / np.log2(np.arange(2, K + 2))).sum()
    ndcg = 0
    hit = 0

    for i in range(K):
        score = liked_items2score.get(reco_items[i])
        if score is not None:
            hit += 1
            ap += hit / (i + 1)
            ndcg += (score / np.log2(i + 2)) / idcg
    ap /= K

    # Calculate recall
    real_count = len(liked_items)
    recall = hit / real_count if real_count != 0 else 0

    return {"ap": ap, "ndcg": ndcg, "recall": recall}
