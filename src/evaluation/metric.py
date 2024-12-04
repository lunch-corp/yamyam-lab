from typing import Iterable

import numpy as np
from numpy.typing import NDArray


def _hit_at_k(actual: list[int], predicted: list[int], k: int = 10) -> float:
    """Compute Hit@K for a single user."""
    if len(predicted) > k:
        predicted = predicted[:k]

    # 적어도 하나의 관련 항목이 추천에 포함되면 Hit
    return 1.0 if any(p in actual for p in predicted) else 0.0


def hit_at_k(actual: Iterable, predicted: Iterable, k: int = 10) -> float:
    """Compute mean Hit@K across all users.

    Parameters
    ----------
    actual : Iterable
        Label (ground truth).
    predicted : Iterable
        Predictions.
    k : int, optional
        k, by default ``10``.

    Returns
    -------
    float
        Mean Hit@K.
    """
    return np.mean([_hit_at_k(a, p, k) for a, p in zip(actual, predicted) if a is not None])


def _recall_at_k(actual: list[int], predicted: list[int], k: int = 10) -> float:
    """Compute Recall@K for a single user."""
    if len(predicted) > k:
        predicted = predicted[:k]

    relevant_items = [p for p in predicted if p in actual]
    if len(actual) == 0:
        return 0.0

    # Recall: 추천된 항목 중 실제 관련 항목의 비율
    return len(relevant_items) / len(actual)


def recall_at_k(actual: Iterable, predicted: Iterable, k: int = 10) -> float:
    """Compute mean Recall@K across all users.

    Parameters
    ----------
    actual : Iterable
        Label (ground truth).
    predicted : Iterable
        Predictions.
    k : int, optional
        k, by default ``10``.

    Returns
    -------
    float
        Mean Recall@K.
    """
    return np.mean([_recall_at_k(a, p, k) for a, p in zip(actual, predicted) if a is not None])


def _ap_at_k(actual: list[int], predicted: list[int], k: int = 10) -> float:
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if actual is None:
        return 0.0

    return score / min(len(actual), k)


def map_at_k(actual: Iterable, predicted: Iterable, k: int = 12) -> float:
    """Compute mean average precision @ k.

    Parameters
    ----------
    actual : Iterable
        Label.
    predicted : Iterable
        Predictions.
    k : int, optional
        k, by default ``12``.

    Returns
    -------
    float
        MAP@k.
    """
    return np.mean([_ap_at_k(a, p, k) for a, p in zip(actual, predicted) if a is not None])


def _dcg_at_k(actual: list[int], predicted: list[int], k: int = 10) -> float:
    """Compute DCG@K for a single user."""
    if len(predicted) > k:
        predicted = predicted[:k]

    dcg = 0.0
    for i, p in enumerate(predicted):
        if p in actual:
            # DCG 계산: relevance / log2(position + 1)
            dcg += 1.0 / np.log2(i + 2)  # log2(i+2) to handle 1-based indexing

    return dcg


def _idcg_at_k(actual: list[int], k: int = 10) -> float:
    """Compute ideal DCG@K for a single user."""
    actual = actual[:k]
    idcg = sum(1.0 / np.log2(i + 2) for i in range(len(actual)))  # Ideal ranking

    return idcg


def _ndcg_at_k(actual: list[int], predicted: list[int], k: int = 10) -> float:
    """Compute NDCG@K for a single user."""
    dcg = _dcg_at_k(actual, predicted, k)
    idcg = _idcg_at_k(actual, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def ndcg_at_k(actual: Iterable, predicted: Iterable, k: int = 12) -> float:
    """Compute mean NDCG@K across all users.

    Parameters
    ----------
    actual : Iterable
        Label (ground truth).
    predicted : Iterable
        Predictions.
    k : int, optional
        k, by default ``12``.

    Returns
    -------
    float
        Mean NDCG@K.
    """
    return np.mean([_ndcg_at_k(a, p, k) for a, p in zip(actual, predicted) if a is not None])


def ranking_metrics_at_k(liked_items: NDArray, reco_items: NDArray, liked_items_score: NDArray = None):
    """
    Calculates ndcg, average precision (aP) for `one user`.
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

    return {"ap": ap, "ndcg": ndcg}
