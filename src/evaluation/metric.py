from typing import Dict

import numpy as np
from numpy.typing import NDArray

from constant.metric.metric import Metric
from tools.utils import safe_divide


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
    P(i) is precision at i and r(i) is indicator variable 1 if ith item is hitted else 0.
    Here, precision@i = # of hitted items until ith ranked item / K

    Note that for some variations of AP, \dfrac{1}{ min(m, K) } is used instead of \dfrac{1}{m} to prevent deviding
    large value when K < m for some special case. If defined denominator as min(m, K), map may not increase as K is
    getting larger.

    For normalized discounted cumulative gain, we use following definition.

    NDCG = \dfrac{DCG}{IDCG}
    where DCG = \sum_{i=1}^{K} \dfrac{1}{\log2{i+1}} * r(i) and
    IDCG = \sum_{i=1}^{M} \dfrac{1}{\log2{i+1}}.
    Here, r(i) is indicator variable 1 if ith item is hitted else 0,
    K is number of recommended items, and M is number of items that users liked.

    Maximum value of NDCG is IDCG by their definitions, therefore, 0 <= NDCG <= 1.

    Note that NDCG does NOT gurantee increasing value as K is getting larger.

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

    return {Metric.AP.value: ap, Metric.NDCG.value: ndcg, Metric.RECALL.value: recall}
