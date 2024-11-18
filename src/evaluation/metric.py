import numpy as np
from collections import defaultdict

def ranking_metrics_at_k(liked_items, reco_items, K=10):
    K = min(len(liked_items), len(reco_items), K)
    # map
    mean_ap = 0
    # ndcg
    cg = (1.0 / np.log2(np.arange(2, K + 2)))
    cg_sum = np.cumsum(cg)
    ndcg = 0
    ap = 0
    hit = 0
    idcg = cg_sum[-1]

    likes = defaultdict(bool)
    for item in liked_items:
        likes[item] = True

    for i in range(K):
        if likes[reco_items[i]] == 1:
            hit += 1
            ap += hit / (i + 1)
            ndcg += cg[i] / idcg
    mean_ap += ap / K

    return {
        "map": mean_ap,
        "ndcg": ndcg
    }