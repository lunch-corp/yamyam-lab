import numpy as np

from evaluation.metric import ranking_metrics_at_k


def test_map_ndcg():
    liked_items = [100, 10, 50, 0, 11, 22, 33, 44, 55, 66]
    # case 1: hit at index 3, 4
    reco_items = [2, 3, 4, 100, 10]
    K = 5
    metric = ranking_metrics_at_k(liked_items, reco_items, K)
    cg = (1.0 / np.log2(np.arange(2, K + 2)))
    idcg = sum(cg)

    expected_ndcg = (cg[3] + cg[4]) / idcg
    expected_map = (1/4 + 2/5) / K

    np.testing.assert_almost_equal(metric["ndcg"], expected_ndcg)
    np.testing.assert_almost_equal(metric["map"], expected_map)

    # case 2: hit at index 0, 1, 2
    reco_items = [100, 10, 50, 2, 3]
    metric = ranking_metrics_at_k(liked_items, reco_items, K)
    expected_ndcg = (cg[0] + cg[1] + cg[2]) / idcg
    expected_map = (1 + 1 + 1) / K
    np.testing.assert_almost_equal(metric["ndcg"], expected_ndcg)
    np.testing.assert_almost_equal(metric["map"], expected_map)
