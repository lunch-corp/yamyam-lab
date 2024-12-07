import numpy as np

from evaluation.metric import ranking_metrics_at_k


def test_map_ndcg_y_binary():
    liked_items = np.array([100, 10, 50, 0, 11, 22, 33, 44, 55, 66])
    # case 1: hit at item_id 100, 10
    reco_items = np.array([2, 3, 4, 100, 10])
    K = len(reco_items)
    metric = ranking_metrics_at_k(liked_items, reco_items)
    dcg = 1.0 / np.log2(np.arange(2, K + 2))
    idcg = np.sum(dcg)

    expected_ndcg = (dcg[3] + dcg[4]) / idcg
    expected_map = (1 / 4 + 2 / 5) / K

    np.testing.assert_almost_equal(metric["ndcg"], expected_ndcg)
    np.testing.assert_almost_equal(metric["ap"], expected_map)

    # case 2: hit at item_id 100, 10, 50
    reco_items = np.array([100, 10, 50, 2, 3])
    metric = ranking_metrics_at_k(liked_items, reco_items)
    expected_ndcg = (dcg[0] + dcg[1] + dcg[2]) / idcg
    expected_map = (1 + 1 + 1) / K
    np.testing.assert_almost_equal(metric["ndcg"], expected_ndcg)
    np.testing.assert_almost_equal(metric["ap"], expected_map)


def test_map_ndcg_y_rating():
    liked_items = np.array([100, 10, 50, 0, 11, 22, 33, 44, 55, 66])
    scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # case 1: hit at item_id 100, 10, 50
    reco_items = np.array([100, 10, 50, 2, 3])  # hit at item_id 100, 10
    K = len(reco_items)
    metric = ranking_metrics_at_k(liked_items, reco_items, scores)

    idcg = np.array(
        [s / np.log2(i + 2) for i, s in enumerate(scores[np.argsort(scores)[::-1]][:K])]
    ).sum()
    expected_ndcg = (
        scores[0] / np.log2(0 + 2)
        + scores[1] / np.log2(1 + 2)
        + scores[2] / np.log2(2 + 2)
    ) / idcg
    expected_map = (1 + 1 + 1) / K
    np.testing.assert_almost_equal(metric["ndcg"], expected_ndcg)
    np.testing.assert_almost_equal(metric["ap"], expected_map)
