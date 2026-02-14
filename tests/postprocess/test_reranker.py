"""Unit tests for postprocess reranking modules."""

import numpy as np
import pandas as pd
import pytest

from yamyam_lab.postprocess.base import BaseReranker
from yamyam_lab.postprocess.hidden import HiddenReranker, Reranker
from yamyam_lab.tools.rerank import geo_similarity_haversine


class _ConcreteReranker(Reranker):
    """Concrete Reranker subclass for testing.

    Reranker._greedy_mmr_loop calls _update_similarities which is only
    implemented by HiddenReranker.  This subclass provides the method so
    the base MMR logic can be tested in isolation.
    """

    def _update_similarities(
        self, best_idx, alive, current_max_sim, cat_codes, lat_rad, lon_rad
    ):
        if alive.any():
            aidx = np.flatnonzero(alive)
            sim_cat = (cat_codes[aidx] == cat_codes[best_idx]).astype(np.float32)
            sim_geo = geo_similarity_haversine(
                lat_rad[aidx],
                lon_rad[aidx],
                float(lat_rad[best_idx]),
                float(lon_rad[best_idx]),
                self.geo_tau_km,
            )
            np.maximum(
                current_max_sim[aidx],
                self.w_cat * sim_cat + self.w_geo * sim_geo,
                out=current_max_sim[aidx],
            )


@pytest.fixture
def item_meta():
    """Create mock item metadata for reranking tests."""
    return pd.DataFrame(
        {
            "diner_idx": [1, 2, 3, 4, 5, 6, 7, 8],
            "diner_category_large": [
                "한식",
                "한식",
                "일식",
                "일식",
                "양식",
                "양식",
                "중식",
                "중식",
            ],
            "diner_lat": [37.56, 37.57, 37.58, 37.59, 37.50, 37.51, 37.52, 37.53],
            "diner_lon": [
                126.97,
                126.98,
                126.99,
                127.00,
                126.93,
                126.94,
                126.95,
                126.96,
            ],
        }
    )


@pytest.fixture
def base_scores():
    """Create mock base scores."""
    return np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], dtype=np.float32)


@pytest.fixture
def item_ids():
    """Create mock item IDs."""
    return np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)


class TestBaseReranker:
    """Tests for BaseReranker abstract class."""

    def test_cannot_instantiate_directly(self):
        """BaseReranker is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseReranker()

    def test_lambda_div_clipping(self):
        """lambda_div is clipped to [0, 1]."""
        reranker = Reranker(lambda_div=1.5)
        assert reranker.lambda_div == 1.0

        reranker = Reranker(lambda_div=-0.5)
        assert reranker.lambda_div == 0.0

    def test_default_parameters(self):
        """Default parameters are set correctly."""
        reranker = Reranker()
        assert reranker.lambda_div == 0.55
        assert reranker.w_cat == 0.5
        assert reranker.w_geo == 0.5
        assert reranker.geo_tau_km == 2.0
        assert reranker.coverage_min == {}
        assert reranker.coverage_max == {}
        assert reranker.prefix_freeze == 0

    def test_coverage_none_defaults_to_empty_dict(self):
        """None coverage params default to empty dict."""
        reranker = Reranker(coverage_min=None, coverage_max=None)
        assert reranker.coverage_min == {}
        assert reranker.coverage_max == {}


class TestReranker:
    """Tests for MMR Reranker via _ConcreteReranker."""

    def test_rerank_returns_correct_count(self, item_ids, base_scores, item_meta):
        """Reranker returns exactly k items."""
        reranker = _ConcreteReranker()
        result_ids, result_scores = reranker.rerank(
            item_ids=item_ids, base_scores=base_scores, item_meta=item_meta, k=5
        )
        assert len(result_ids) == 5

    def test_rerank_returns_subset_of_input(self, item_ids, base_scores, item_meta):
        """All returned IDs are from the input set."""
        reranker = _ConcreteReranker()
        result_ids, _ = reranker.rerank(
            item_ids=item_ids, base_scores=base_scores, item_meta=item_meta, k=5
        )
        for rid in result_ids:
            assert rid in item_ids

    def test_rerank_no_duplicates(self, item_ids, base_scores, item_meta):
        """Returned IDs contain no duplicates."""
        reranker = _ConcreteReranker()
        result_ids, _ = reranker.rerank(
            item_ids=item_ids, base_scores=base_scores, item_meta=item_meta, k=5
        )
        assert len(set(result_ids)) == len(result_ids)

    def test_rerank_empty_input(self):
        """Empty input returns empty result."""
        reranker = _ConcreteReranker()
        result_ids, result_scores = reranker.rerank(
            item_ids=np.array([], dtype=np.int64),
            base_scores=np.array([], dtype=np.float32),
            item_meta=pd.DataFrame(
                columns=["diner_idx", "diner_category_large", "diner_lat", "diner_lon"]
            ),
            k=5,
        )
        assert len(result_ids) == 0

    def test_rerank_k_larger_than_items(self, item_ids, base_scores, item_meta):
        """When k > num items, returns all items."""
        reranker = _ConcreteReranker()
        result_ids, _ = reranker.rerank(
            item_ids=item_ids, base_scores=base_scores, item_meta=item_meta, k=100
        )
        assert len(result_ids) == len(item_ids)

    def test_high_lambda_preserves_relevance_order(
        self, item_ids, base_scores, item_meta
    ):
        """High lambda_div (pure relevance) keeps top items first."""
        reranker = _ConcreteReranker(lambda_div=1.0)
        result_ids, _ = reranker.rerank(
            item_ids=item_ids, base_scores=base_scores, item_meta=item_meta, k=3
        )
        # Top scored item should be first
        assert result_ids[0] == 1

    def test_low_lambda_promotes_diversity(self, item_ids, base_scores, item_meta):
        """Low lambda_div promotes diversity over pure relevance."""
        reranker = _ConcreteReranker(lambda_div=0.1)
        result_ids, _ = reranker.rerank(
            item_ids=item_ids, base_scores=base_scores, item_meta=item_meta, k=4
        )
        # Result should include items from different categories
        cats = item_meta.set_index("diner_idx").loc[result_ids]["diner_category_large"]
        assert cats.nunique() > 1


class TestHiddenReranker:
    """Tests for HiddenReranker."""

    @pytest.fixture
    def hidden_df(self):
        """Create mock DataFrame for HiddenReranker."""
        return pd.DataFrame(
            {
                "diner_idx": [1, 2, 3, 4, 5, 6],
                "diner_category_large": [
                    "한식",
                    "한식",
                    "일식",
                    "양식",
                    "중식",
                    "한식",
                ],
                "diner_lat": [37.56, 37.57, 37.58, 37.50, 37.52, 37.56],
                "diner_lon": [126.97, 126.98, 126.99, 126.93, 126.95, 126.97],
                "diner_road_address": [
                    "서울 강남구 A",
                    "서울 강남구 B",
                    "서울 강남구 C",
                    "서울 서초구 D",
                    "서울 서초구 E",
                    "서울 강남구 F",
                ],
                "bayesian_score": [3.0, 2.5, 4.0, 3.5, 2.0, 1.5],
                "avg_rating": [4.5, 3.5, 4.0, 4.8, 3.0, 2.5],
                "recent_review_count": [10, 5, 8, 15, 3, 1],
            }
        )

    def test_init_default_params(self):
        """Default parameters are set correctly."""
        reranker = HiddenReranker()
        assert reranker.n_auto_hotspots == 10
        assert reranker.periphery_strength == 0.5
        assert reranker.periphery_cap == 0.5
        assert reranker.rating_weight == 0.2
        assert reranker.recent_weight == 0.3

    def test_rerank_returns_dataframe(self, hidden_df):
        """HiddenReranker returns a DataFrame."""
        reranker = HiddenReranker(n_auto_hotspots=2)
        result = reranker.rerank(df=hidden_df, k=3)
        assert isinstance(result, pd.DataFrame)

    def test_rerank_returns_k_items(self, hidden_df):
        """Result contains at most k items."""
        reranker = HiddenReranker(n_auto_hotspots=2)
        result = reranker.rerank(df=hidden_df, k=3)
        assert len(result) <= 3

    def test_rerank_adds_bonus_columns(self, hidden_df):
        """Result DataFrame includes periphery_bonus and hidden_score columns."""
        reranker = HiddenReranker(n_auto_hotspots=2)
        result = reranker.rerank(df=hidden_df, k=3)
        assert "periphery_bonus" in result.columns
        assert "hidden_score" in result.columns

    def test_rerank_empty_df(self):
        """Empty DataFrame returns empty result."""
        reranker = HiddenReranker()
        result = reranker.rerank(
            df=pd.DataFrame(
                columns=[
                    "diner_idx",
                    "diner_category_large",
                    "diner_lat",
                    "diner_lon",
                    "diner_road_address",
                    "bayesian_score",
                ]
            ),
            k=5,
        )
        assert len(result) == 0

    def test_rerank_k_zero(self, hidden_df):
        """k=0 returns empty result."""
        reranker = HiddenReranker(n_auto_hotspots=2)
        result = reranker.rerank(df=hidden_df, k=0)
        assert len(result) == 0

    def test_periphery_bonus_capped(self, hidden_df):
        """Periphery bonus does not exceed periphery_cap."""
        cap = 0.3
        reranker = HiddenReranker(
            n_auto_hotspots=2, periphery_strength=1.0, periphery_cap=cap
        )
        result = reranker.rerank(df=hidden_df, k=6)
        if "periphery_bonus" in result.columns:
            assert result["periphery_bonus"].max() <= cap + 1e-6
