import numpy as np
import pytest

from yamyam_lab.model.classic_cf.item_based import ItemBasedCollaborativeFiltering
from yamyam_lab.similarity import load_model_for_inference


class TestItemBasedSimilarity:
    """Test Item-based Collaborative Filtering similarity search."""

    @pytest.fixture(scope="class")
    def setup_model(self):
        """Setup test model and data once for all tests."""
        model, data, item_interaction_counts = load_model_for_inference()

        # Get a valid item ID with sufficient interactions
        valid_indices = np.where(item_interaction_counts >= 10)[0]
        valid_item_id = (
            model.idx_to_item[valid_indices[0]]
            if len(valid_indices) > 0
            else list(model.item_mapping.keys())[0]
        )

        return model, data, item_interaction_counts, valid_item_id

    def test_load_model_for_inference(self, setup_model):
        """Test that load_model_for_inference returns valid objects."""
        model, data, counts, _ = setup_model

        assert isinstance(model, ItemBasedCollaborativeFiltering)
        assert isinstance(data, dict)
        assert "X_train" in data
        assert isinstance(counts, np.ndarray)

    def test_find_similar_items_returns_results(self, setup_model):
        """Test that find_similar_items returns valid results."""
        model, _, _, target_id = setup_model

        similar_items = model.find_similar_items(target_item_id=target_id, top_k=10)

        assert isinstance(similar_items, list)
        assert len(similar_items) <= 10

        if similar_items:
            assert "item_id" in similar_items[0]
            assert "similarity_score" in similar_items[0]
            assert 0.0 <= similar_items[0]["similarity_score"] <= 1.0

    def test_find_similar_items_with_jaccard(self, setup_model):
        """Test that Jaccard method works."""
        model, _, _, target_id = setup_model

        similar_items = model.find_similar_items(
            target_item_id=target_id, top_k=5, method="jaccard"
        )

        assert isinstance(similar_items, list)
        assert len(similar_items) <= 5

    def test_find_similar_items_with_invalid_id(self, setup_model):
        """Test that invalid ID returns empty list."""
        model, _, _, _ = setup_model

        similar_items = model.find_similar_items(target_item_id=-999999, top_k=10)

        assert similar_items == []

    def test_recommend_for_user(self, setup_model):
        """Test that recommend_for_user returns valid results."""
        model, _, _, _ = setup_model

        valid_user_id = list(model.user_mapping.keys())[0]
        recommendations = model.recommend_for_user(user_id=valid_user_id, top_k=10)

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 10

        if recommendations:
            assert "item_id" in recommendations[0]
            assert "predicted_score" in recommendations[0]
