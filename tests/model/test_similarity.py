from unittest.mock import patch

import numpy as np
import pytest

from yamyam_lab.similarity import (
    find_similar_items_cli,
    find_similar_items_demo,
    load_model_for_inference,
    main,
)


class TestSimilarityFunctions:
    """Test similarity.py main functions."""

    @pytest.fixture(scope="function")
    def valid_item_id(self, mock_load_dataset):
        """Get a valid item ID for testing."""
        with patch(
            "yamyam_lab.data.base.BaseDatasetLoader.load_dataset",
            return_value=mock_load_dataset,
        ):
            model, _, counts = load_model_for_inference()
            valid_indices = np.where(counts >= 1)[0]
            if len(valid_indices) > 0:
                return model.idx_to_item[valid_indices[0]]
            return list(model.item_mapping.keys())[0]

    def test_load_model_for_inference(self, mock_load_dataset):
        """Test that load_model_for_inference works."""
        with patch(
            "yamyam_lab.data.base.BaseDatasetLoader.load_dataset",
            return_value=mock_load_dataset,
        ):
            model, data, counts = load_model_for_inference()

            assert model is not None
            assert isinstance(data, dict)
            assert "X_train" in data
            assert isinstance(counts, np.ndarray)

    def test_find_similar_items_demo(self, capsys, mock_load_dataset):
        """Test that find_similar_items_demo executes without errors."""
        with patch(
            "yamyam_lab.data.base.BaseDatasetLoader.load_dataset",
            return_value=mock_load_dataset,
        ):
            find_similar_items_demo(top_k=3, num_examples=1, min_interactions=1)

            captured = capsys.readouterr()
            assert "Example 1:" in captured.out
            assert "Restaurant" in captured.out

    def test_find_similar_items_cli(self, capsys, valid_item_id, mock_load_dataset):
        """Test that find_similar_items_cli executes without errors."""
        with patch(
            "yamyam_lab.data.base.BaseDatasetLoader.load_dataset",
            return_value=mock_load_dataset,
        ):
            find_similar_items_cli(
                target_id=valid_item_id,
                top_k=5,
                method="cosine_matrix",
                min_interactions=1,
            )

            captured = capsys.readouterr()
            assert "Target:" in captured.out
            assert "Found" in captured.out

    def test_find_similar_items_cli_with_jaccard(
        self, capsys, valid_item_id, mock_load_dataset
    ):
        """Test that find_similar_items_cli works with jaccard method."""
        with patch(
            "yamyam_lab.data.base.BaseDatasetLoader.load_dataset",
            return_value=mock_load_dataset,
        ):
            find_similar_items_cli(
                target_id=valid_item_id, top_k=3, method="jaccard", min_interactions=1
            )

            captured = capsys.readouterr()
            assert "Target:" in captured.out

    def test_main_function(self, mock_load_dataset):
        """Test that main function executes without errors."""
        import warnings

        # Suppress expected warnings during evaluation
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in divide",
            )
            with patch(
                "yamyam_lab.data.base.BaseDatasetLoader.load_dataset",
                return_value=mock_load_dataset,
            ):
                try:
                    main()
                except SystemExit:
                    pass  # It's ok if it exits normally
                except Exception as e:
                    pytest.fail(f"main() raised unexpected exception: {e}")
