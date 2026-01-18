"""Integration tests that actually run training for all models using unified pipeline."""

from unittest.mock import patch

import pytest

from yamyam_lab.train import TrainerFactory


class TestALSModelTraining:
    """Integration test for ALS model training using unified pipeline."""

    def test_als_model_trains(self, setup_als_config, mock_load_dataset):
        """Test that ALS model can complete full training."""
        with patch(
            "yamyam_lab.data.base.BaseDatasetLoader.load_dataset",
            return_value=mock_load_dataset,
        ):
            trainer = TrainerFactory.create_trainer(setup_als_config)
            trainer.train()


class TestGraphModelTraining:
    """Integration test for graph-based model training using unified pipeline."""

    @pytest.mark.parametrize(
        "setup_config",
        [
            ("node2vec", False),
            ("metapath2vec", True),
            ("graphsage", False),
            ("lightgcn", False),
        ],
        indirect=["setup_config"],
    )
    def test_graph_models_train(self, setup_config, mock_load_dataset):
        """Test that graph models can complete full training."""
        with patch(
            "yamyam_lab.data.base.BaseDatasetLoader.load_dataset",
            return_value=mock_load_dataset,
        ):
            trainer = TrainerFactory.create_trainer(setup_config)
            trainer.train()


class TestTorchModelTraining:
    """Integration test for PyTorch-based model training using unified pipeline."""

    @pytest.mark.parametrize(
        "setup_config", [("svd_bias", False)], indirect=["setup_config"]
    )
    def test_torch_models_train(self, setup_config, mock_load_dataset):
        """Test that PyTorch models can complete full training."""
        with patch(
            "yamyam_lab.data.base.BaseDatasetLoader.load_dataset",
            return_value=mock_load_dataset,
        ):
            trainer = TrainerFactory.create_trainer(setup_config)
            trainer.train()
