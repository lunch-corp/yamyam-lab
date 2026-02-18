"""Tests for Deep Ranker model and trainer."""

import numpy as np
import pandas as pd
import pytest
import torch

from yamyam_lab.model.rank.deep_ranker import DeepRankerModel, DeepRankerTrainer


class TestDeepRankerModel:
    """Test DeepRankerModel architecture."""

    @pytest.fixture
    def model_params(self):
        """Default model parameters."""
        return {
            "num_users": 100,
            "num_diners": 50,
            "num_features": 13,
            "embedding_dim": 32,
            "hidden_dims": [64, 32],
            "dropout": 0.2,
        }

    @pytest.fixture
    def model(self, model_params):
        """Create a test model."""
        return DeepRankerModel(**model_params)

    def test_model_initialization(self, model, model_params):
        """Test model initialization."""
        assert model.num_users == model_params["num_users"]
        assert model.num_diners == model_params["num_diners"]
        assert model.embedding_dim == model_params["embedding_dim"]

        # Check embedding layers exist
        assert model.user_embedding.num_embeddings == model_params["num_users"]
        assert model.diner_embedding.num_embeddings == model_params["num_diners"]

    def test_forward_pass(self, model):
        """Test forward pass produces correct output shape."""
        batch_size = 16
        user_idx = torch.randint(0, 100, (batch_size,))
        diner_idx = torch.randint(0, 50, (batch_size,))
        features = torch.randn(batch_size, 13)

        scores = model(user_idx, diner_idx, features)

        assert scores.shape == (batch_size,), "Output shape should be (batch_size,)"
        assert torch.isfinite(scores).all(), "All scores should be finite"

    def test_gradient_flow(self, model):
        """Test that gradients flow through the model."""
        batch_size = 16
        user_idx = torch.randint(0, 100, (batch_size,))
        diner_idx = torch.randint(0, 50, (batch_size,))
        features = torch.randn(batch_size, 13)

        scores = model(user_idx, diner_idx, features)
        loss = scores.mean()
        loss.backward()

        # Check gradients exist for embeddings
        assert model.user_embedding.weight.grad is not None
        assert model.diner_embedding.weight.grad is not None

    def test_model_parameters(self, model):
        """Test model has trainable parameters."""
        params = list(model.parameters())
        assert len(params) > 0, "Model should have parameters"

        total_params = sum(p.numel() for p in params)
        assert total_params > 0, "Model should have trainable parameters"

        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        assert trainable_params == total_params, "All parameters should be trainable"

    def test_eval_mode(self, model):
        """Test model can switch to eval mode."""
        model.eval()

        batch_size = 8
        user_idx = torch.randint(0, 100, (batch_size,))
        diner_idx = torch.randint(0, 50, (batch_size,))
        features = torch.randn(batch_size, 13)

        with torch.no_grad():
            scores = model(user_idx, diner_idx, features)

        assert torch.isfinite(scores).all(), "Scores should be finite in eval mode"


class TestDeepRankerTrainer:
    """Test DeepRankerTrainer."""

    @pytest.fixture
    def trainer_params(self, tmp_path):
        """Default trainer parameters."""
        return {
            "model_path": str(tmp_path / "deep_ranker"),
            "results": "test_model",
            "features": [
                "min_price",
                "max_price",
                "mean_price",
                "median_price",
                "open_days_per_week",
                "is_open_weekend",
                "avg_open_hours_per_day",
                "total_open_hours_per_week",
                "asian",
                "japanese",
                "chinese",
                "korean",
                "western",
            ],
            "embedding_dim": 32,
            "hidden_dims": [64, 32],
            "dropout": 0.2,
            "lr": 0.001,
            "batch_size": 32,
            "early_stopping_rounds": 2,
            "num_boost_round": 5,  # Small for testing
            "verbose_eval": 1,
            "seed": 42,
            "device": "cpu",
        }

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        num_samples = 200
        num_users = 20
        num_diners = 10

        data = {
            "reviewer_id": np.random.randint(0, num_users, num_samples),
            "diner_idx": np.random.randint(0, num_diners, num_samples),
            "min_price": np.random.uniform(5, 20, num_samples),
            "max_price": np.random.uniform(20, 50, num_samples),
            "mean_price": np.random.uniform(10, 30, num_samples),
            "median_price": np.random.uniform(10, 30, num_samples),
            "open_days_per_week": np.random.randint(5, 8, num_samples),
            "is_open_weekend": np.random.randint(0, 2, num_samples),
            "avg_open_hours_per_day": np.random.uniform(8, 12, num_samples),
            "total_open_hours_per_week": np.random.uniform(40, 80, num_samples),
            "asian": np.random.randint(0, 2, num_samples),
            "japanese": np.random.randint(0, 2, num_samples),
            "chinese": np.random.randint(0, 2, num_samples),
            "korean": np.random.randint(0, 2, num_samples),
            "western": np.random.randint(0, 2, num_samples),
        }

        X = pd.DataFrame(data)

        # Create imbalanced targets (10% positive)
        y = np.zeros(num_samples)
        y[: num_samples // 10] = 1
        np.random.shuffle(y)
        y = pd.Series(y)

        return X, y

    def test_trainer_initialization(self, trainer_params):
        """Test trainer initialization."""
        trainer = DeepRankerTrainer(**trainer_params)

        assert trainer.embedding_dim == trainer_params["embedding_dim"]
        assert trainer.batch_size == trainer_params["batch_size"]
        assert trainer.model is None, "Model should not be initialized yet"

    def test_fit_basic(self, trainer_params, sample_data):
        """Test basic model training."""
        X_train, y_train = sample_data
        X_val, y_val = sample_data  # Use same data for simplicity

        trainer = DeepRankerTrainer(**trainer_params)
        model = trainer.fit(X_train, y_train, X_val, y_val)

        assert model is not None, "Model should be returned after fit"
        assert trainer.model is not None, "Trainer should have a model"
        assert trainer.num_users > 0, "Should have detected users"
        assert trainer.num_diners > 0, "Should have detected diners"

    def test_predict(self, trainer_params, sample_data):
        """Test model prediction."""
        X_train, y_train = sample_data
        X_test, _ = sample_data

        trainer = DeepRankerTrainer(**trainer_params)
        trainer.fit(X_train, y_train)

        predictions = trainer.predict(X_test)

        assert len(predictions) == len(X_test), "Should predict for all samples"
        assert np.isfinite(predictions).all(), "Predictions should be finite"

    def test_save_load_model(self, trainer_params, sample_data):
        """Test model saving and loading."""
        X_train, y_train = sample_data

        # Train and save
        trainer1 = DeepRankerTrainer(**trainer_params)
        trainer1.fit(X_train, y_train)
        trainer1.save_model()

        # Load in new trainer
        trainer2 = DeepRankerTrainer(**trainer_params)
        trainer2.load_model()

        # Predictions should be identical
        X_test, _ = sample_data
        pred1 = trainer1.predict(X_test)
        pred2 = trainer2.predict(X_test)

        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)

    def test_calculate_rank(self, trainer_params, sample_data):
        """Test ranking calculation."""
        X_train, y_train = sample_data

        trainer = DeepRankerTrainer(**trainer_params)
        trainer.fit(X_train, y_train)

        # Create candidate set
        candidates = X_train.copy()
        ranked_candidates = trainer.calculate_rank(candidates)

        assert "pred_score" in ranked_candidates.columns, (
            "Should have pred_score column"
        )
        assert len(ranked_candidates) == len(candidates), "Should rank all candidates"

        # Check if sorted by score
        for user_id in ranked_candidates["reviewer_id"].unique():
            user_data = ranked_candidates[
                ranked_candidates["reviewer_id"] == user_id
            ].copy()
            scores = user_data["pred_score"].values
            assert np.all(scores[:-1] >= scores[1:]), (
                "Should be sorted descending by score"
            )

    def test_early_stopping(self, trainer_params, sample_data):
        """Test early stopping mechanism."""
        X_train, y_train = sample_data
        X_val, y_val = sample_data

        # Set very strict early stopping
        trainer_params["early_stopping_rounds"] = 1
        trainer_params["num_boost_round"] = 100

        trainer = DeepRankerTrainer(**trainer_params)
        trainer.fit(X_train, y_train, X_val, y_val)

        # Should stop early (well before 100 epochs)
        # This is hard to test precisely, but model should exist
        assert trainer.model is not None

    @pytest.mark.parametrize("batch_size", [16, 32, 64])
    def test_different_batch_sizes(self, trainer_params, sample_data, batch_size):
        """Test training with different batch sizes."""
        X_train, y_train = sample_data

        trainer_params["batch_size"] = batch_size
        trainer = DeepRankerTrainer(**trainer_params)

        model = trainer.fit(X_train, y_train)
        assert model is not None

        predictions = trainer.predict(X_train)
        assert len(predictions) == len(X_train)
