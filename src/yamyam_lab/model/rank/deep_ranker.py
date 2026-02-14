from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from yamyam_lab.loss.bpr_loss import bpr_loss_sampled
from yamyam_lab.model.rank.base import BaseModel


class DeepRankerModel(nn.Module):
    """
    Deep Learning Ranker using hybrid MLP architecture.

    Architecture:
        - User Embedding (num_users × embed_dim)
        - Diner Embedding (num_diners × embed_dim)
        - Tabular Features → BatchNorm
        - Concatenate: [User Embed | Diner Embed | Features]
        - MLP Layers with BatchNorm + ReLU + Dropout
        - Output: Scalar score
    """

    def __init__(
        self,
        num_users: int,
        num_diners: int,
        num_features: int,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
    ):
        """
        Args:
            num_users (int): Number of unique users.
            num_diners (int): Number of unique diners.
            num_features (int): Number of tabular features.
            embedding_dim (int): Dimension of user/diner embeddings.
            hidden_dims (List[int]): List of hidden layer dimensions.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.num_users = num_users
        self.num_diners = num_diners
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.diner_embedding = nn.Embedding(num_diners, embedding_dim)

        # Feature normalization
        self.feature_norm = nn.BatchNorm1d(num_features)

        # MLP layers
        input_dim = embedding_dim * 2 + num_features
        mlp_layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        mlp_layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*mlp_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.diner_embedding.weight)

        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, user_idx: Tensor, diner_idx: Tensor, features: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            user_idx (Tensor): User indices. Shape: (batch_size,)
            diner_idx (Tensor): Diner indices. Shape: (batch_size,)
            features (Tensor): Tabular features. Shape: (batch_size, num_features)

        Returns (Tensor):
            Predicted scores. Shape: (batch_size,)
        """
        # Get embeddings
        user_emb = self.user_embedding(user_idx)  # (batch_size, embedding_dim)
        diner_emb = self.diner_embedding(diner_idx)  # (batch_size, embedding_dim)

        # Normalize features
        features_norm = self.feature_norm(features)  # (batch_size, num_features)

        # Concatenate all inputs
        x = torch.cat([user_emb, diner_emb, features_norm], dim=1)

        # Pass through MLP
        output = self.mlp(x).squeeze(-1)  # (batch_size,)

        return output


class DeepRankerTrainer(BaseModel):
    """
    Trainer for Deep Learning Ranker using BPR loss.

    This trainer wraps DeepRankerModel and provides the BaseModel interface
    for compatibility with the existing ranking pipeline.
    """

    def __init__(
        self,
        model_path: str,
        results: str,
        features: List[str],
        embedding_dim: int = 64,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        lr: float = 0.001,
        batch_size: int = 2048,
        early_stopping_rounds: int = 10,
        num_boost_round: int = 100,  # epochs
        verbose_eval: int = 5,
        seed: int = 42,
        device: str = "cuda",
        recommend_batch_size: int = 1000,
    ) -> None:
        # Initialize base model without params
        super().__init__(
            model_path=model_path,
            results=results,
            params={},
            early_stopping_rounds=early_stopping_rounds,
            num_boost_round=num_boost_round,
            verbose_eval=verbose_eval,
            seed=seed,
            features=features,
            recommend_batch_size=recommend_batch_size,
        )

        # Deep learning specific parameters
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Model will be initialized in _fit
        self.model = None
        self.num_users = None
        self.num_diners = None

    def _prepare_dataloader(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Prepare PyTorch DataLoader from pandas DataFrame.

        Args:
            X (pd.DataFrame): Features with reviewer_id, diner_idx, and feature columns.
            y (pd.Series | np.ndarray): Target labels (0 or 1).
            shuffle (bool): Whether to shuffle the data.

        Returns (DataLoader):
            PyTorch DataLoader.
        """
        # Extract user and diner indices
        user_idx = torch.LongTensor(X["reviewer_id"].values)
        diner_idx = torch.LongTensor(X["diner_idx"].values)

        # Extract features (exclude user/diner IDs)
        feature_cols = [
            col for col in self.features if col not in ["reviewer_id", "diner_idx"]
        ]
        features = torch.FloatTensor(X[feature_cols].values)

        # Prepare targets
        if isinstance(y, pd.Series):
            targets = torch.FloatTensor(y.values)
        else:
            targets = torch.FloatTensor(y)

        # Create dataset and dataloader
        dataset = TensorDataset(user_idx, diner_idx, features, targets)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        return dataloader

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> DeepRankerModel:
        """
        Train the Deep Ranker model using BPR loss.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series | np.ndarray): Training labels.
            X_valid (pd.DataFrame | None): Validation features.
            y_valid (pd.Series | np.ndarray | None): Validation labels.

        Returns (DeepRankerModel):
            Trained model.
        """
        # Set random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Get number of users and diners
        self.num_users = X_train["reviewer_id"].max() + 1
        self.num_diners = X_train["diner_idx"].max() + 1
        num_features = len(
            [col for col in self.features if col not in ["reviewer_id", "diner_idx"]]
        )

        # Initialize model
        self.model = DeepRankerModel(
            num_users=self.num_users,
            num_diners=self.num_diners,
            num_features=num_features,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        # Prepare data loaders
        train_loader = self._prepare_dataloader(X_train, y_train, shuffle=True)
        if X_valid is not None and y_valid is not None:
            valid_loader = self._prepare_dataloader(X_valid, y_valid, shuffle=False)
        else:
            valid_loader = None

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.num_boost_round):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0

            for user_idx, diner_idx, features, targets in tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_boost_round}",
                leave=False,
            ):
                # Move to device
                user_idx = user_idx.to(self.device)
                diner_idx = diner_idx.to(self.device)
                features = features.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                scores = self.model(user_idx, diner_idx, features)

                # Calculate BPR loss (pass user_ids for user-wise comparison)
                # Enable debug mode for first batch of first epoch
                debug_mode = epoch == 0 and train_batches == 0
                loss = bpr_loss_sampled(
                    scores, targets, user_ids=user_idx, debug=debug_mode
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            avg_train_loss = train_loss / train_batches

            # Validation phase
            if valid_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0

                with torch.no_grad():
                    for user_idx, diner_idx, features, targets in valid_loader:
                        # Move to device
                        user_idx = user_idx.to(self.device)
                        diner_idx = diner_idx.to(self.device)
                        features = features.to(self.device)
                        targets = targets.to(self.device)

                        # Forward pass
                        scores = self.model(user_idx, diner_idx, features)

                        # Calculate BPR loss (pass user_ids for user-wise comparison)
                        loss = bpr_loss_sampled(scores, targets, user_ids=user_idx)

                        val_loss += loss.item()
                        val_batches += 1

                avg_val_loss = val_loss / val_batches

                # Logging
                if (epoch + 1) % self.verbose_eval == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.num_boost_round} - "
                        f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                    )

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_rounds:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                # No validation set
                if (epoch + 1) % self.verbose_eval == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.num_boost_round} - "
                        f"Train Loss: {avg_train_loss:.4f}"
                    )

        return self.model

    def _predict(self, X_test: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict scores for test data.

        Args:
            X_test (pd.DataFrame | np.ndarray): Test features.

        Returns (np.ndarray):
            Predicted scores.
        """
        if self.model is None:
            self.load_model()

        self.model.eval()

        # Prepare dataloader (no targets needed for prediction)
        dummy_targets = np.zeros(len(X_test))
        test_loader = self._prepare_dataloader(X_test, dummy_targets, shuffle=False)

        predictions = []

        with torch.no_grad():
            for user_idx, diner_idx, features, _ in test_loader:
                # Move to device
                user_idx = user_idx.to(self.device)
                diner_idx = diner_idx.to(self.device)
                features = features.to(self.device)

                # Forward pass
                scores = self.model(user_idx, diner_idx, features)

                predictions.append(scores.cpu().numpy())

        return np.concatenate(predictions)

    def save_model(self) -> None:
        """Save the trained model to disk."""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        model_file = Path(self.model_path) / f"{self.results}.pt"

        # Save model state dict and metadata
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_users": self.num_users,
                "num_diners": self.num_diners,
                "embedding_dim": self.embedding_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "features": self.features,
            },
            model_file,
        )

        print(f"Model saved to {model_file}")

    def load_model(self) -> DeepRankerModel:
        """Load a trained model from disk."""
        model_file = Path(self.model_path) / f"{self.results}.pt"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Load checkpoint
        checkpoint = torch.load(model_file, map_location=self.device)

        # Extract metadata
        self.num_users = checkpoint["num_users"]
        self.num_diners = checkpoint["num_diners"]
        self.embedding_dim = checkpoint["embedding_dim"]
        self.hidden_dims = checkpoint["hidden_dims"]
        self.dropout = checkpoint["dropout"]
        self.features = checkpoint["features"]

        # Initialize model
        num_features = len(
            [col for col in self.features if col not in ["reviewer_id", "diner_idx"]]
        )
        self.model = DeepRankerModel(
            num_users=self.num_users,
            num_diners=self.num_diners,
            num_features=num_features,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        # Load state dict
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Model loaded from {model_file}")

        return self.model

    def calculate_rank(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rank for candidates.

        Override base method to ensure ID columns are included in predictions.

        Args:
            candidates (pd.DataFrame): Candidates to calculate rank.

        Returns (pd.DataFrame):
            Candidates with rank.
        """
        predictions = np.zeros(len(candidates))

        num_batches = (
            len(candidates) + self.recommend_batch_size - 1
        ) // self.recommend_batch_size

        for i in tqdm(range(num_batches)):
            start_idx = i * self.recommend_batch_size
            end_idx = min((i + 1) * self.recommend_batch_size, len(candidates))

            # Include both ID columns and features for DeepRanker prediction
            required_cols = ["reviewer_id", "diner_idx"] + [
                col for col in self.features if col not in ["reviewer_id", "diner_idx"]
            ]
            batch = candidates[required_cols].iloc[start_idx:end_idx]
            predictions[start_idx:end_idx] = self.predict(batch)

        candidates["pred_score"] = predictions
        candidates = candidates.sort_values(
            by=["reviewer_id", "pred_score"], ascending=[True, False]
        )

        return candidates
