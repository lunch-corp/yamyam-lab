from __future__ import annotations

import copy
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
    Deep Learning Ranker using hybrid MLP architecture with proper regularization.

    Architecture:
        - User Embedding (num_users × small_embed_dim)
        - Diner Embedding (num_diners × small_embed_dim)
        - Tabular Features → Layer normalization (more stable than BatchNorm)
        - Concatenate: [User Embed | Diner Embed | Features]
        - MLP Layers with LayerNorm + ReLU + Dropout
        - Output: Sigmoid-bounded score (0-1)
    """

    def __init__(
        self,
        num_users: int,
        num_diners: int,
        num_features: int,
        embedding_dim: int = 16,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.5,
        num_categorical: int = 0,
        categorical_embedding_dim: int = 8,
        categorical_embedding_buckets: int = 20,
    ):
        """
        Args:
            num_users (int): Number of unique users.
            num_diners (int): Number of unique diners.
            num_features (int): Number of tabular features (continuous only when
                categorical embedding is used).
            embedding_dim (int): Dimension of user/diner embeddings.
            hidden_dims (List[int]): List of hidden layer dimensions.
            dropout (float): Dropout probability.
            num_categorical (int): Number of categorical features to embed.
            categorical_embedding_dim (int): Embedding dim per categorical feature.
            categorical_embedding_buckets (int): Number of buckets for discretization.
        """
        super().__init__()

        self.num_users = num_users
        self.num_diners = num_diners
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.num_categorical = num_categorical
        self.categorical_embedding_dim = categorical_embedding_dim
        self.categorical_embedding_buckets = categorical_embedding_buckets

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.diner_embedding = nn.Embedding(num_diners, embedding_dim)

        # Categorical feature embeddings (bucket index → dense vector)
        if num_categorical > 0:
            self.categorical_embeddings = nn.ModuleList(
                [
                    nn.Embedding(
                        categorical_embedding_buckets, categorical_embedding_dim
                    )
                    for _ in range(num_categorical)
                ]
            )
            cat_total_dim = num_categorical * categorical_embedding_dim
        else:
            self.categorical_embeddings = None
            cat_total_dim = 0

        # Layer normalization for continuous features
        if num_features > 0:
            self.feature_norm = nn.LayerNorm(num_features)
        else:
            self.feature_norm = None

        # MLP layers
        input_dim = embedding_dim * 2 + num_features + cat_total_dim
        mlp_layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            mlp_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output layer with sigmoid to bound scores
        mlp_layers.append(nn.Linear(prev_dim, 1))
        mlp_layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*mlp_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using uniform initialization (better for embeddings)."""
        nn.init.uniform_(self.user_embedding.weight, -0.05, 0.05)
        nn.init.uniform_(self.diner_embedding.weight, -0.05, 0.05)

        if self.categorical_embeddings is not None:
            for emb in self.categorical_embeddings:
                nn.init.uniform_(emb.weight, -0.05, 0.05)

        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        user_idx: Tensor,
        diner_idx: Tensor,
        features: Tensor,
        categorical_bucket_idx: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            user_idx (Tensor): User indices. Shape: (batch_size,)
            diner_idx (Tensor): Diner indices. Shape: (batch_size,)
            features (Tensor): Continuous features. Shape: (batch_size, num_features)
            categorical_bucket_idx (Tensor | None): Bucket indices for categorical
                features. Shape: (batch_size, num_categorical). LongTensor.

        Returns (Tensor):
            Predicted scores. Shape: (batch_size,)
        """
        user_emb = self.user_embedding(user_idx)
        diner_emb = self.diner_embedding(diner_idx)

        parts = [user_emb, diner_emb]

        # Continuous features
        if self.feature_norm is not None and features.shape[1] > 0:
            parts.append(self.feature_norm(features))

        # Categorical feature embeddings
        if (
            self.categorical_embeddings is not None
            and categorical_bucket_idx is not None
        ):
            cat_embs = [
                emb(categorical_bucket_idx[:, i])
                for i, emb in enumerate(self.categorical_embeddings)
            ]
            parts.append(torch.cat(cat_embs, dim=1))

        x = torch.cat(parts, dim=1)
        output = self.mlp(x).squeeze(-1)

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
        embedding_dim: int = 16,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.5,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 2048,
        early_stopping_rounds: int = 10,
        num_boost_round: int = 100,
        verbose_eval: int = 5,
        seed: int = 42,
        device: str = "cuda",
        use_categorical_embedding: bool = False,
        categorical_features: List[str] | None = None,
        categorical_embedding_dim: int = 8,
        categorical_embedding_buckets: int = 20,
    ) -> None:
        super().__init__(
            model_path=model_path,
            results=results,
            params={},
            early_stopping_rounds=early_stopping_rounds,
            num_boost_round=num_boost_round,
            verbose_eval=verbose_eval,
            seed=seed,
            features=features,
        )

        # Deep learning specific parameters
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Categorical embedding parameters
        self.use_categorical_embedding = use_categorical_embedding
        self.categorical_features = categorical_features or []
        self.categorical_embedding_dim = categorical_embedding_dim
        self.categorical_embedding_buckets = categorical_embedding_buckets
        self._bucket_boundaries: dict[str, np.ndarray] | None = None

        # Model will be initialized in _fit
        self.model = None
        self.num_users = None
        self.num_diners = None

    def _compute_bucket_boundaries(self, X: pd.DataFrame) -> None:
        """Compute quantile-based bucket boundaries from training data."""
        self._bucket_boundaries = {}
        n_buckets = self.categorical_embedding_buckets
        for col in self.categorical_features:
            quantiles = np.linspace(0, 1, n_buckets + 1)[1:-1]
            boundaries = np.quantile(X[col].values, quantiles)
            boundaries = np.unique(boundaries)
            self._bucket_boundaries[col] = boundaries

    def _bucketize(self, X: pd.DataFrame) -> torch.LongTensor:
        """Convert categorical feature values to bucket indices."""
        bucket_indices = []
        for col in self.categorical_features:
            boundaries = self._bucket_boundaries[col]
            indices = np.searchsorted(boundaries, X[col].values, side="right")
            bucket_indices.append(indices)
        return torch.LongTensor(np.stack(bucket_indices, axis=1))

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
        user_idx = torch.LongTensor(X["reviewer_id"].values)
        diner_idx = torch.LongTensor(X["diner_idx"].values)

        # Split features into continuous and categorical
        excluded = {"reviewer_id", "diner_idx"}
        if self.use_categorical_embedding:
            excluded.update(self.categorical_features)

        continuous_cols = [col for col in self.features if col not in excluded]
        features = torch.FloatTensor(X[continuous_cols].values)

        if isinstance(y, pd.Series):
            targets = torch.FloatTensor(y.values)
        else:
            targets = torch.FloatTensor(y)

        if self.use_categorical_embedding and self.categorical_features:
            cat_bucket_idx = self._bucketize(X)
            dataset = TensorDataset(
                user_idx, diner_idx, features, cat_bucket_idx, targets
            )
        else:
            dataset = TensorDataset(user_idx, diner_idx, features, targets)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

        return dataloader

    def _unpack_batch(
        self, batch: tuple[Tensor, ...]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        """Unpack a batch from DataLoader and move to device.

        Returns (user_idx, diner_idx, features, cat_idx_or_None, targets).
        """
        if len(batch) == 5:
            user_idx, diner_idx, features, cat_idx, targets = batch
            cat_idx = cat_idx.to(self.device)
        else:
            user_idx, diner_idx, features, targets = batch
            cat_idx = None

        user_idx = user_idx.to(self.device)
        diner_idx = diner_idx.to(self.device)
        features = features.to(self.device)
        targets = targets.to(self.device)

        return user_idx, diner_idx, features, cat_idx, targets

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> DeepRankerModel:
        """
        Train the Deep Ranker model using BPR loss with proper regularization.

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

        # Compute bucket boundaries from training data
        if self.use_categorical_embedding and self.categorical_features:
            self._compute_bucket_boundaries(X_train)
            num_categorical = len(self.categorical_features)
        else:
            num_categorical = 0

        # Count continuous features (exclude IDs and categorical)
        excluded = {"reviewer_id", "diner_idx"}
        if self.use_categorical_embedding:
            excluded.update(self.categorical_features)
        num_features = len([col for col in self.features if col not in excluded])

        # Initialize model
        self.model = DeepRankerModel(
            num_users=self.num_users,
            num_diners=self.num_diners,
            num_features=num_features,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            num_categorical=num_categorical,
            categorical_embedding_dim=self.categorical_embedding_dim,
            categorical_embedding_buckets=self.categorical_embedding_buckets,
        ).to(self.device)

        # Prepare data loaders
        train_loader = self._prepare_dataloader(X_train, y_train, shuffle=True)
        if X_valid is not None and y_valid is not None:
            valid_loader = self._prepare_dataloader(X_valid, y_valid, shuffle=False)
        else:
            valid_loader = None

        # Optimizer with weight decay (L2 regularization)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Learning rate scheduler for stability
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        # Training loop
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.num_boost_round):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0

            for batch in tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_boost_round}",
                leave=False,
            ):
                user_idx, diner_idx, features, cat_idx, targets = self._unpack_batch(
                    batch
                )

                # Forward pass
                scores = self.model(user_idx, diner_idx, features, cat_idx)

                # Calculate BPR loss (pass user_ids for user-wise comparison)
                loss = bpr_loss_sampled(scores, targets, user_ids=user_idx)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
                    for batch in valid_loader:
                        user_idx, diner_idx, features, cat_idx, targets = (
                            self._unpack_batch(batch)
                        )

                        # Forward pass
                        scores = self.model(user_idx, diner_idx, features, cat_idx)

                        # Calculate BPR loss (pass user_ids for user-wise comparison)
                        loss = bpr_loss_sampled(scores, targets, user_ids=user_idx)

                        val_loss += loss.item()
                        val_batches += 1

                avg_val_loss = val_loss / val_batches

                # Update learning rate based on validation loss
                scheduler.step(avg_val_loss)

                # Logging
                if (epoch + 1) % self.verbose_eval == 0:
                    print(
                        f"Epoch {epoch + 1}/{self.num_boost_round} - "
                        f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                    )

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
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

        # Restore best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model (val_loss: {best_val_loss:.4f})")

        return self.model

    def calculate_rank(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Override to pass full DataFrame (reviewer_id and diner_idx are required)."""
        candidates["pred_score"] = self.predict(candidates)
        candidates = candidates.sort_values(
            by=["reviewer_id", "pred_score"], ascending=[True, False]
        )
        return candidates

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
            for batch in tqdm(test_loader):
                user_idx, diner_idx, features, cat_idx, _ = self._unpack_batch(batch)

                scores = self.model(user_idx, diner_idx, features, cat_idx)

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
                "weight_decay": self.weight_decay,
                "features": self.features,
                "use_categorical_embedding": self.use_categorical_embedding,
                "categorical_features": self.categorical_features,
                "categorical_embedding_dim": self.categorical_embedding_dim,
                "categorical_embedding_buckets": self.categorical_embedding_buckets,
                "bucket_boundaries": self._bucket_boundaries,
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
        self.weight_decay = checkpoint.get("weight_decay", 1e-4)
        self.features = checkpoint["features"]

        # Categorical embedding metadata
        self.use_categorical_embedding = checkpoint.get(
            "use_categorical_embedding", False
        )
        self.categorical_features = checkpoint.get("categorical_features", [])
        self.categorical_embedding_dim = checkpoint.get("categorical_embedding_dim", 8)
        self.categorical_embedding_buckets = checkpoint.get(
            "categorical_embedding_buckets", 20
        )
        self._bucket_boundaries = checkpoint.get("bucket_boundaries", None)

        # Count features
        excluded = {"reviewer_id", "diner_idx"}
        if self.use_categorical_embedding:
            excluded.update(self.categorical_features)
            num_categorical = len(self.categorical_features)
        else:
            num_categorical = 0
        num_features = len([col for col in self.features if col not in excluded])

        # Initialize model
        self.model = DeepRankerModel(
            num_users=self.num_users,
            num_diners=self.num_diners,
            num_features=num_features,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            num_categorical=num_categorical,
            categorical_embedding_dim=self.categorical_embedding_dim,
            categorical_embedding_buckets=self.categorical_embedding_buckets,
        ).to(self.device)

        # Load state dict
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"Model loaded from {model_file}")

        return self.model
