"""Trainer for multimodal triplet embedding model.

This module implements the trainer for the multimodal triplet embedding model using
triplet loss for learning restaurant embeddings.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from yamyam_lab.data.multimodal_triplet import (
    MultimodalTripletDataset,
    create_multimodal_triplet_dataloader,
)
from yamyam_lab.engine.base_embedding_trainer import BaseEmbeddingTrainer
from yamyam_lab.loss.infonce import infonce_loss_with_multiple_negatives
from yamyam_lab.loss.triplet import triplet_margin_loss_with_multiple_negatives
from yamyam_lab.model.embedding.multimodal_triplet import Model, MultimodalTripletConfig


class MultimodalTripletTrainer(BaseEmbeddingTrainer):
    """Trainer for multimodal triplet embedding model.

    Extends BaseEmbeddingTrainer for triplet-based training of diner embeddings.
    Uses Recall@10 and MRR for validation with early stopping.

    Config is loaded from config/models/embedding/multimodal_triplet.yaml.
    Args can override specific values (epochs, lr, batch_size, patience).
    """

    def __init__(self, args):
        """Initialize trainer.

        Args:
            args: Parsed command-line arguments.
        """
        super().__init__(args)
        self.dataset: Optional[MultimodalTripletDataset] = None
        self.val_dataset: Optional[MultimodalTripletDataset] = None

    def load_data(self) -> None:
        """Load and prepare multimodal triplet dataset."""
        # Get data paths from config
        data_config = self.config.data
        features_path = data_config.features_path
        pairs_path = data_config.pairs_path
        category_mapping_path = data_config.category_mapping_path
        val_pairs_path = data_config.val_pairs_path

        # Get training config
        training_config = self.config.training
        batch_size = self._get_config("batch_size")
        num_workers = getattr(self.args, "num_workers", 4)
        random_seed = getattr(self.args, "random_seed", 42)

        # Create training dataloader
        self.train_loader, self.dataset = create_multimodal_triplet_dataloader(
            features_path=features_path,
            pairs_path=pairs_path,
            category_mapping_path=category_mapping_path,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            num_hard_negatives=training_config.num_hard_negatives,
            num_nearby_negatives=training_config.num_nearby_negatives,
            num_random_negatives=training_config.num_random_negatives,
            random_seed=random_seed,
        )

        # Create validation dataloader if val_pairs exists
        if os.path.exists(val_pairs_path):
            self.val_loader, self.val_dataset = create_multimodal_triplet_dataloader(
                features_path=features_path,
                pairs_path=val_pairs_path,
                category_mapping_path=category_mapping_path,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                num_hard_negatives=training_config.num_hard_negatives,
                num_nearby_negatives=training_config.num_nearby_negatives,
                num_random_negatives=training_config.num_random_negatives,
                random_seed=random_seed,
            )
        else:
            self.val_loader = None
            self.val_dataset = None
            self.logger.warning(
                f"Validation pairs not found at {val_pairs_path}. "
                "Validation will be skipped."
            )

        # Store data for later use
        self.data = {
            "num_diners": self.dataset.num_diners,
            "num_large_categories": len(set(self.dataset.large_category_ids.numpy())),
            "num_middle_categories": len(set(self.dataset.middle_category_ids.numpy())),
            "num_small_categories": len(set(self.dataset.small_category_ids.numpy())),
            "all_features": self.dataset.get_all_features(),
        }

        self.log_data_statistics()

    def log_data_statistics(self) -> None:
        """Log data statistics after loading."""
        self.logger.info("=" * 50)
        self.logger.info("Multimodal Triplet Embedding Data Statistics")
        self.logger.info("=" * 50)
        self.logger.info(f"Number of diners: {self.data['num_diners']}")
        self.logger.info(
            f"Number of large categories: {self.data['num_large_categories']}"
        )
        self.logger.info(
            f"Number of middle categories: {self.data['num_middle_categories']}"
        )
        self.logger.info(
            f"Number of small categories: {self.data['num_small_categories']}"
        )
        self.logger.info(f"Number of training pairs: {len(self.dataset)}")
        if self.val_dataset:
            self.logger.info(f"Number of validation pairs: {len(self.val_dataset)}")
        self.logger.info("=" * 50)

    def build_model(self) -> None:
        """Build multimodal triplet embedding model."""
        top_k_values = self.get_top_k_values()
        model_config = self.config.model

        # Create model config
        self.model_config = MultimodalTripletConfig(
            num_large_categories=self.data["num_large_categories"],
            num_middle_categories=self.data["num_middle_categories"],
            num_small_categories=self.data["num_small_categories"],
            embedding_dim=model_config.embedding_dim,
            category_dim=model_config.category_dim,
            menu_dim=model_config.menu_dim,
            diner_name_dim=model_config.diner_name_dim,
            price_dim=model_config.price_dim,
            review_text_dim=getattr(model_config, "review_text_dim", 0),
            num_attention_heads=model_config.num_attention_heads,
            dropout=model_config.dropout,
            kobert_model_name=model_config.kobert_model_name,
            use_precomputed_menu_embeddings=model_config.use_precomputed_menu_embeddings,
            use_precomputed_name_embeddings=model_config.use_precomputed_name_embeddings,
            use_precomputed_review_text_embeddings=getattr(
                model_config, "use_precomputed_review_text_embeddings", True
            ),
            device=self.args.device,
            top_k_values=top_k_values,
            diner_ids=torch.arange(self.data["num_diners"]),
            recommend_batch_size=self.config.training.evaluation.recommend_batch_size,
        )

        # Create model
        self.model = Model(config=self.model_config).to(self.args.device)

        self.logger.info(f"Model created with {self._count_parameters()} parameters")

    def _get_features_by_indices(
        self, all_features: Dict[str, Tensor], indices: Tensor
    ) -> Dict[str, Tensor]:
        """Extract features for given indices.

        Args:
            all_features: Dictionary of all feature tensors.
            indices: Tensor of indices to extract.

        Returns:
            Dictionary of feature tensors for the given indices.
        """
        features = {
            "large_category_ids": all_features["large_category_ids"][indices],
            "middle_category_ids": all_features["middle_category_ids"][indices],
            "small_category_ids": all_features["small_category_ids"][indices],
            "menu_embeddings": all_features["menu_embeddings"][indices],
            "diner_name_embeddings": all_features["diner_name_embeddings"][indices],
            "price_features": all_features["price_features"][indices],
        }
        if "review_text_embeddings" in all_features:
            features["review_text_embeddings"] = all_features["review_text_embeddings"][
                indices
            ]
        return features

    def _create_test_dataloader(self):
        """Create dataloader for test set.

        Returns:
            Tuple of (dataloader, dataset) for test set.
        """
        data_config = self.config.data
        return create_multimodal_triplet_dataloader(
            features_path=data_config.features_path,
            pairs_path=data_config.test_pairs_path,
            category_mapping_path=data_config.category_mapping_path,
            batch_size=self._get_config("batch_size"),
            shuffle=False,
            num_workers=1,
        )

    def train_loop(self) -> None:
        """Training loop with early stopping based on Recall@10 and MRR."""
        training_config = self.config.training

        # Get hyperparameters (args override config)
        lr = self._get_config("lr")
        epochs = self._get_config("epochs")
        patience = self._get_config("patience")

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=training_config.weight_decay,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )

        # Get loss hyperparameters from config
        loss_type = self._get_config("loss_type") or "triplet"
        margin = training_config.margin
        temperature = getattr(training_config, "temperature", 0.07)
        category_weight = training_config.category_weight
        gradient_clip = training_config.gradient_clip

        self.logger.info(f"Using loss function: {loss_type}")

        # Move all features to device
        all_features = {
            k: v.to(self.args.device) for k, v in self.data["all_features"].items()
        }

        def train_one_epoch(epoch: int, optimizer, all_features) -> float:
            """Train one epoch and return average loss."""
            self.model.train()
            total_loss = 0.0
            num_batches = len(self.train_loader)

            for batch_idx, batch in enumerate(self.train_loader):
                optimizer.zero_grad()

                # Get batch indices
                anchor_indices = batch["anchor_indices"].to(self.args.device)
                positive_indices = batch["positive_indices"].to(self.args.device)
                negative_indices = batch["negative_indices"].to(self.args.device)
                anchor_categories = batch["anchor_categories"].to(self.args.device)
                positive_categories = batch["positive_categories"].to(self.args.device)
                negative_categories = batch["negative_categories"].to(self.args.device)

                # Get features for anchors, positives, and negatives
                anchor_features = self._get_features_by_indices(
                    all_features, anchor_indices
                )
                positive_features = self._get_features_by_indices(
                    all_features, positive_indices
                )

                # Compute anchor and positive embeddings
                anchor_emb = self.model(anchor_features)
                positive_emb = self.model(positive_features)

                # Batch all negatives into a single forward pass
                batch_size = negative_indices.size(0)
                num_negatives = negative_indices.size(1)
                flat_negative_indices = negative_indices.view(-1)

                # Get features for all negatives at once
                flat_negative_features = self._get_features_by_indices(
                    all_features, flat_negative_indices
                )

                # Single forward pass for all negatives
                flat_negative_emb = self.model(flat_negative_features)

                # Reshape to (B, num_negatives, embedding_dim)
                embedding_dim = flat_negative_emb.size(-1)
                negative_emb = flat_negative_emb.view(
                    batch_size, num_negatives, embedding_dim
                )

                # Compute loss based on selected loss type
                if loss_type == "infonce":
                    batch_loss = infonce_loss_with_multiple_negatives(
                        anchor=anchor_emb,
                        positive=positive_emb,
                        negatives=negative_emb,
                        anchor_category=anchor_categories,
                        positive_category=positive_categories,
                        negative_categories=negative_categories,
                        temperature=temperature,
                        category_weight=category_weight,
                    )
                else:
                    batch_loss = triplet_margin_loss_with_multiple_negatives(
                        anchor=anchor_emb,
                        positive=positive_emb,
                        negatives=negative_emb,
                        anchor_category=anchor_categories,
                        positive_category=positive_categories,
                        negative_categories=negative_categories,
                        margin=margin,
                        category_weight=category_weight,
                    )

                # Backward pass
                batch_loss.backward()

                # Gradient clipping
                if gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

                optimizer.step()

                total_loss += batch_loss.item()

                if batch_idx % 100 == 0:
                    self.logger.info(
                        f"Batch {batch_idx}/{num_batches}, "
                        f"Loss: {batch_loss.item():.4f}"
                    )

            return total_loss / num_batches

        # Run training with early stopping
        self._run_early_stopping_loop(
            epochs=epochs,
            patience=patience,
            optimizer=optimizer,
            scheduler=scheduler,
            all_features=all_features,
            train_epoch_fn=train_one_epoch,
        )
