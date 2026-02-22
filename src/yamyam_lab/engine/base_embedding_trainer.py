"""Base trainer class for embedding models.

This module provides a base trainer for embedding models that use
similarity-based evaluation (Recall@K, MRR) and triplet-style training.
"""

import copy
import os
import pickle
from abc import abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from yamyam_lab.engine.base_trainer import BaseTrainer
from yamyam_lab.tools.plot import plot_diner_embedding_metrics


class BaseEmbeddingTrainer(BaseTrainer):
    """Base trainer for embedding models.

    Provides common functionality for embedding model training:
    - Config value retrieval with args override
    - Recall@K and MRR evaluation
    - Checkpoint saving
    - Early stopping based on validation metrics
    - Training metrics plotting

    Subclasses must implement:
    - load_data(): Load dataset specific to the model
    - build_model(): Build the embedding model
    - train_loop(): Model-specific training logic
    - _get_features_by_indices(): Extract features for given indices
    """

    def __init__(self, args):
        """Initialize trainer.

        Args:
            args: Parsed command-line arguments.
        """
        super().__init__(args)
        self.dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.model_config = None

        # Validation metrics history for plotting
        self.val_metrics_history: Dict[str, list] = {
            "recall@1": [],
            "recall@5": [],
            "recall@10": [],
            "recall@20": [],
            "mrr": [],
        }

        # K values for evaluation
        self.k_values: List[int] = [1, 5, 10, 20]

    def _get_config(self, key: str, section: str = "training") -> Any:
        """Get config value, with args override if provided.

        Args:
            key: Config key name.
            section: Config section ('model', 'training', 'data').

        Returns:
            Value from args if set, otherwise from config.
        """
        # Check if args has a non-None override
        args_value = getattr(self.args, key, None)
        if args_value is not None:
            return args_value

        # Get from config
        config_section = getattr(self.config, section, {})
        return getattr(config_section, key, None)

    def _count_parameters(self) -> int:
        """Count trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def build_metric_calculator(self) -> None:
        """Build metric calculator for embedding model.

        Uses custom metrics: Recall@K, MRR.
        Metrics are computed inline in _evaluate_epoch.
        """
        pass

    @abstractmethod
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
        raise NotImplementedError

    def _evaluate_epoch(self, epoch: int) -> Dict[str, float]:
        """Evaluate model on validation set.

        Computes Recall@K and MRR metrics.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of metric values.
        """
        self.model.eval()
        metrics = {}

        # Get all embeddings
        all_embeddings = self.model._embedding

        # Compute metrics on validation pairs
        val_pairs = self.val_dataset.pairs_df
        diner_idx_to_pos = self.val_dataset.diner_idx_to_position

        # Sample validation queries (anchor -> positive)
        num_samples = min(1000, len(val_pairs))
        sample_pairs = val_pairs.sample(n=num_samples, random_state=42)

        # Compute Recall@K and MRR
        recalls = {k: [] for k in self.k_values}
        reciprocal_ranks = []

        with torch.no_grad():
            for _, row in sample_pairs.iterrows():
                anchor_diner_idx = int(row["anchor_idx"])
                positive_diner_idx = int(row["positive_idx"])

                # Convert to position indices
                anchor_pos = diner_idx_to_pos.get(anchor_diner_idx)
                positive_pos = diner_idx_to_pos.get(positive_diner_idx)

                # Skip if either diner not in features
                if anchor_pos is None or positive_pos is None:
                    continue

                # Get anchor embedding
                anchor_emb = all_embeddings[anchor_pos : anchor_pos + 1]

                # Compute similarities with all diners
                similarities = self.model.similarity(anchor_emb, all_embeddings)
                similarities = similarities.squeeze(0)

                # Exclude anchor itself
                similarities[anchor_pos] = -float("inf")

                # Get ranked indices
                _, ranked_indices = torch.sort(similarities, descending=True)
                ranked_indices = ranked_indices.cpu().numpy()

                # Find position of positive
                pos_rank = np.where(ranked_indices == positive_pos)[0]
                if len(pos_rank) > 0:
                    rank = pos_rank[0] + 1  # 1-indexed

                    # Recall@K
                    for k in self.k_values:
                        recalls[k].append(1.0 if rank <= k else 0.0)

                    # Reciprocal rank
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    for k in self.k_values:
                        recalls[k].append(0.0)
                    reciprocal_ranks.append(0.0)

        # Compute mean metrics
        for k in self.k_values:
            metrics[f"recall@{k}"] = np.mean(recalls[k])

        metrics["mrr"] = np.mean(reciprocal_ranks)

        # Log metrics
        self.logger.info(f"Validation metrics at epoch {epoch}:")
        for name, value in metrics.items():
            self.logger.info(f"  {name}: {value:.4f}")

        self.model.train()
        return metrics

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric values.
        """
        file_name = self.config.post_training.file_name

        # Save model weights and config
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.model_config,
                "epoch": epoch,
                "metrics": metrics,
            },
            os.path.join(self.result_path, file_name.weight),
        )

        # Save training loss history
        pickle.dump(
            self.model.tr_loss,
            open(os.path.join(self.result_path, file_name.training_loss), "wb"),
        )

    def _run_early_stopping_loop(
        self,
        epochs: int,
        patience: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        all_features: Dict[str, Tensor],
        train_epoch_fn,
    ) -> None:
        """Run training loop with early stopping.

        Args:
            epochs: Maximum number of epochs.
            patience: Early stopping patience.
            optimizer: Optimizer instance.
            scheduler: Learning rate scheduler (optional).
            all_features: All features moved to device.
            train_epoch_fn: Function to train one epoch, returns average loss.
        """
        early_stop_metric = self._get_config("early_stop_metric") or "recall@10"
        self.logger.info(f"Early stopping metric: {early_stop_metric}")

        best_val_metric = -float("inf")
        best_val_epoch = -1
        best_model_weights = None
        patience_counter = patience

        for epoch in range(epochs):
            self.logger.info(f"################## Epoch {epoch} ##################")

            # Train one epoch
            avg_loss = train_epoch_fn(epoch, optimizer, all_features)
            self.model.tr_loss.append(avg_loss)
            self.logger.info(f"Epoch {epoch}: Average training loss: {avg_loss:.4f}")

            # Compute and store embeddings for evaluation
            self.model.compute_and_store_embeddings(
                all_features=all_features,
                batch_size=self._get_config("batch_size"),
            )

            # Validation
            if self.val_loader is not None:
                val_metrics = self._evaluate_epoch(epoch)
                val_metric = val_metrics.get(early_stop_metric, 0.0)

                # Track metrics history for plotting
                for metric_name, value in val_metrics.items():
                    if metric_name in self.val_metrics_history:
                        self.val_metrics_history[metric_name].append(value)

                # Update learning rate scheduler
                if scheduler is not None:
                    scheduler.step(val_metric)

                # Early stopping
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_val_epoch = epoch
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                    patience_counter = patience

                    # Save checkpoint
                    self._save_checkpoint(epoch, val_metrics)
                    self.logger.info(
                        f"New best {early_stop_metric}: {best_val_metric:.4f} "
                        f"at epoch {epoch}"
                    )
                else:
                    patience_counter -= 1
                    self.logger.info(
                        f"{early_stop_metric} did not improve. "
                        f"Patience: {patience_counter}/{patience}"
                    )

                    if patience_counter <= 0:
                        self.logger.info(
                            f"Early stopping at epoch {epoch}. "
                            f"Best {early_stop_metric}: {best_val_metric:.4f} "
                            f"at epoch {best_val_epoch}"
                        )
                        break

        # Load best model weights
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
            self.logger.info("Loaded best model weights")
            self._save_checkpoint(best_val_epoch, {"best": True})

    def evaluate_validation(self) -> None:
        """Evaluate on validation set. Called after training."""
        if self.val_loader is not None:
            self.logger.info("Final validation evaluation:")
            self._evaluate_epoch(epoch=-1)

    @abstractmethod
    def _create_test_dataloader(self):
        """Create dataloader for test set.

        Returns:
            Tuple of (dataloader, dataset) for test set.
        """
        raise NotImplementedError

    def evaluate_test(self) -> None:
        """Evaluate on test set."""
        data_config = self.config.data
        test_pairs_path = data_config.test_pairs_path

        if not os.path.exists(test_pairs_path):
            self.logger.warning(
                f"Test pairs not found at {test_pairs_path}. Test evaluation skipped."
            )
            return

        # Temporarily swap validation dataset for test
        original_val_dataset = self.val_dataset

        _, self.val_dataset = self._create_test_dataloader()

        self.logger.info("=" * 50)
        self.logger.info("Test Set Evaluation")
        self.logger.info("=" * 50)
        test_metrics = self._evaluate_epoch(epoch=-1)

        # Restore original validation dataset
        self.val_dataset = original_val_dataset

        return test_metrics

    def post_process(self) -> None:
        """Post-processing after training."""
        # Plot training metrics
        self.logger.info("Generating training plots...")
        plot_diner_embedding_metrics(
            tr_loss=self.model.tr_loss,
            val_metrics_history=self.val_metrics_history,
            parent_save_path=self.result_path,
        )
        self.logger.info(f"Saved plots to {self.result_path}")

        # Save validation metrics history
        pickle.dump(
            self.val_metrics_history,
            open(os.path.join(self.result_path, "val_metrics_history.pkl"), "wb"),
        )

        # Generate candidate similarities for all diners
        if getattr(self.args, "save_candidate", False):
            self.logger.info("Generating candidate similarities...")
            top_k = self.config.post_training.candidate_generation.top_k
            candidates_df = self.model.generate_candidates_for_each_diner(top_k)
            candidates_df.to_parquet(
                os.path.join(
                    self.result_path, self.config.post_training.file_name.candidate
                ),
                index=False,
            )
            self.logger.info(f"Saved {len(candidates_df)} candidate pairs")
