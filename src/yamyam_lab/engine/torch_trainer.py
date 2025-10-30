"""PyTorch-based model Trainer implementation."""

import copy
import importlib
import os
import pickle

import torch
from torch import optim

from yamyam_lab.data.config import DataConfig
from yamyam_lab.data.mf import MFDatasetLoader
from yamyam_lab.engine.base_trainer import BaseTrainer
from yamyam_lab.evaluation.metric_calculator import SVDBiasMetricCalculator
from yamyam_lab.loss.custom import svd_loss
from yamyam_lab.tools.plot import plot_metric_at_k


class TorchTrainer(BaseTrainer):
    """Trainer for PyTorch-based models (SVD with bias, etc.)."""

    def load_data(self) -> None:
        """Load MF dataset."""
        data_loader = MFDatasetLoader(
            data_config=DataConfig(
                X_columns=["diner_idx", "reviewer_id"],
                y_columns=["reviewer_review_score"],
                is_timeseries_by_time_point=self.config.preprocess.data.is_timeseries_by_time_point,
                train_time_point=self.config.preprocess.data.train_time_point,
                val_time_point=self.config.preprocess.data.val_time_point,
                test_time_point=self.config.preprocess.data.test_time_point,
                end_time_point=self.config.preprocess.data.end_time_point,
                test=getattr(self.args, "test", False),
                config_root_path=self.args.config_root_path,
            ),
        )
        self.data = data_loader.prepare_mf_dataset(
            is_tensor=True,
            filter_config=self.preprocess_config.filter,
        )

        # Save data object
        file_name = self.config.post_training.file_name
        pickle.dump(
            self.data, open(os.path.join(self.result_path, file_name.data_object), "wb")
        )

        # Log data statistics after loading
        self.log_data_statistics()

    def build_model(self) -> None:
        """Build PyTorch model."""
        top_k_values = self.get_top_k_values()

        # Import model module
        model_path = f"yamyam_lab.model.mf.{self.args.model}"
        model_module = importlib.import_module(model_path).Model

        self.model = model_module(
            user_ids=torch.tensor(list(self.data["user_mapping"].values())).to(
                self.args.device
            ),
            diner_ids=torch.tensor(list(self.data["diner_mapping"].values())).to(
                self.args.device
            ),
            embedding_dim=self.args.embedding_dim,
            top_k_values=top_k_values,
            model_name=self.args.model,
            mu=torch.tensor(self.data["y_train"].mean(), dtype=torch.float32),
        ).to(self.args.device)

    def build_metric_calculator(self) -> None:
        """Build SVD bias metric calculator."""
        top_k_values = self.get_top_k_values()

        self.metric_calculator = SVDBiasMetricCalculator(
            diner_ids=list(self.data["diner_mapping"].values()),
            model=self.model,
            top_k_values=top_k_values,
            filter_already_liked=True,
            recommend_batch_size=self.config.training.evaluation.recommend_batch_size,
            logger=self.logger,
            embed_user=self.model.embed_user,
            embed_item=self.model.embed_item,
            user_bias=self.model.user_bias,
            item_bias=self.model.item_bias,
        )

    def train_loop(self) -> None:
        """Training loop with validation and early stopping."""
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)

        train_dataloader = self.data["train_dataloader"]
        val_dataloader = self.data["val_dataloader"]

        best_loss = float("inf")
        patience = self.args.patience
        best_model_weights = None

        for epoch in range(self.args.epochs):
            self.logger.info(f"################## epoch {epoch} ##################")

            # Training
            self.model.train()
            tr_loss = 0.0
            for X_train, y_train in train_dataloader:
                diners, users = X_train[:, 0], X_train[:, 1]
                optimizer.zero_grad()
                y_pred = self.model(users, diners)
                loss = svd_loss(
                    pred=y_pred,
                    true=y_train,
                    params=[param.data for param in self.model.parameters()],
                    regularization=self.args.regularization,
                    user_idx=users,
                    diner_idx=diners,
                    num_users=self.data["num_users"],
                    num_diners=self.data["num_diners"],
                )
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()

            tr_loss = round(tr_loss / len(train_dataloader), 6)
            self.model.tr_loss.append(tr_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for X_val, y_val in val_dataloader:
                    diners, users = X_val[:, 0], X_val[:, 1]
                    y_pred = self.model(users, diners)
                    loss = svd_loss(
                        pred=y_pred,
                        true=y_val,
                        params=[param.data for param in self.model.parameters()],
                        regularization=self.args.regularization,
                        user_idx=users,
                        diner_idx=diners,
                        num_users=self.data["num_users"],
                        num_diners=self.data["num_diners"],
                    )
                    val_loss += loss.item()
                val_loss = round(val_loss / len(val_dataloader), 6)

            self.logger.info(f"Train Loss: {tr_loss}")
            self.logger.info(f"Validation Loss: {val_loss}")

            # Evaluate validation metrics
            self._evaluate_epoch(epoch)

            # Save model at each epoch
            self._save_train_results_at_current_epoch(epoch)

            # Early stopping logic
            if best_loss > val_loss:
                prev_best_loss = best_loss
                best_loss = val_loss
                best_model_weights = copy.deepcopy(self.model.state_dict())
                patience = self.args.patience
                self.logger.info(
                    f"Best validation: {best_loss}, Previous validation loss: {prev_best_loss}"
                )
            else:
                patience -= 1
                self.logger.info(
                    f"Validation loss did not decrease. Patience {patience} left."
                )
                if patience == 0:
                    self.logger.info(
                        f"Patience over. Early stopping at epoch {epoch} with {best_loss} validation loss"
                    )
                    break

        # Load and save only the best model weights
        if best_model_weights:
            self.model.load_state_dict(best_model_weights)
            self.logger.info("Load weight with best validation ndcg@3")
            # Only save weights, not training results
            self._save_train_results_at_current_epoch(save_all_results=False)
            self.logger.info("Save final model with best validation ndcg@3")

    def _evaluate_epoch(self, epoch: int) -> None:
        """Evaluate at current epoch."""
        metric_dict = (
            self.metric_calculator.generate_recommendations_and_calculate_metric(
                X_train=self.data["X_train"],
                X_val_warm_users=self.data["X_val_warm_users"],
                X_val_cold_users=self.data["X_val_cold_users"],
                most_popular_diner_ids=self.data["most_popular_diner_ids"],
                filter_already_liked=True,
            )
        )

        for user_type, metric in metric_dict.items():
            self.metric_calculator.calculate_mean_metric(metric)

        self.logger.info(
            f"################## Validation data metric report for {epoch} epoch ##################"
        )
        self.metric_calculator.report_metric_with_warm_cold_all_users(
            metric_dict=metric_dict, data_type="val"
        )

        self.metric_calculator.save_metric_at_current_epoch(
            metric_at_k=metric_dict["all"],
            metric_at_k_total_epochs=self.model.metric_at_k_total_epochs,
        )

    def _save_train_results_at_current_epoch(
        self, save_all_results: bool = True
    ) -> None:
        """Save model at current epoch.

        Args:
            save_all_results: If True, save weights, losses, and metrics.
                             If False, save only weights.
        """
        file_name = self.config.post_training.file_name

        torch.save(
            self.model.state_dict(),
            str(os.path.join(self.result_path, file_name.weight)),
        )

        # Only save training results if specified
        # During patience epochs, we don't save these
        if save_all_results:
            pickle.dump(
                self.model.tr_loss,
                open(os.path.join(self.result_path, file_name.training_loss), "wb"),
            )
            pickle.dump(
                self.model.metric_at_k_total_epochs,
                open(os.path.join(self.result_path, file_name.metric), "wb"),
            )

    def evaluate_validation(self) -> None:
        """Already handled in train_loop."""
        pass

    def evaluate_test(self) -> None:
        """Evaluate on test set."""
        metric_dict = (
            self.metric_calculator.generate_recommendations_and_calculate_metric(
                X_train=self.data["X_train"],
                X_val_warm_users=self.data["X_test_warm_users"],
                X_val_cold_users=self.data["X_test_cold_users"],
                most_popular_diner_ids=self.data["most_popular_diner_ids"],
                filter_already_liked=True,
            )
        )

        for user_type, metric in metric_dict.items():
            self.metric_calculator.calculate_mean_metric(metric)

        self.logger.info(
            "################################ Test data metric report ################################"
        )
        self.metric_calculator.report_metric_with_warm_cold_all_users(
            metric_dict=metric_dict, data_type="test"
        )

    def post_process(self) -> None:
        """Plot metrics after training."""
        top_k_values_for_pred = self.config.training.evaluation.top_k_values_for_pred
        top_k_values_for_candidate = (
            self.config.training.evaluation.top_k_values_for_candidate
        )

        plot_metric_at_k(
            metric=self.model.metric_at_k_total_epochs,
            tr_loss=self.model.tr_loss,
            parent_save_path=self.result_path,
            top_k_values_for_pred=top_k_values_for_pred,
            top_k_values_for_candidate=top_k_values_for_candidate,
        )
