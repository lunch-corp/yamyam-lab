"""Graph-based model Trainer implementation."""

import copy
import importlib
import os
import pickle

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from yamyam_lab.data.config import DataConfig
from yamyam_lab.data.graph import GraphDatasetLoader
from yamyam_lab.engine.base_trainer import BaseTrainer
from yamyam_lab.evaluation.metric_calculator import EmbeddingMetricCalculator
from yamyam_lab.tools.plot import plot_metric_at_k


class GraphTrainer(BaseTrainer):
    """Trainer for graph-based models (Node2Vec, GraphSAGE, etc.)."""

    def load_data(self) -> None:
        """Load graph dataset."""
        # Set multiprocessing start method to spawn
        mp.set_start_method("spawn", force=True)

        fe = self.config.preprocess.feature_engineering

        data_loader = GraphDatasetLoader(
            data_config=DataConfig(
                X_columns=["diner_idx", "reviewer_id"],
                y_columns=["reviewer_review_score"],
                category_column_for_meta=self.args.category_column_for_meta,
                user_engineered_feature_names=fe.user_engineered_feature_names,
                diner_engineered_feature_names=fe.diner_engineered_feature_names,
                is_timeseries_by_time_point=self.config.preprocess.data.is_timeseries_by_time_point,
                train_time_point=self.config.preprocess.data.train_time_point,
                val_time_point=self.config.preprocess.data.val_time_point,
                test_time_point=self.config.preprocess.data.test_time_point,
                end_time_point=self.config.preprocess.data.end_time_point,
                use_unique_mapping_id=True,
                test=getattr(self.args, "test", False),
                config_root_path=self.args.config_root_path,
            ),
        )
        self.data = data_loader.prepare_graph_dataset(
            is_networkx_graph=True,
            use_metadata=self.args.use_metadata,
            weighted_edge=self.args.weighted_edge,
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
        """Build graph-based model."""
        num_nodes = self.data["num_users"] + self.data["num_diners"]
        if self.args.model == "metapath2vec":
            num_nodes += self.data["num_metas"]

        top_k_values = self.get_top_k_values()

        # Import embedding module
        model_path = f"yamyam_lab.model.graph.{self.args.model}"
        model_module = importlib.import_module(model_path).Model

        self.model = model_module(
            user_ids=torch.tensor(list(self.data["user_mapping"].values())).to(
                self.args.device
            ),
            diner_ids=torch.tensor(list(self.data["diner_mapping"].values())).to(
                self.args.device
            ),
            graph=self.data["train_graph"],
            embedding_dim=self.args.embedding_dim,
            walk_length=self.args.walk_length,
            walks_per_node=self.args.walks_per_node,
            num_nodes=num_nodes,
            num_negative_samples=self.args.num_negative_samples,
            q=self.args.q,
            p=self.args.p,
            top_k_values=top_k_values,
            model_name=self.args.model,
            device=self.args.device,
            recommend_batch_size=self.config.training.evaluation.recommend_batch_size,
            num_workers=4,
            meta_path=getattr(self.args, "meta_path", None),
            num_sage_layers=getattr(self.args, "num_sage_layers", None),
            aggregator_funcs=getattr(self.args, "aggregator_funcs", None),
            num_neighbor_samples=getattr(self.args, "num_neighbor_samples", None),
            user_raw_features=self.data["user_feature"].to(self.args.device),
            diner_raw_features=self.data["diner_feature"].to(self.args.device),
            num_layers=getattr(self.args, "num_lightgcn_layers", None),
            drop_ratio=getattr(self.args, "drop_ratio", None),
        ).to(self.args.device)

    def build_metric_calculator(self) -> None:
        """Build embedding metric calculator."""
        top_k_values = self.get_top_k_values()

        all_embeds = (
            self.model._embedding
            if self.args.model in ["graphsage", "lightgcn"]
            else self.model._embedding.weight
        )

        self.metric_calculator = EmbeddingMetricCalculator(
            diner_ids=list(self.data["diner_mapping"].values()),
            top_k_values=top_k_values,
            all_embeds=all_embeds,
            filter_already_liked=True,
            recommend_batch_size=self.config.training.evaluation.recommend_batch_size,
            device=self.args.device,
            logger=self.logger,
        )

    def train_loop(self) -> None:
        """Training loop with early stopping."""
        optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.args.lr)

        loader = self.model.loader(
            batch_size=self.args.batch_size,
            shuffle=True,
        )

        best_val_ndcg = -float("inf")
        best_val_ndcg_epoch = -1
        patience = self.args.patience
        best_model_weights = None

        for epoch in range(self.args.epochs):
            self.logger.info(f"################## epoch {epoch} ##################")

            # Training
            total_loss = 0
            batch_len = len(loader)
            for batch_idx, (pos_rw, neg_rw) in enumerate(loader):
                optimizer.zero_grad()
                loss = self.model.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if batch_idx % 500 == 0:
                    self.logger.info(
                        f"current batch index: {batch_idx} out of {batch_len}"
                    )

            # when training graphsage or lightgcn for every epoch,
            # propagation should be run to store embeddings for each node
            if self.args.model in ["graphsage", "lightgcn"]:
                for batch_nodes in DataLoader(
                    torch.tensor([node for node in self.data["train_graph"].nodes()]),
                    batch_size=self.args.batch_size,
                    shuffle=True,
                ):
                    self.model.propagate_and_store_embedding(
                        batch_nodes.to(self.args.device)
                    )

            total_loss /= len(loader)
            self.model.tr_loss.append(total_loss)
            self.logger.info(f"epoch {epoch}: train loss {total_loss:.4f}")

            # Validation evaluation
            self._evaluate_epoch(epoch)

            # Early stopping logic
            val_ndcg = self.model.metric_at_k_total_epochs[3]["ndcg"][-1]

            if val_ndcg == 0:
                self.logger.info(
                    "Validation ndcg@3 is still ZERO... Going to train again..."
                )
                continue

            if best_val_ndcg < val_ndcg:
                best_val_ndcg_epoch = epoch
                best_val_ndcg = round(val_ndcg, 6)
                best_model_weights = copy.deepcopy(self.model.state_dict())
                patience = self.args.patience

                # Save training results when validation improves
                self._save_train_results_at_current_epoch(save_all_results=True)
                self.logger.info(
                    f"Best validation ndcg@3: {best_val_ndcg} at epoch {best_val_ndcg_epoch}"
                )
            else:
                patience -= 1
                self.logger.info(
                    f"Validation ndcg@3 did not decrease. Patience {patience} left."
                )
                if patience == 0:
                    self.logger.info(
                        f"Patience over. Early stopping at epoch {epoch} with {best_val_ndcg} validation ndcg@3"
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
