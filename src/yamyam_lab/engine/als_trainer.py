"""ALS Trainer implementation."""

from yamyam_lab.data.config import DataConfig
from yamyam_lab.data.csr import CsrDatasetLoader
from yamyam_lab.engine.base_trainer import BaseTrainer
from yamyam_lab.evaluation.metric_calculator import ALSMetricCalculator
from yamyam_lab.model import ALS


class ALSTrainer(BaseTrainer):
    """Trainer for ALS model."""

    def load_data(self) -> None:
        """Load CSR dataset and log statistics."""
        fe = self.config.preprocess.feature_engineering

        data_loader = CsrDatasetLoader(
            data_config=DataConfig(
                X_columns=["diner_idx", "reviewer_id"],
                y_columns=["reviewer_review_score"],
                user_engineered_feature_names=fe.user_engineered_feature_names,
                diner_engineered_feature_names=fe.diner_engineered_feature_names,
                is_timeseries_by_time_point=self.config.preprocess.data.is_timeseries_by_time_point,
                train_time_point=self.config.preprocess.data.train_time_point,
                val_time_point=self.config.preprocess.data.val_time_point,
                test_time_point=self.config.preprocess.data.test_time_point,
                end_time_point=self.config.preprocess.data.end_time_point,
                test=getattr(self.args, "test", False),
                config_root_path=self.args.config_root_path,
            ),
        )
        self.data = data_loader.prepare_csr_dataset(
            is_csr=True,
            filter_config=self.preprocess_config.filter,
        )

        # Log data statistics after loading
        self.log_data_statistics()

    def build_model(self) -> None:
        """Build ALS model."""
        self.model = ALS(
            alpha=self.args.alpha,
            factors=self.args.factors,
            regularization=self.args.regularization,
            iterations=self.args.iterations,
            use_gpu=self.args.use_gpu,
            diner_mapping=self.data["diner_mapping"],
            calculate_training_loss=self.args.calculate_training_loss,
        )

    def build_metric_calculator(self) -> None:
        """Build ALS metric calculator."""
        top_k_values = self.get_top_k_values()

        self.metric_calculator = ALSMetricCalculator(
            diner_ids=list(self.data["diner_mapping"].values()),
            model=self.model,
            top_k_values=top_k_values,
            filter_already_liked=True,
            recommend_batch_size=self.config.training.evaluation.recommend_batch_size,
            logger=self.logger,
        )

    def train_loop(self) -> None:
        """Train ALS model."""
        self.model.fit(self.data["X_train"])

    def evaluate_validation(self) -> None:
        """
        Evaluate on validation set.

        Calculate metric for **validation data** with warm / cold / all users separately.
        Note that, we should calculate this metric for each iteration while training als,
        but we could not find any methods to integrate it into implicit library,
        so, we report validation metric after finishing training als.
        """
        metric_dict = (
            self.metric_calculator.generate_recommendations_and_calculate_metric(
                X_train=self.data["X_train_df"],
                X_val_warm_users=self.data["X_val_warm_users"],
                X_val_cold_users=self.data["X_val_cold_users"],
                most_popular_diner_ids=self.data["most_popular_diner_ids"],
                filter_already_liked=True,
                train_csr=self.data["X_train"],
            )
        )

        for user_type, metric in metric_dict.items():
            self.metric_calculator.calculate_mean_metric(metric)

        self.logger.info(
            "################################ Validation data metric report ################################"
        )
        self.metric_calculator.report_metric_with_warm_cold_all_users(
            metric_dict=metric_dict, data_type="val"
        )

    def evaluate_test(self) -> None:
        """Evaluate on test set."""
        metric_dict = (
            self.metric_calculator.generate_recommendations_and_calculate_metric(
                X_train=self.data["X_train_df"],
                X_val_warm_users=self.data["X_test_warm_users"],
                X_val_cold_users=self.data["X_test_cold_users"],
                most_popular_diner_ids=self.data["most_popular_diner_ids"],
                filter_already_liked=True,
                train_csr=self.data["X_train"],
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
