from __future__ import annotations

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from data.dataset import DataConfig, DatasetLoader
from evaluation.metric_calculator.ranker_metric_calculator import RankerMetricCalculator


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def main(cfg: DictConfig):
    # load dataset
    data_loader = DatasetLoader(data_config=DataConfig(**cfg.data))
    data = data_loader.prepare_train_val_dataset(is_rank=True)

    # mapping reverse
    X_train, y_train, X_valid, y_valid, X_test_cold_users, X_test_warm_users = (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test_cold_start_user"],
        data["X_test_warm_start_user"],
    )

    # build model
    trainer = instantiate(
        cfg.models, recommend_batch_size=cfg.training.evaluation.recommend_batch_size
    )

    # train model
    trainer.fit(X_train, y_train, X_valid, y_valid)

    # save model
    trainer.save_model()

    trainer.plot_feature_importance()

    # train liked items
    X_train["target"] = y_train
    train_liked_items = (
        X_train[X_train["target"] == 1]
        .groupby("reviewer_id")["diner_idx"]
        .apply(np.array)
    )

    # candidate predictions
    candidates = trainer.calculate_rank(data["candidates"])

    # metric calculator
    metric_calculator = RankerMetricCalculator(
        top_k_values=cfg.training.evaluation.top_k_values_for_pred,
        model=trainer,
        features=cfg.models.features,
        recommend_batch_size=cfg.training.evaluation.recommend_batch_size,
        filter_already_liked=True,
    )

    metric_dict = metric_calculator.generate_recommendations_and_calculate_metric(
        X_train=X_train,
        X_val_warm_users=X_test_warm_users,
        X_val_cold_users=X_test_cold_users,
        most_popular_diner_ids=data["most_popular_diner_ids"],
        candidates=candidates,
        train_liked_series=train_liked_items,
    )

    # for each user type, the metric is not yet averaged but summed, so calculate mean
    for user_type, metric in metric_dict.items():
        metric_calculator.calculate_mean_metric(metric)

    # for each user type, report map, ndcg, recall
    metric_calculator.report_metric_with_warm_cold_all_users(
        metric_dict=metric_dict, data_type="test"
    )


if __name__ == "__main__":
    main()
