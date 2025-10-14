from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import mlflow.lightgbm
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from yamyam_lab.data.config import DataConfig
from yamyam_lab.data.ranker import RankerDatasetLoader
from yamyam_lab.evaluation.metric_calculator.ranker_metric_calculator import (
    RankerMetricCalculator,
)


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def main(cfg: DictConfig):
    # MLflow 활성화 여부 확인
    enable_mlflow = cfg.log.get("enable_mlflow", True)

    if enable_mlflow:
        mlflow.set_experiment(cfg.log.experiment_name)
        mlflow.start_run(run_name=cfg.log.run_name)
        mlflow.log_params(
            {
                **cfg.data,
                **cfg.models.ranker,
                **cfg.training,
            }
        )

    try:
        data_loader = RankerDatasetLoader(data_config=DataConfig(**cfg.data))
        data = data_loader.prepare_ranker_dataset(
            is_rank=True,
            filter_config=cfg.preprocess.filter,
        )

        X_train, y_train, X_valid, y_valid, X_test_cold_users, X_test_warm_users = (
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            data["X_test_cold_start_user"],
            data["X_test_warm_start_user"],
        )

        trainer = instantiate(
            cfg.models.ranker,
            recommend_batch_size=cfg.training.evaluation.recommend_batch_size,
        )

        trainer.fit(X_train, y_train, X_valid, y_valid)

        # save model
        trainer.save_model()

        # 모델 저장 및 MLflow artifact 업로드
        if enable_mlflow:
            model_path = Path(trainer.model_path) / f"{trainer.results}.model"

            if model_path.exists():
                mlflow.log_artifact(str(model_path))

            # Feature importance 플롯 저장 및 업로드
            if hasattr(trainer, "plot_feature_importance"):
                trainer.plot_feature_importance()
                # plot_feature_importance가 파일로 저장하므로 해당 파일을 artifact로 업로드
                feature_importance_path = (
                    Path(trainer.model_path)
                    / f"{trainer.results}_feature_importance.png"
                )
                if feature_importance_path.exists():
                    mlflow.log_artifact(str(feature_importance_path))

        X_train["target"] = y_train
        train_liked_items = (
            X_train[X_train["target"] == 1]
            .groupby("reviewer_id")["diner_idx"]
            .apply(np.array)
        )

        candidates = trainer.calculate_rank(data["candidates"])

        metric_calculator = RankerMetricCalculator(
            top_k_values=cfg.training.evaluation.top_k_values_for_pred,
            model=trainer,
            features=cfg.models.ranker.features,
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

        # warm/cold/all 별 평균 계산 + 로그
        for user_type, metric in metric_dict.items():
            metric_calculator.calculate_mean_metric(metric)
            # metric은 in-place로 업데이트되므로 직접 사용
            if enable_mlflow:
                for k in cfg.training.evaluation.top_k_values_for_pred:
                    for metric_name, metric_value in metric[k].items():
                        mlflow.log_metric(
                            f"{user_type}_{metric_name}_at_{k}", metric_value
                        )

        # 최종 리포트 출력
        metric_calculator.report_metric_with_warm_cold_all_users(
            metric_dict=metric_dict, data_type="test"
        )
    finally:
        if enable_mlflow:
            mlflow.end_run()


if __name__ == "__main__":
    main()
