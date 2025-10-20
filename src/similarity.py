import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from data.config import DataConfig
from data.csr import CsrDatasetLoader
from evaluation.metric_calculator.similarity_metric_calculator import (
    ItemBasedMetricCalculator,
)
from model.classic_cf.item_based import ItemBasedCollaborativeFiltering
from tools.config import load_yaml
from tools.logger import common_logging, setup_logger
from tools.parse_args import save_command_to_file

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/models/mf/{model}.yaml")
PREPROCESS_CONFIG_PATH = os.path.join(ROOT_PATH, "./config/preprocess/preprocess.yaml")
RESULT_PATH = os.path.join(ROOT_PATH, "./result/{test}/{model}/{dt}")


def main() -> None:
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    test_flag = "untest"
    result_path = RESULT_PATH.format(test=test_flag, model="item_based", dt=dt)
    os.makedirs(result_path, exist_ok=True)

    config = load_yaml(CONFIG_PATH.format(model="als"))
    preprocess_config = load_yaml(PREPROCESS_CONFIG_PATH)

    save_command_to_file(result_path)

    top_k_values_for_pred = config.training.evaluation.top_k_values_for_pred
    top_k_values_for_candidate = config.training.evaluation.top_k_values_for_candidate
    top_k_values = top_k_values_for_pred + top_k_values_for_candidate
    file_name = config.post_training.file_name
    fe = config.preprocess.feature_engineering

    logger = setup_logger(os.path.join(result_path, file_name.log))

    try:
        logger.info("model: item_based")
        logger.info(f"results will be saved in {result_path}")

        # ------------------ 데이터 로딩 ------------------
        data_config = DataConfig(
            X_columns=["diner_idx", "reviewer_id"],
            y_columns=["reviewer_review_score"],
            user_engineered_feature_names=fe.user_engineered_feature_names,
            diner_engineered_feature_names=fe.diner_engineered_feature_names,
            is_timeseries_by_time_point=config.preprocess.data.is_timeseries_by_time_point,
            train_time_point=config.preprocess.data.train_time_point,
            val_time_point=config.preprocess.data.val_time_point,
            test_time_point=config.preprocess.data.test_time_point,
            end_time_point=config.preprocess.data.end_time_point,
            test=False,
        )
        data_loader = CsrDatasetLoader(data_config=data_config)
        data = data_loader.prepare_csr_dataset(
            is_csr=False, filter_config=preprocess_config.filter
        )

        common_logging(config, data, logger)

        # ------------------ Item-based CF 모델 초기화 ------------------
        logger.info("Initializing Item-based Collaborative Filtering model...")

        num_items = data["X_train"].shape[0]
        dummy_embeddings = np.eye(num_items)

        item_based_model = ItemBasedCollaborativeFiltering(
            user_item_matrix=data["X_train"],  # DataFrame 형태
            item_embeddings=dummy_embeddings,
            user_mapping=data["user_mapping"],
            item_mapping=data["diner_mapping"],
            diner_df=None,
        )

        # ------------------ test_dict 생성 (단일 train/test) ------------------
        data["test_dict"] = {
            "train": data["X_train_df"],  # DataFrame
            "test": pd.concat(
                [data["X_test_warm_users"], data["X_test_cold_users"]],
                ignore_index=True,
            ),
        }

        # ------------------ 평가 ------------------
        logger.info("Evaluating Item-based CF model...")
        metric_calculator = ItemBasedMetricCalculator(
            model=item_based_model,
            test_data=data["test_dict"],
            top_k_values=top_k_values,
            logger=logger,
        )

        metrics = metric_calculator.evaluate()

        logger.info("Evaluation results:")
        for metric_name, score in metrics.items():
            logger.info(f"{metric_name}: {score:.4f}")

    except Exception as e:
        logger.error("An error occurred during training")
        logger.error(traceback.format_exc())
        raise e


if __name__ == "__main__":
    main()
