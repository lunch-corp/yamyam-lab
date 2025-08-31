import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from data.dataset import DataConfig, DatasetLoader
from evaluation.metric_calculator import MostPopularMetricCalculator
from postprocess.postprocess import rerank_region_periphery
from tools.config import load_yaml
from tools.logger import common_logging, setup_logger
from tools.parse_args import save_command_to_file

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/models/mf/{model}.yaml")
PREPROCESS_CONFIG_PATH = os.path.join(ROOT_PATH, "./config/preprocess/preprocess.yaml")
RESULT_PATH = os.path.join(ROOT_PATH, "./result/{test}/{model}/{dt}")


def main() -> None:
    # set result path
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    test_flag = "untest"
    result_path = RESULT_PATH.format(test=test_flag, model="most_popular", dt=dt)
    os.makedirs(result_path, exist_ok=True)
    # load config
    config = load_yaml(
        CONFIG_PATH.format(model="als")
    )  # Note: use als config, because all configs are overlapped.
    preprocess_config = load_yaml(PREPROCESS_CONFIG_PATH)

    file_name = config.post_training.file_name
    logger = setup_logger(os.path.join(result_path, file_name.log))
    save_command_to_file(str(RESULT_PATH))

    try:
        logger.info("model: most_popular")
        logger.info(f"training results will be saved in {RESULT_PATH}")

        # 데이터 로딩
        fe = config.preprocess.feature_engineering
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
        data_loader = DatasetLoader(data_config=data_config)
        data = data_loader.prepare_train_val_dataset(
            is_csr=True, filter_config=preprocess_config.filter
        )
        common_logging(config=config, data=data, logger=logger)

        # K 설정
        top_k_values_for_pred = config.training.evaluation.top_k_values_for_pred
        top_k_values_for_candidate = (
            config.training.evaluation.top_k_values_for_candidate
        )
        top_k_values = top_k_values_for_pred + top_k_values_for_candidate

        item_meta = data["diner_meta"]
        candidates = np.array(data["most_popular_diner_ids"], dtype=np.int64)
        base_scores = 1.0 / (np.arange(len(candidates)) + 1)

        meta_ids = item_meta["diner_idx"]
        if not pd.api.types.is_integer_dtype(meta_ids.dtype):
            meta_vals = (
                pd.to_numeric(meta_ids, errors="coerce")
                .dropna()
                .astype(np.int64)
                .to_numpy()
            )
        else:
            meta_vals = meta_ids.to_numpy(dtype=np.int64, copy=False)

        mask = np.isin(candidates, meta_vals)
        dropped = int((~mask).sum())
        if dropped:
            logger.warning(f"dropped {dropped} candidates not in item_meta")

        candidates = candidates[mask]
        base_scores = base_scores[mask]

        reranked_ids, _ = rerank_region_periphery(
            item_ids=candidates,
            base_scores=base_scores,
            item_meta_std=item_meta,
            k=max(top_k_values),
            region_label="서울 강남구",
            hotspot_coords=None,
            n_auto_hotspots=5,
            periphery_strength=0.5,
            periphery_cap=0.15,
            lambda_div=0.55,
            w_cat=0.5,
            w_geo=0.5,
            geo_tau_km=2.0,
        )
        reranked_most_popular = reranked_ids.tolist()

        # 평가
        metric_calculator = MostPopularMetricCalculator(
            top_k_values=top_k_values,
            filter_already_liked=True,
            recommend_batch_size=config.training.evaluation.recommend_batch_size,
            logger=logger,
        )

        metric_dict = metric_calculator.generate_recommendations_and_calculate_metric(
            X_train=data["X_train_df"],
            X_val_warm_users=data["X_val_warm_users"],
            X_val_cold_users=data["X_val_cold_users"],
            most_popular_diner_ids=reranked_most_popular,
            filter_already_liked=True,
            most_popular_rec_to_warm_users=True,
        )
        for user_type, metric in metric_dict.items():
            metric_calculator.calculate_mean_metric(metric)
        logger.info(
            "################################ Validation data metric report ################################"
        )
        metric_calculator.report_metric_with_warm_cold_all_users(
            metric_dict=metric_dict, data_type="val"
        )

        metric_dict = metric_calculator.generate_recommendations_and_calculate_metric(
            X_train=data["X_train_df"],
            X_val_warm_users=data["X_test_warm_users"],
            X_val_cold_users=data["X_test_cold_users"],
            most_popular_diner_ids=reranked_most_popular,
            filter_already_liked=True,
            most_popular_rec_to_warm_users=True,
        )
        for user_type, metric in metric_dict.items():
            metric_calculator.calculate_mean_metric(metric)
        logger.info(
            "################################ Test data metric report ################################"
        )
        metric_calculator.report_metric_with_warm_cold_all_users(
            metric_dict=metric_dict, data_type="test"
        )


    except Exception as e:
        print("Error during project run:", repr(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()
