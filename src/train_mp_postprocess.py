import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from data.dataset import DataConfig, DatasetLoader

from tools.config import load_yaml
from tools.logger import setup_logger, common_logging
from tools.parse_args import save_command_to_file
from postprocess.postprocess import RegionPeripheryReranker
from evaluation.metric_calculator import MostPopularMetricCalculator


ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/models/mf/{model}.yaml")
RESULT_PATH = os.path.join(ROOT_PATH, "./result/{test}/{model}/{dt}")


def main(args) -> None:
    # set result path
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    test_flag = "untest"
    result_path = RESULT_PATH.format(test=test_flag, model="most_popular", dt=dt)
    os.makedirs(result_path, exist_ok=True)
    # load config
    config = load_yaml(
        CONFIG_PATH.format(model="als")
    )  # Note: use als config, because all configs are overlapped.

    file_name = config.post_training.file_name
    logger = setup_logger(os.path.join(result_path, file_name.log))
    save_command_to_file(str(RESULT_PATH))

    try:
        logger.info("model: rerank_most_popular")
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
        data = data_loader.prepare_train_val_dataset(is_csr=True)

        common_logging(config, data, logger)

        _, diner_data, diner_category_data = data_loader.load_dataset()
        diner_meta = pd.merge(
            diner_data[["diner_idx", "diner_lat", "diner_lon"]],
            diner_category_data[["diner_idx", "diner_category_large"]],
            on="diner_idx",
            how="left",
        )
        top_k_values_for_pred = config.training.evaluation.top_k_values_for_pred
        top_k_values_for_candidate = (
            config.training.evaluation.top_k_values_for_candidate
        )
        top_k_values = top_k_values_for_pred + top_k_values_for_candidate

        candidates = np.array(data["most_popular_diner_ids"], dtype=np.int64)
        base_scores = 1.0 / (np.arange(len(candidates)) + 1)


        reranker = RegionPeripheryReranker(
            region_label=args.region_label,
            hotspot_coords=args.hotspot_coords,
            n_auto_hotspots=args.n_auto_hotspots,
            periphery_strength=args.periphery_strength,
            periphery_cap=args.periphery_cap,
            lambda_div=args.lambda_div,
            w_cat=args.w_cat,
            w_geo=args.w_geo,
            geo_tau_km=args.geo_tau_km,
        )

        reranked_ids, _ = reranker.rerank(
            item_ids=candidates,
            base_scores=base_scores,
            item_meta=diner_meta,  
            k=max(top_k_values),
        )

        # reranked_most_popular = reranked_ids.tolist()
        diner_mapping = data["diner_mapping"]
        reranked_most_popular_internal = [
            diner_mapping[ext_id] for ext_id in reranked_ids if ext_id in diner_mapping
        ]
        


        metric_calculator = MostPopularMetricCalculator(
            diner_ids=list(data["diner_mapping"].values()),
            top_k_values=top_k_values,
            filter_already_liked=True,
            recommend_batch_size=config.training.evaluation.recommend_batch_size,
            logger=logger,
        )


        metric_dict = metric_calculator.generate_recommendations_and_calculate_metric(
            X_train=data["X_train_df"],
            X_val_warm_users=data["X_val_warm_users"],
            X_val_cold_users=data["X_val_cold_users"],
            most_popular_diner_ids=reranked_most_popular_internal,
            filter_already_liked=True,
            train_csr=data["X_train"],
        )

        # for each user type, the metric is not yet averaged but summed, so calculate mean
        for user_type, metric in metric_dict.items():
            metric_calculator.calculate_mean_metric(metric)

        # for each user type, report map, ndcg, recall
        logger.info(
            "################################ Validation data metric report ################################"
        )
        metric_calculator.report_metric_with_warm_cold_all_users(
            metric_dict=metric_dict, data_type="val"
        )

        # calculate metric for **test data** with warm / cold / all users separately
        metric_dict = metric_calculator.generate_recommendations_and_calculate_metric(
            X_train=data["X_train_df"],
            X_val_warm_users=data["X_test_warm_users"],
            X_val_cold_users=data["X_test_cold_users"],
            most_popular_diner_ids=reranked_most_popular_internal,
            filter_already_liked=True,
            train_csr=data["X_train"],
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
