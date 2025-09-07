import os
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from model import ALS
from tools.parse_args import parse_args_als, save_command_to_file

from data.dataset import DataConfig, DatasetLoader
from evaluation.metric_calculator import ALSMetricCalculator
# from postprocess.postprocess import rerank_region_periphery
from tools.config import load_yaml
from tools.logger import setup_logger
from tools.parse_args import save_command_to_file
from postprocess.postprocess import RegionPeripheryReranker

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

        reranked_ids, _ = reranker.rerank(
            item_ids=candidates,
            base_scores=base_scores,
            item_meta=diner_meta,   # 주의: 함수 버전에서는 item_meta_std였는데, 클래스에서는 그냥 item_meta로 받음
            k=max(top_k_values),
        )

        reranked_most_popular = reranked_ids.tolist()


        logger.info("######## Number of reviews statistics ########")
        logger.info(f"Number of reviews in train: {data['X_train'].data.shape[0]}")
        logger.info(f"Number of reviews in val: {data['X_val'].data.shape[0]}")
        logger.info(f"Number of reviews in test: {data['X_test'].data.shape[0]}")

        logger.info("######## Train data statistics ########")
        logger.info(f"Number of users in train: {len(data['train_user_ids'])}")
        logger.info(f"Number of diners in train: {len(data['train_diner_ids'])}")
        logger.info(f"Number of feedbacks in train: {data['X_train'].data.shape[0]}")
        train_density = round(
            100
            * data["X_train"].data.shape[0]
            / (len(data["train_user_ids"]) * len(data["train_diner_ids"])),
            4,
        )
        logger.info(f"Train data density: {train_density}%")

        logger.info("######## Validation data statistics ########")
        logger.info(f"Number of users in val: {len(data['val_user_ids'])}")
        logger.info(f"Number of diners in val: {len(data['val_diner_ids'])}")
        logger.info(f"Number of feedbacks in val: {data['X_val'].data.shape[0]}")
        val_density = round(
            100
            * data["X_val"].data.shape[0]
            / (len(data["val_user_ids"]) * len(data["val_diner_ids"])),
            4,
        )
        logger.info(f"Validation data density: {val_density}%")

        logger.info("######## Test data statistics ########")
        logger.info(f"Number of users in test: {len(data['test_user_ids'])}")
        logger.info(f"Number of diners in test: {len(data['test_diner_ids'])}")
        logger.info(f"Number of feedbacks in test: {data['X_test'].data.shape[0]}")
        test_density = round(
            100
            * data["X_test"].data.shape[0]
            / (len(data["test_user_ids"]) * len(data["test_diner_ids"])),
            4,
        )
        logger.info(f"Test data density: {test_density}%")

        logger.info(
            "######## Warm / Cold users analysis in validation and test dataset ########"
        )
        logger.info(
            f"Number of users within train, but not in val: {len(set(data['train_user_ids']) - set(data['val_user_ids']))}"
        )
        logger.info(
            f"Number of users within train, but not in test: {len(set(data['train_user_ids']) - set(data['test_user_ids']))}"
        )
        logger.info(
            f"Number of warm start users in val: {len(data['val_warm_start_user_ids'])}"
        )
        logger.info(
            f"Number of cold start users in val: {len(data['val_cold_start_user_ids'])}"
        )
        logger.info(
            f"Ratio of cold start users in val: {100 * round(len(data['val_cold_start_user_ids']) / (len(data['val_warm_start_user_ids']) + len(data['val_cold_start_user_ids'])), 4)}%"
        )
        logger.info(
            f"Number of warm start users in test: {len(data['test_warm_start_user_ids'])}"
        )
        logger.info(
            f"Number of cold start users in test: {len(data['test_cold_start_user_ids'])}"
        )
        logger.info(
            f"Ratio of cold start users in test: {100 * round(len(data['test_cold_start_user_ids']) / (len(data['test_warm_start_user_ids']) + len(data['test_cold_start_user_ids'])), 4)}%"
        )

        model = ALS(
            alpha=args.alpha,
            factors=args.factors,
            regularization=args.regularization,
            iterations=args.iterations,
            use_gpu=args.use_gpu,
            calculate_training_loss=args.calculate_training_loss,
        )


        model.fit(data["X_train"])

        metric_calculator = ALSMetricCalculator(
            diner_ids=list(data["diner_mapping"].values()),
            model=model,
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
            most_popular_diner_ids=reranked_most_popular,
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
    args = parse_args_als()
    main(args)
