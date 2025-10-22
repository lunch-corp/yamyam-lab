import os
import traceback
from datetime import datetime

from evaluation.metric_calculator import MostPopularMetricCalculator
from tools.config import load_yaml
from tools.logger import common_logging, setup_logger
from tools.parse_args import save_command_to_file

from data.config import DataConfig
from data.csr import CsrDatasetLoader

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
    # save command used in argparse
    save_command_to_file(result_path)

    # predefine config
    top_k_values_for_pred = config.training.evaluation.top_k_values_for_pred
    top_k_values_for_candidate = config.training.evaluation.top_k_values_for_candidate
    top_k_values = top_k_values_for_pred + top_k_values_for_candidate
    file_name = config.post_training.file_name
    fe = config.preprocess.feature_engineering

    logger = setup_logger(os.path.join(result_path, file_name.log))

    try:
        logger.info("model: most_popular")
        logger.info(f"training results will be saved in {result_path}")

        data_loader = CsrDatasetLoader(
            data_config=DataConfig(
                X_columns=["diner_idx", "reviewer_id"],
                y_columns=["reviewer_review_score"],
                user_engineered_feature_names=fe.user_engineered_feature_names,
                diner_engineered_feature_names=fe.diner_engineered_feature_names,
                is_timeseries_by_time_point=config.preprocess.data.is_timeseries_by_time_point,
                train_time_point=config.preprocess.data.train_time_point,
                val_time_point=config.preprocess.data.val_time_point,
                test_time_point=config.preprocess.data.test_time_point,
                end_time_point=config.preprocess.data.end_time_point,
                test=False,  # hard coding
            ),
        )
        # Note: although is_csr is set True, we do not use train_csr dataset, but use val / test data in pandas dataframe.
        data = data_loader.prepare_csr_dataset(
            is_csr=True,
            filter_config=preprocess_config.filter,
        )

        common_logging(
            config=config,
            data=data,
            logger=logger,
        )

        # Note: In most popular model, we do not actually have any machine learning models.
        # Therefore, we directly calculate validation / test data metric using most popular diner_ids

        # define metric calculator for validation / test data metric
        metric_calculator = MostPopularMetricCalculator(
            top_k_values=top_k_values,
            filter_already_liked=True,
            recommend_batch_size=config.training.evaluation.recommend_batch_size,
            logger=logger,
        )

        # calculate metric for **validation data** with warm / cold / all users separately
        metric_dict = metric_calculator.generate_recommendations_and_calculate_metric(
            X_train=data["X_train_df"],
            X_val_warm_users=data["X_val_warm_users"],
            X_val_cold_users=data["X_val_cold_users"],
            most_popular_diner_ids=data["most_popular_diner_ids"],
            filter_already_liked=True,
            most_popular_rec_to_warm_users=True,
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
            most_popular_diner_ids=data["most_popular_diner_ids"],
            filter_already_liked=True,
            most_popular_rec_to_warm_users=True,
        )

        # for each user type, the metric is not yet averaged but summed, so calculate mean
        logger.info(
            "################################ Test data metric report ################################"
        )
        for user_type, metric in metric_dict.items():
            metric_calculator.calculate_mean_metric(metric)

        # for each user type, report map, ndcg, recall
        metric_calculator.report_metric_with_warm_cold_all_users(
            metric_dict=metric_dict, data_type="test"
        )

    except:
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
