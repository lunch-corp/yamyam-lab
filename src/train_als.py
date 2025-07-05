import os
import pickle
import traceback
from argparse import ArgumentParser
from datetime import datetime

from data.dataset import DataConfig, DatasetLoader
from evaluation.metric_calculator import ALSMetricCalculator
from model import ALS
from tools.config import load_yaml
from tools.google_drive import GoogleDriveManager
from tools.logger import common_logging, setup_logger
from tools.parse_args import parse_args_als, save_command_to_file
from tools.zip import zip_files_in_directory

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/models/mf/{model}.yaml")
RESULT_PATH = os.path.join(ROOT_PATH, "./result/{test}/{model}/{dt}")
ZIP_PATH = os.path.join(ROOT_PATH, "./zip/{test}/{model}/{dt}")


def main(args: ArgumentParser.parse_args) -> None:
    # set result path
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    test_flag = "test" if args.test else "untest"
    result_path = RESULT_PATH.format(test=test_flag, model="als", dt=dt)
    os.makedirs(result_path, exist_ok=True)
    # load config
    config = load_yaml(CONFIG_PATH.format(model="als"))
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
        logger.info(f"alpha: {args.alpha}")
        logger.info(f"factors: {args.factors}")
        logger.info(f"regularization: {args.regularization}")
        logger.info(f"iterations: {args.iterations}")
        logger.info(f"use_gpu: {args.use_gpu}")
        logger.info(f"calculate_training_loss: {args.calculate_training_loss}")
        logger.info(f"test: {args.test}")
        logger.info(f"training results will be saved in {result_path}")

        data_loader = DatasetLoader(
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
                test=args.test,
            ),
        )
        data = data_loader.prepare_train_val_dataset(is_csr=True)

        common_logging(
            config=config,
            data=data,
            logger=logger,
        )

        # train als.
        # currently, validation loss is not reported because implicit library does not support calculating validation loss.
        model = ALS(
            alpha=args.alpha,
            factors=args.factors,
            regularization=args.regularization,
            iterations=args.iterations,
            use_gpu=args.use_gpu,
            diner_mapping=data["diner_mapping"],
            calculate_training_loss=args.calculate_training_loss,
        )
        model.fit(data["X_train"])

        # define metric calculator for test data metric
        metric_calculator = ALSMetricCalculator(
            diner_ids=list(data["diner_mapping"].values()),
            model=model,
            top_k_values=top_k_values,
            filter_already_liked=True,
            recommend_batch_size=config.training.evaluation.recommend_batch_size,
            logger=logger,
        )

        # calculate metric for **validation data** with warm / cold / all users separately
        # Note that, we should calculate this metric for each iteration while training als,
        # but we could not find any methods to integrate it into implicit library,
        # so, we report validation metric after finishing training als.
        metric_dict = metric_calculator.generate_recommendations_and_calculate_metric(
            X_train=data["X_train_df"],
            X_val_warm_users=data["X_val_warm_users"],
            X_val_cold_users=data["X_val_cold_users"],
            most_popular_diner_ids=data["most_popular_diner_ids"],
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
            most_popular_diner_ids=data["most_popular_diner_ids"],
            filter_already_liked=True,
            train_csr=data["X_train"],
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

        if args.save_candidate:
            # generate candidates and zip related files
            zip_path = ZIP_PATH.format(
                test=test_flag, model="als", dt=dt
            )  # hard coding
            os.makedirs(zip_path, exist_ok=True)
            candidates_df = model.generate_candidates_for_each_user(
                top_k_value=config.post_training.candidate_generation.top_k,
                train_csr=data["X_train"],
            )
            # save files to zip
            pickle.dump(
                data["user_mapping"],
                open(os.path.join(zip_path, file_name.user_mapping), "wb"),
            )
            pickle.dump(
                data["diner_mapping"],
                open(os.path.join(zip_path, file_name.diner_mapping), "wb"),
            )
            candidates_df.to_parquet(
                os.path.join(zip_path, file_name.candidate), index=False
            )
            # zip file
            zip_files_in_directory(
                dir_path=zip_path,
                zip_file_name=f"{dt}.zip",
                allowed_type=[".pkl", ".parquet"],
                logger=logger,
            )
            # upload zip file to google drive
            manager = GoogleDriveManager(
                reusable_token_path=args.reusable_token_path,
                reuse_auth_info=True,
            )
            file_id = manager.upload_result(
                model_name="als",  # hard coding
                file_path=os.path.join(zip_path, f"{dt}.zip"),
                download_file_type="candidates",
            )
            logger.info(
                f"Successfully uploaded candidate results to google drive."
                f"File id: {file_id}"
            )

    except:
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    args = parse_args_als()
    main(args)
