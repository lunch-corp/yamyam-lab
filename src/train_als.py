import os
import traceback
from argparse import ArgumentParser
from datetime import datetime

from data.dataset import DataConfig, DatasetLoader
from evaluation.metric_calculator import ALSMetricCalculator
from model import ALS
from tools.config import load_yaml
from tools.logger import setup_logger
from tools.parse_args import parse_args_als, save_command_to_file

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/models/mf/{model}.yaml")
RESULT_PATH = os.path.join(ROOT_PATH, "./result/{test}/{model}/{dt}")


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
        logger.info(
            f"train dataset period: {config.preprocess.data.train_time_point} <= dt < {config.preprocess.data.val_time_point}"
        )
        logger.info(
            f"val dataset period: {config.preprocess.data.val_time_point} <= dt < {config.preprocess.data.test_time_point}"
        )
        logger.info(
            f"test dataset period: {config.preprocess.data.test_time_point} <= dt < {config.preprocess.data.end_time_point}"
        )

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

        # train als.
        # currently, validation loss is not reported because implicit library does not support calculating validation loss.
        model = ALS(
            alpha=args.alpha,
            factors=args.factors,
            regularization=args.regularization,
            iterations=args.iterations,
            use_gpu=args.use_gpu,
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

        # calculate metric for test data with warm / cold / all users separately
        metric_dict = metric_calculator.generate_recommendations_and_calculate_metric(
            X_train=data["X_train_df"],
            X_val_warm_users=data["X_test_warm_users"],
            X_val_cold_users=data["X_test_cold_users"],
            most_popular_diner_ids=data["most_popular_diner_ids"],
            filter_already_liked=True,
            train_csr=data["X_train"],
        )

        # for each user type, the metric is not yet averaged but summed, so calculate mean
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
    args = parse_args_als()
    main(args)
