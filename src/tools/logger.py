import logging
from typing import Any, Dict

from data.base import BaseDatasetLoader


def setup_logger(log_file):
    # Create a logger object
    logger = logging.getLogger("yamyam")
    logger.setLevel(logging.DEBUG)

    # Create a file handler to log messages to a file
    file_handler = logging.FileHandler(
        log_file, mode="w"
    )  # Open file in write mode to overwrite on each run

    # Set a logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def common_logging(
    config: Dict[str, Any], data: BaseDatasetLoader, logger: logging.Logger = logging
):
    """
    Common logging with dataset period and basic statistics about train / val / test dataset.
    """
    logger.info(
        f"train dataset period: {config.preprocess.data.train_time_point} <= dt < {config.preprocess.data.val_time_point}"
    )
    logger.info(
        f"val dataset period: {config.preprocess.data.val_time_point} <= dt < {config.preprocess.data.test_time_point}"
    )
    logger.info(
        f"test dataset period: {config.preprocess.data.test_time_point} <= dt < {config.preprocess.data.end_time_point}"
    )

    logger.info("######## Warm users / diners statistics ########")
    logger.info(f"Number of warm users: {len(set(data['train_user_ids']))}")
    logger.info(f"Number of warm diners: {len(set(data['train_diner_ids']))}")

    logger.info("######## Number of reviews statistics ########")
    logger.info(f"Number of reviews in train: {data['X_train'].shape[0]}")
    logger.info(f"Number of reviews in val: {data['X_val'].shape[0]}")
    logger.info(f"Number of reviews in test: {data['X_test'].shape[0]}")

    logger.info("######## Train data statistics ########")
    logger.info(f"Number of users in train: {len(data['train_user_ids'])}")
    logger.info(f"Number of diners in train: {len(data['train_diner_ids'])}")
    logger.info(f"Number of feedbacks in train: {data['X_train'].shape[0]}")
    train_density = round(
        100
        * data["X_train"].shape[0]
        / (len(data["train_user_ids"]) * len(data["train_diner_ids"])),
        4,
    )
    logger.info(f"Train data density: {train_density}%")

    logger.info("######## Validation data statistics ########")
    logger.info(f"Number of users in val: {len(data['val_user_ids'])}")
    logger.info(f"Number of diners in val: {len(data['val_diner_ids'])}")
    logger.info(f"Number of feedbacks in val: {data['X_val'].shape[0]}")
    val_density = round(
        100
        * data["X_val"].shape[0]
        / (len(data["val_user_ids"]) * len(data["val_diner_ids"])),
        4,
    )
    logger.info(f"Validation data density: {val_density}%")

    logger.info("######## Test data statistics ########")
    logger.info(f"Number of users in test: {len(data['test_user_ids'])}")
    logger.info(f"Number of diners in test: {len(data['test_diner_ids'])}")
    logger.info(f"Number of feedbacks in test: {data['X_test'].shape[0]}")
    test_density = round(
        100
        * data["X_test"].shape[0]
        / (len(data["test_user_ids"]) * len(data["test_diner_ids"])),
        4,
    )
    logger.info(f"Test data density: {test_density}%")

    logger.info(
        "######## Warm / Cold users analysis in validation and test dataset ########"
    )
    all_users = (
        set(data["train_user_ids"])
        | set(data["val_user_ids"])
        | set(data["test_user_ids"])
    )
    all_cold_users = set(data["val_cold_start_user_ids"]) | set(
        data["test_cold_start_user_ids"]
    )
    logger.info(f"Total number of users including warm / cold: {len(all_users)}")
    logger.info(
        f"Number of warm users: {len(set(data['train_user_ids']))} ({round(100 * len(set(data['train_user_ids'])) / len(all_users), 2)}%)"
    )
    logger.info(
        f"Number of cold users: {len(all_cold_users)} ({round(100 * len(all_cold_users) / len(all_users), 2)}%)"
    )
    logger.info(
        f"Number of warm start users in val: {len(data['val_warm_start_user_ids'])} ({round(100 * len(data['val_warm_start_user_ids']) / len(data['val_user_ids']), 2)}%)"
    )
    logger.info(
        f"Number of cold start users in val: {len(data['val_cold_start_user_ids'])} ({round(100 * len(data['val_cold_start_user_ids']) / len(data['val_user_ids']), 2)}%)"
    )
    logger.info(
        f"Number of warm start users in test: {len(data['test_warm_start_user_ids'])} ({round(100 * len(data['test_warm_start_user_ids']) / len(data['test_user_ids']), 2)}%)"
    )
    logger.info(
        f"Number of cold start users in test: {len(data['test_cold_start_user_ids'])} ({round(100 * len(data['test_cold_start_user_ids']) / len(data['test_user_ids']), 2)}%)"
    )
