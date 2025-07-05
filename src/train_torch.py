import copy
import importlib
import os
import pickle
import traceback
from argparse import ArgumentParser
from datetime import datetime

import pandas as pd
import torch
from torch import optim

from data.dataset import (
    DataConfig,
    DatasetLoader,
)
from evaluation.metric_calculator import SVDBiasMetricCalculator
from loss.custom import svd_loss
from preprocess.preprocess import prepare_torch_dataloader
from tools.config import load_yaml
from tools.logger import common_logging, setup_logger
from tools.plot import plot_metric_at_k

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/models/mf/{model}.yaml")
RESULT_PATH = os.path.join(ROOT_PATH, "./result/{test}/{model}/{dt}")


def main(args: ArgumentParser.parse_args):
    # set result path
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    test_flag = "test" if args.test else "untest"
    result_path = RESULT_PATH.format(test=test_flag, model=args.model, dt=dt)
    os.makedirs(result_path, exist_ok=True)
    config = load_yaml(CONFIG_PATH.format(model=args.model))

    # predefine config
    top_k_values = config.training.evaluation.top_k_values_for_pred
    file_name = config.post_training.file_name

    logger = setup_logger(os.path.join(result_path, file_name.log))

    try:
        logger.info(f"model: {args.model}")
        logger.info(f"device: {args.device}")
        logger.info(f"batch size: {args.batch_size}")
        logger.info(f"learning rate: {args.lr}")
        logger.info(f"regularization: {args.regularization}")
        logger.info(f"epochs: {args.epochs}")
        logger.info(
            f"number of factors for user / item embedding: {args.embedding_dim}"
        )
        logger.info(f"test ratio: {args.test_ratio}")
        logger.info(f"patience for watching validation loss: {args.patience}")
        logger.info(f"test: {args.test}")
        logger.info(f"training results will be saved in {result_path}")

        # generate dataloader for pytorch training pipeline
        data_loader = DatasetLoader(
            data_config=DataConfig(
                X_columns=["diner_idx", "reviewer_id"],
                y_columns=["reviewer_review_score"],
                is_timeseries_by_time_point=config.preprocess.data.is_timeseries_by_time_point,
                train_time_point=config.preprocess.data.train_time_point,
                val_time_point=config.preprocess.data.val_time_point,
                test_time_point=config.preprocess.data.test_time_point,
                end_time_point=config.preprocess.data.end_time_point,
                test=args.test,
            ),
        )
        data = data_loader.prepare_train_val_dataset()

        common_logging(
            config=config,
            data=data,
            logger=logger,
        )

        train_dataloader, val_dataloader = prepare_torch_dataloader(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"]
        )

        # for qualitative eval
        pickle.dump(data, open(os.path.join(result_path, file_name.data_object), "wb"))

        # import model module
        model_path = f"model.mf.{args.model}"
        model_module = importlib.import_module(model_path).Model
        model = model_module(
            user_ids=torch.tensor(list(data["user_mapping"].values())).to(args.device),
            diner_ids=torch.tensor(list(data["diner_mapping"].values())).to(
                args.device
            ),
            embedding_dim=args.embedding_dim,
            top_k_values=top_k_values,
            model_name=args.model,
            mu=data["y_train"].mean(),
        ).to(args.device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        # define metric calculator for test data metric
        metric_calculator = SVDBiasMetricCalculator(
            diner_ids=list(data["diner_mapping"].values()),
            model=model,
            top_k_values=top_k_values,
            filter_already_liked=True,
            recommend_batch_size=config.training.evaluation.recommend_batch_size,
            logger=logger,
        )

        # train model
        best_loss = float("inf")
        for epoch in range(args.epochs):
            logger.info(f"################## epoch {epoch} ##################")

            # training
            model.train()
            tr_loss = 0.0
            for X_train, y_train in train_dataloader:
                diners, users = X_train[:, 0], X_train[:, 1]
                optimizer.zero_grad()
                y_pred = model(users, diners)
                loss = svd_loss(
                    pred=y_pred,
                    true=y_train,
                    params=[param.data for param in model.parameters()],
                    regularization=args.regularization,
                    user_idx=users,
                    diner_idx=diners,
                    num_users=data["num_users"],
                    num_diners=data["num_diners"],
                )
                loss.backward()
                optimizer.step()

                tr_loss += loss.item()

            tr_loss = round(tr_loss / len(train_dataloader), 6)
            model.tr_loss.append(tr_loss)

            # validation
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for X_val, y_val in val_dataloader:
                    diners, users = X_val[:, 0], X_val[:, 1]
                    y_pred = model(users, diners)
                    loss = svd_loss(
                        pred=y_pred,
                        true=y_val,
                        params=[param.data for param in model.parameters()],
                        regularization=args.regularization,
                        user_idx=users,
                        diner_idx=diners,
                        num_users=data["num_users"],
                        num_diners=data["num_diners"],
                    )
                    val_loss += loss.item()
                val_loss = round(val_loss / len(val_dataloader), 6)

            logger.info(f"Train Loss: {tr_loss}")
            logger.info(f"Validation Loss: {val_loss}")

            # calculate metric for test data with warm / cold / all users separately
            metric_dict = (
                metric_calculator.generate_recommendations_and_calculate_metric(
                    X_train=pd.DataFrame(
                        data["X_train"], columns=["diner_idx", "reviewer_id"]
                    ),
                    X_val_warm_users=pd.DataFrame(
                        data["X_val_warm_users"], columns=["diner_idx", "reviewer_id"]
                    ),
                    X_val_cold_users=pd.DataFrame(
                        data["X_val_cold_users"], columns=["diner_idx", "reviewer_id"]
                    ),
                    most_popular_diner_ids=data["most_popular_diner_ids"],
                    filter_already_liked=True,
                )
            )

            # for each user type, the metric is not yet averaged but summed, so calculate mean
            for user_type, metric in metric_dict.items():
                metric_calculator.calculate_mean_metric(metric)

            # for each user type, report map, ndcg, recall
            logger.info(
                f"################## Validation data metric report for {epoch} epoch ##################"
            )
            metric_calculator.report_metric_with_warm_cold_all_users(
                metric_dict=metric_dict, data_type="val"
            )

            # save metric at current epoch for later metric plotting
            metric_calculator.save_metric_at_current_epoch(
                metric_at_k=metric_dict["all"],
                metric_at_k_total_epochs=model.metric_at_k_total_epochs,
            )

            torch.save(
                model.state_dict(),
                str(os.path.join(result_path, file_name.weight)),
            )
            pickle.dump(
                model.tr_loss,
                open(os.path.join(result_path, file_name.training_loss), "wb"),
            )
            pickle.dump(
                model.metric_at_k_total_epochs,
                open(os.path.join(result_path, file_name.metric), "wb"),
            )
            logger.info(f"successfully saved svd_bias torch model: epoch {epoch}")

            pickle.dump(
                model.tr_loss,
                open(os.path.join(result_path, file_name.training_loss), "wb"),
            )
            pickle.dump(
                model.metric_at_k_total_epochs,
                open(os.path.join(result_path, file_name.metric), "wb"),
            )

            if best_loss > val_loss:
                prev_best_loss = best_loss
                best_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience = args.patience
                torch.save(
                    model.state_dict(),
                    str(os.path.join(result_path, file_name.weight)),
                )
                logger.info(
                    f"Best validation: {best_loss}, Previous validation loss: {prev_best_loss}"
                )
            else:
                patience -= 1
                logger.info(
                    f"Validation loss did not decrease. Patience {patience} left."
                )
                if patience == 0:
                    logger.info(
                        f"Patience over. Early stopping at epoch {epoch} with {best_loss} validation loss"
                    )
                    break

            # Load the best model weights
            model.load_state_dict(best_model_weights)
            logger.info("Load weight with best validation loss")

            torch.save(
                model.state_dict(),
                str(os.path.join(result_path, file_name.weight)),
            )
            logger.info("Save final model")

        # define metric calculator for test data metric
        metric_calculator = SVDBiasMetricCalculator(
            diner_ids=list(data["diner_mapping"].values()),
            model=model,
            top_k_values=top_k_values,
            filter_already_liked=True,
            recommend_batch_size=config.training.evaluation.recommend_batch_size,
            logger=logger,
        )

        # calculate metric for test data with warm / cold / all users separately
        metric_dict = metric_calculator.generate_recommendations_and_calculate_metric(
            X_train=pd.DataFrame(data["X_train"], columns=["diner_idx", "reviewer_id"]),
            X_val_warm_users=pd.DataFrame(
                data["X_test_warm_users"], columns=["diner_idx", "reviewer_id"]
            ),
            X_val_cold_users=pd.DataFrame(
                data["X_test_cold_users"], columns=["diner_idx", "reviewer_id"]
            ),
            most_popular_diner_ids=data["most_popular_diner_ids"],
            filter_already_liked=True,
        )

        # for each user type, the metric is not yet averaged but summed, so calculate mean
        for user_type, metric in metric_dict.items():
            metric_calculator.calculate_mean_metric(metric)

        # for each user type, report map, ndcg, recall
        metric_calculator.report_metric_with_warm_cold_all_users(
            metric_dict=metric_dict, data_type="test"
        )

        # plot metrics
        plot_metric_at_k(
            metric=model.metric_at_k_total_epochs,
            tr_loss=model.tr_loss,
            parent_save_path=result_path,
            top_k_values_for_pred=top_k_values,
            top_k_values_for_candidate=[],
        )
    except:
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    from tools.parse_args import parse_args

    args = parse_args()
    main(args)
