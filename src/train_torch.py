import copy
import importlib
import os
import pickle
import traceback
from argparse import ArgumentParser

import torch
from torch import optim

from constant.metric.metric import Metric
from data.dataset import DatasetLoader
from loss.custom import svd_loss
from preprocess.preprocess import prepare_torch_dataloader
from tools.config import load_yaml
from tools.logger import setup_logger
from tools.plot import plot_metric_at_k

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/models/{model}.yaml")


def main(args: ArgumentParser.parse_args):
    os.makedirs(args.result_path, exist_ok=True)
    config = load_yaml(CONFIG_PATH.format(model=args.model))

    # predefine config
    top_k_values_for_pred = config.training.evaluation.top_k_values_for_pred
    file_name = config.post_training.file_name

    logger = setup_logger(os.path.join(args.result_path, file_name.log))

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
        logger.info(f"result path: {args.result_path}")
        logger.info(f"test: {args.test}")

        # generate dataloader for pytorch training pipeline
        data_loader = DatasetLoader(
            test_size=args.test_ratio,
            min_reviews=config.preprocess.data.min_review,
            X_columns=["diner_idx", "reviewer_id"],
            y_columns=["reviewer_review_score"],
            test=args.test,
        )
        data = data_loader.prepare_train_val_dataset()
        logger.info(f"number of diners: {data['num_diners']}")
        logger.info(f"number of users: {data['num_users']}")

        train_dataloader, val_dataloader = prepare_torch_dataloader(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"]
        )

        # for qualitative eval
        pickle.dump(
            data, open(os.path.join(args.result_path, file_name.data_object), "wb")
        )

        # import model module
        model_path = f"model.{args.model}"
        model_module = importlib.import_module(model_path).Model
        model = model_module(
            num_users=data["num_users"],
            num_items=data["num_diners"],
            embedding_dim=args.embedding_dim,
            top_k_values=top_k_values_for_pred,
            mu=data["y_train"].mean(),
        ).to(args.device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr)

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

            model.recommend_all(
                X_train=data["X_train"],
                X_val=data["X_val"],
                recommend_batch_size=config.training.evaluation.recommend_batch_size,
                top_k_values=top_k_values_for_pred,
                filter_already_liked=True,
            )

            maps = []
            ndcgs = []

            for k in top_k_values_for_pred:
                # no candidate metric
                map = round(model.metric_at_k[k][Metric.MAP], 5)
                ndcg = round(model.metric_at_k[k][Metric.NDCG], 5)

                count = model.metric_at_k[k][Metric.COUNT]

                logger.info(
                    f"maP@{k}: {map} with {count} users out of all {model.num_users} users"
                )
                logger.info(
                    f"ndcg@{k}: {ndcg} with {count} users out of all {model.num_users} users"
                )

                maps.append(str(map))
                ndcgs.append(str(ndcg))

            logger.info("top k results for direct prediction @3, @7, @10, @20 in order")
            logger.info(f"map result: {'|'.join(maps)}")
            logger.info(f"ndcg result: {'|'.join(ndcgs)}")

            pickle.dump(
                model.tr_loss,
                open(os.path.join(args.result_path, file_name.training_loss), "wb"),
            )
            pickle.dump(
                model.metric_at_k_total_epochs,
                open(os.path.join(args.result_path, file_name.metric), "wb"),
            )

            if best_loss > val_loss:
                prev_best_loss = best_loss
                best_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience = args.patience
                torch.save(
                    model.state_dict(),
                    str(os.path.join(args.result_path, file_name.weight)),
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
                str(os.path.join(args.result_path, file_name.weight)),
            )
            logger.info("Save final model")

        # plot metrics
        plot_metric_at_k(
            metric=model.metric_at_k_total_epochs,
            tr_loss=model.tr_loss,
            parent_save_path=args.result_path,
            top_k_values_for_pred=top_k_values_for_pred,
            top_k_values_for_candidate=[],
        )
    except:
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    from tools.parse_args import parse_args

    args = parse_args()
    main(args)
