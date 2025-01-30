import os
import copy
import traceback
import importlib
from argparse import ArgumentParser

import torch
from torch import optim

from preprocess.preprocess import (
    prepare_torch_dataloader,
    train_test_split_stratify,
)
from candidate.near import NearCandidateGenerator
from loss.custom import svd_loss
from tools.logger import setup_logger
from constant.preprocess.preprocess import MIN_REVIEWS
from constant.candidate.near import MAX_DISTANCE_KM
from constant.save.file_name import FileName


def main(args: ArgumentParser.parse_args):
    os.makedirs(args.result_path, exist_ok=True)
    logger = setup_logger(os.path.join(args.result_path, FileName.LOG.value))

    try:
        logger.info(f"model: {args.model}")
        logger.info(f"batch size: {args.batch_size}")
        logger.info(f"learning rate: {args.lr}")
        logger.info(f"regularization: {args.regularization}")
        logger.info(f"epochs: {args.epochs}")
        logger.info(f"number of factors for user / item embedding: {args.num_factors}")
        logger.info(f"test ratio: {args.test_ratio}")
        logger.info(f"patience for watching validation loss: {args.patience}")
        data = train_test_split_stratify(
            test_size=args.test_ratio,
            min_reviews=MIN_REVIEWS,
            X_columns=["diner_idx", "reviewer_id"],
            y_columns=["reviewer_review_score"],
            test=args.test,
        )
        train_dataloader, val_dataloader = prepare_torch_dataloader(
            data["X_train"], data["y_train"], data["X_val"], data["y_val"]
        )

        # import embedding module
        model_path = f"model.{args.model}"
        model_module = importlib.import_module(model_path).Model
        model = model_module(
            num_users=data["num_users"],
            num_items=data["num_diners"],
            num_factors=args.num_factors,
            mu=data["y_train"].mean(),
        )

        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        # get near 1km diner_ids
        candidate_generator = NearCandidateGenerator()
        near_diners = candidate_generator.get_near_candidates_for_all_diners(
            max_distance_km=MAX_DISTANCE_KM
        )
        # convert diner_ids
        diner_mapping = data["diner_mapping"]
        nearby_candidates_mapping = {}
        for ref_id, nearby_id in near_diners.items():
            # only get diner appeared in train/val dataset
            if diner_mapping.get(ref_id) is None:
                continue
            nearby_id_mapping = [
                diner_mapping.get(diner_id)
                for diner_id in nearby_id
                if diner_mapping.get(diner_id) is not None
            ]
            nearby_candidates_mapping[diner_mapping[ref_id]] = nearby_id_mapping

        # train model
        best_loss = float("inf")
        for epoch in range(args.epochs):
            logger.info(f"####### Epoch {epoch} #######")

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

            model.recommend(
                X_train=data["X_train"],
                X_val=data["X_val"],
                nearby_candidates=nearby_candidates_mapping,
                filter_already_liked=True,
            )
            maps = []
            ndcgs = []
            ranked_precs = []
            for K in model.metric_at_K.keys():
                map = round(model.metric_at_K[K]["map"], 5)
                ndcg = round(model.metric_at_K[K]["ndcg"], 5)
                ranked_prec = round(model.metric_at_K[K]["ranked_prec"], 5)
                count = model.metric_at_K[K]["count"]
                logger.info(
                    f"maP@{K}: {map} with {count} users out of all {model.num_users} users"
                )
                logger.info(
                    f"ndcg@{K}: {ndcg} with {count} users out of all {model.num_users} users"
                )
                logger.info(f"ranked precision@{K}: {ranked_prec}")

                maps.append(str(map))
                ndcgs.append(str(ndcg))
                ranked_precs.append(str(ranked_prec))

            logger.info(f"map result: {'|'.join(maps)}")
            logger.info(f"ndcg result: {'|'.join(ndcgs)}")
            logger.info(f"ranked_prec result: {'|'.join(ranked_precs)}")

            if best_loss > val_loss:
                prev_best_loss = best_loss
                best_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience = args.patience
                torch.save(
                    model.state_dict(),
                    str(os.path.join(args.result_path, FileName.WEIGHT.value)),
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
                str(os.path.join(args.result_path, FileName.WEIGHT.value)),
            )
            logger.info("Save final model")
    except:
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    from tools.parse_args import parse_args

    args = parse_args()
    main(args)
