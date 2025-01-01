import importlib
import os
import pickle
import traceback
from argparse import ArgumentParser

import torch

from candidate.near import NearCandidateGenerator
from constant.candidate.near import MAX_DISTANCE_KM
from constant.device.device import DEVICE
from constant.evaluation.recommend import (
    TOP_K_VALUES_FOR_CANDIDATE,
    TOP_K_VALUES_FOR_PRED,
)
from constant.metric.metric import Metric, NearCandidateMetric
from constant.preprocess.preprocess import MIN_REVIEWS
from constant.save.file_name import FileName
from preprocess.preprocess import prepare_networkx_data, train_test_split_stratify
from tools.logger import setup_logger
from tools.parse_args import parse_args_embedding
from tools.plot import plot_metric_at_k
from tools.utils import get_num_workers


def main(args: ArgumentParser.parse_args) -> None:
    os.makedirs(args.result_path, exist_ok=True)
    logger = setup_logger(os.path.join(args.result_path, FileName.LOG.value))

    try:
        logger.info(f"embedding model: {args.model}")
        logger.info(f"batch size: {args.batch_size}")
        logger.info(f"learning rate: {args.lr}")
        logger.info(f"epochs: {args.epochs}")
        logger.info(f"embedding dimension: {args.embedding_dim}")
        logger.info(f"walk length: {args.walk_length}")
        logger.info(f"walks per node: {args.walks_per_node}")
        logger.info(f"num neg samples: {args.num_negative_samples}")
        logger.info(f"p: {args.p}")
        logger.info(f"q: {args.q}")
        logger.info(f"result path: {args.result_path}")

        data = train_test_split_stratify(
            test_size=args.test_ratio,
            min_reviews=MIN_REVIEWS,
            X_columns=["diner_idx", "reviewer_id"],
            y_columns=["reviewer_review_score"],
            pg_model=True,
        )
        train_graph, val_graph = prepare_networkx_data(
            X_train=data["X_train"],
            X_val=data["X_val"],
        )

        # for qualitative eval
        pickle.dump(
            data, open(os.path.join(args.result_path, FileName.DATA_OBJECT.value), "wb")
        )

        num_nodes = data["num_users"] + data["num_diners"]
        top_k_values = TOP_K_VALUES_FOR_PRED + TOP_K_VALUES_FOR_CANDIDATE

        # import embedding module
        model_path = f"embedding.{args.model}"
        model_module = importlib.import_module(model_path).Model
        model = model_module(
            user_ids=torch.tensor(list(data["user_mapping"].values())).to(DEVICE),
            diner_ids=torch.tensor(list(data["diner_mapping"].values())).to(DEVICE),
            graph=train_graph,
            embedding_dim=args.embedding_dim,
            walk_length=args.walk_length,
            walks_per_node=args.walks_per_node,
            num_nodes=num_nodes,
            num_negative_samples=args.num_negative_samples,
            q=args.q,
            p=args.p,
            top_k_values=top_k_values,
        ).to(DEVICE)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)

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

        loader = model.loader(
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=get_num_workers(),
        )
        for epoch in range(args.epochs):
            logger.info(f"################## epoch {epoch} ##################")
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(DEVICE), neg_rw.to(DEVICE))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            total_loss /= len(loader)
            model.tr_loss.append(total_loss)

            logger.info(f"epoch {epoch}: train loss {total_loss:.4f}")

            model.recommend_all(
                X_train=data["X_train"],
                X_val=data["X_val"],
                top_k_values=top_k_values,
                nearby_candidates=nearby_candidates_mapping,
                filter_already_liked=True,
            )

            maps = []
            ndcgs = []
            recalls = []
            ranked_precs = []
            candidate_recalls = []

            for k in TOP_K_VALUES_FOR_PRED:
                # no candidate metric
                map = round(model.metric_at_k[k][Metric.MAP.value], 5)
                ndcg = round(model.metric_at_k[k][Metric.NDCG.value], 5)
                recall = round(model.metric_at_k[k][Metric.RECALL.value], 5)
                ranked_prec = round(
                    model.metric_at_k[k][NearCandidateMetric.RANKED_PREC.value], 5
                )
                count = model.metric_at_k[k][Metric.COUNT.value]
                prec_count = model.metric_at_k[k][
                    NearCandidateMetric.RANKED_PREC_COUNT.value
                ]

                logger.info(
                    f"maP@{k}: {map} with {count} users out of all {model.num_users} users"
                )
                logger.info(
                    f"ndcg@{k}: {ndcg} with {count} users out of all {model.num_users} users"
                )
                logger.info(
                    f"recall@{k}: {recall} with {count} users out of all {model.num_users} users"
                )
                logger.info(
                    f"ranked_prec@{k}: {ranked_prec} out of all {prec_count} validation dataset"
                )

                maps.append(str(map))
                ndcgs.append(str(ndcg))
                recalls.append(str(recall))
                ranked_precs.append(str(ranked_prec))

            logger.info(
                "top k results for direct prediction @3, @7, @10, @20 in order"
            )
            logger.info(f"map result: {'|'.join(maps)}")
            logger.info(f"ndcg result: {'|'.join(ndcgs)}")
            logger.info(f"recall: {'|'.join(recalls)}")
            logger.info(f"ranked_prec: {'|'.join(ranked_precs)}")

            for k in TOP_K_VALUES_FOR_CANDIDATE:
                # near candidate metric
                prec_count = model.metric_at_k[k][
                    NearCandidateMetric.RANKED_PREC_COUNT.value
                ]
                near_candidate_recall = round(
                    model.metric_at_k[k][NearCandidateMetric.NEAR_RECALL.value], 5
                )
                recall_count = model.metric_at_k[k][
                    NearCandidateMetric.RECALL_COUNT.value
                ]
                logger.info(
                    f"near_candidate_recall@{k}: {near_candidate_recall} with {recall_count} count out of all {prec_count} validation dataset"
                )
                candidate_recalls.append(str(near_candidate_recall))

            logger.info("top k results for candidate generation @100, @300, @500")
            logger.info(f"candidate_recall: {'|'.join(candidate_recalls)}")

            torch.save(
                model.state_dict(),
                str(os.path.join(args.result_path, FileName.WEIGHT.value)),
            )
            pickle.dump(
                model.tr_loss,
                open(
                    os.path.join(args.result_path, FileName.TRAINING_LOSS.value), "wb"
                ),
            )
            pickle.dump(
                model.metric_at_k_total_epochs,
                open(os.path.join(args.result_path, FileName.METRIC.value), "wb"),
            )
            logger.info(f"successfully saved node2vec torch model: epoch {epoch}")

        # plot metrics
        plot_metric_at_k(
            metric=model.metric_at_k_total_epochs,
            tr_loss=model.tr_loss,
            parent_save_path=args.result_path,
        )

    except:
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    args = parse_args_embedding()
    main(args)
