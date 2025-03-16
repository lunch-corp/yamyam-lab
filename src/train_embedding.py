import importlib
import os
import pickle
import traceback
from argparse import ArgumentParser
from datetime import datetime

import torch
from torch.utils.data import DataLoader

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
from preprocess.preprocess import (
    prepare_networkx_undirected_graph,
    train_test_split_stratify,
)
from tools.config import load_yaml
from tools.google_drive import GoogleDriveManager
from tools.logger import setup_logger
from tools.parse_args import parse_args_embedding
from tools.plot import plot_metric_at_k
from tools.zip import zip_files_in_directory

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/data/embedding.yaml")
ZIP_PATH = os.path.join(ROOT_PATH, "./zip")


def main(args: ArgumentParser.parse_args) -> None:
    os.makedirs(args.result_path, exist_ok=True)
    logger = setup_logger(os.path.join(args.result_path, FileName.LOG.value))
    config = load_yaml(CONFIG_PATH)

    try:
        logger.info(f"embedding model: {args.model}")
        logger.info(f"batch size: {args.batch_size}")
        logger.info(f"learning rate: {args.lr}")
        logger.info(f"epochs: {args.epochs}")
        logger.info(f"embedding dimension: {args.embedding_dim}")
        logger.info(f"walks per node: {args.walks_per_node}")
        logger.info(f"num neg samples: {args.num_negative_samples}")
        logger.info(f"weighted edge: {args.weighted_edge}")
        if args.model == "node2vec":
            logger.info(f"walk length: {args.walk_length}")
            logger.info(f"p: {args.p}")
            logger.info(f"q: {args.q}")
        elif args.model == "metapath2vec":
            logger.info(f"defined meta_path: {args.meta_path}")
            logger.info(
                f"category column for node meta: {args.category_column_for_meta}"
            )
        elif args.model == "graphsage":
            logger.info(f"number of sage layers: {args.num_sage_layers}")
        logger.info(f"result path: {args.result_path}")
        logger.info(f"test: {args.test}")

        data = train_test_split_stratify(
            test_size=args.test_ratio,
            min_reviews=MIN_REVIEWS,
            X_columns=["diner_idx", "reviewer_id"],
            y_columns=["reviewer_review_score"],
            is_graph_model=True,
            category_column_for_meta=args.category_column_for_meta,
            user_engineered_feature_names=config.user_engineered_feature_names,
            diner_engineered_feature_names=config.diner_engineered_feature_names,
            test=args.test,
        )
        train_graph, val_graph = prepare_networkx_undirected_graph(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
            diner=data["diner"],
            user_mapping=data["user_mapping"],
            diner_mapping=data["diner_mapping"],
            meta_mapping=data["meta_mapping"],
            weighted=args.weighted_edge,
            use_metadata=args.use_metadata,
        )

        # for qualitative eval
        pickle.dump(
            data, open(os.path.join(args.result_path, FileName.DATA_OBJECT.value), "wb")
        )

        num_nodes = data["num_users"] + data["num_diners"]
        if args.model == "metapath2vec":
            num_nodes += data["num_metas"]
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
            model_name=args.model,
            meta_path=args.meta_path,  # metapath2vec parameter
            num_layers=args.num_sage_layers,  # graphsage parameter
            user_raw_features=data["user_feature"],  # graphsage parameter
            diner_raw_features=data["diner_feature"],  # graphsage parameter
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

            # when training graphsage for every epoch,
            # propagation should be run to store embeddings for each node at every epoch
            if args.model == "graphsage":
                for batch_nodes in DataLoader(
                    torch.tensor([node for node in train_graph.nodes()]),
                    batch_size=args.batch_size,
                    shuffle=True,
                ):
                    model.propagate_and_store_embedding(batch_nodes)

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
                    f"ranked_prec@{k}: {ranked_prec} out of all {prec_count} validation dataset"
                )

                maps.append(str(map))
                ndcgs.append(str(ndcg))
                ranked_precs.append(str(ranked_prec))

            logger.info("top k results for direct prediction @3, @7, @10, @20 in order")
            logger.info(f"map result: {'|'.join(maps)}")
            logger.info(f"ndcg result: {'|'.join(ndcgs)}")
            logger.info(f"ranked_prec: {'|'.join(ranked_precs)}")

            for k in TOP_K_VALUES_FOR_CANDIDATE:
                recall = round(model.metric_at_k[k][Metric.RECALL.value], 5)
                count = model.metric_at_k[k][Metric.COUNT.value]
                logger.info(
                    f"recall@{k}: {recall} with {count} users out of all {model.num_users} users"
                )

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
                recalls.append(str(recall))

            logger.info(
                "top k results for candidate generation @100, @300, @500, @1000, @2000"
            )
            logger.info(f"recall: {'|'.join(recalls)}")
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

        if args.save_candidate:
            # generate candidates and zip related files
            dt = datetime.now().strftime("%Y%m%d%H%M")
            zip_path = os.path.join(ZIP_PATH, args.model, dt)
            os.makedirs(zip_path, exist_ok=True)
            candidates_df = model.generate_candidates_for_each_user(top_k_value=50)
            # save files to zip
            pickle.dump(
                data["user_mapping"],
                open(os.path.join(zip_path, "user_mapping.pkl"), "wb"),
            )
            pickle.dump(
                data["diner_mapping"],
                open(os.path.join(zip_path, "dimer_mapping.pkl"), "wb"),
            )
            candidates_df.to_parquet(
                os.path.join(zip_path, "candidate.parquet"), index=False
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
            file_id = manager.upload_candidates_result(
                model_name=args.model,
                file_path=os.path.join(zip_path, f"{dt}.zip"),
            )
            logger.info(
                f"Successfully uploaded candidate results to google drive."
                f"File id: {file_id}"
            )

    except:
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    args = parse_args_embedding()
    main(args)
