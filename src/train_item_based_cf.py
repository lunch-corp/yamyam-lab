import os
import pickle
import traceback
from argparse import ArgumentParser
from datetime import datetime

from data.config import DataConfig
from data.csr import CsrDatasetLoader
from evaluation.metric_calculator.item_based_metric_calculator import (
    ItemBasedMetricCalculator,
)
from model.classic_cf.item_based import ItemBasedCollaborativeFiltering
from tools.config import load_yaml
from tools.google_drive import GoogleDriveManager
from tools.logger import common_logging, setup_logger
from tools.parse_args import parse_args_item_based_cf, save_command_to_file
from tools.zip import zip_files_in_directory

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/models/classic_cf/item_based.yaml")
PREPROCESS_CONFIG_PATH = os.path.join(ROOT_PATH, "./config/preprocess/preprocess.yaml")
RESULT_PATH = os.path.join(ROOT_PATH, "./result/{test}/{model}/{dt}")
ZIP_PATH = os.path.join(ROOT_PATH, "./zip/{test}/{model}/{dt}")


def main(args: ArgumentParser.parse_args) -> None:
    # set result path
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    test_flag = "test" if args.test else "untest"
    result_path = RESULT_PATH.format(test=test_flag, model="item_based_cf", dt=dt)
    os.makedirs(result_path, exist_ok=True)
    # load config
    config = load_yaml(CONFIG_PATH)
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
        logger.info(f"method: {args.method}")
        logger.info(f"test: {args.test}")
        logger.info(f"use_hybrid: {args.use_hybrid}")
        if args.use_hybrid:
            logger.info(f"cf_weight: {args.cf_weight}")
            logger.info(f"content_weight: {args.content_weight}")
            logger.info(f"embedding_weight: {args.embedding_weight}")
            logger.info(f"pre_trained_model_path: {args.pre_trained_model_path}")
            logger.info(f"embedding_dim: {args.embedding_dim}")
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
                test=args.test,
            ),
        )
        data = data_loader.prepare_csr_dataset(
            is_csr=True,
            filter_config=preprocess_config.filter,
        )

        common_logging(
            config=config,
            data=data,
            logger=logger,
        )

        # Load diner data and embeddings for hybrid mode
        diner_df = None
        item_embeddings = None

        if args.use_hybrid:
            import networkx as nx
            import pandas as pd
            import torch

            from model.graph.node2vec import Model as Node2Vec

            # Load diner data
            try:
                diner_df = pd.read_csv("data/diner.csv", low_memory=False)
                category_df = pd.read_csv("data/diner_category_raw.csv")
                diner_df = pd.merge(diner_df, category_df, on="diner_idx", how="left")
                logger.info(
                    "Successfully loaded diner data for content-based similarity"
                )
            except Exception as e:
                logger.warning(f"Could not load diner data: {e}")

            # Load embeddings
            if args.pre_trained_model_path:
                try:
                    model_node2vec = Node2Vec(
                        user_ids=torch.tensor(list(data["user_mapping"].values())),
                        diner_ids=torch.tensor(list(data["diner_mapping"].values())),
                        embedding_dim=args.embedding_dim,
                        inference=True,
                        top_k_values=[1],
                        graph=nx.Graph(),
                        walks_per_node=1,
                        num_negative_samples=1,
                        num_nodes=len(data["user_mapping"])
                        + len(data["diner_mapping"]),
                        model_name="node2vec",
                        device="cpu",
                        recommend_batch_size=2000,
                        num_workers=4,
                        walk_length=1,
                    )
                    model_node2vec.load_state_dict(
                        torch.load(args.pre_trained_model_path, weights_only=True)
                    )
                    model_node2vec.eval()
                    item_embeddings = model_node2vec._embedding.weight.detach().numpy()[
                        data["num_users"] :
                    ]
                    logger.info(
                        f"Successfully loaded item embeddings with dimension {args.embedding_dim}"
                    )
                except Exception as e:
                    logger.warning(f"Could not load embeddings: {e}")

        # Initialize Item-Based CF model
        model = ItemBasedCollaborativeFiltering(
            user_item_matrix=data["X_train"],
            item_embeddings=item_embeddings,
            user_mapping=data["user_mapping"],
            item_mapping=data["diner_mapping"],
            diner_df=diner_df,
        )

        # define metric calculator for test data metric
        metric_calculator = ItemBasedMetricCalculator(
            diner_ids=list(data["diner_mapping"].values()),
            model=model,
            top_k_values=top_k_values,
            filter_already_liked=True,
            recommend_batch_size=config.training.evaluation.recommend_batch_size,
            method=args.method,
            use_hybrid=args.use_hybrid,
            cf_weight=args.cf_weight if args.use_hybrid else 1.0,
            content_weight=args.content_weight if args.use_hybrid else 0.0,
            embedding_weight=args.embedding_weight if args.use_hybrid else 0.0,
            logger=logger,
        )

        # calculate metric for **validation data** with warm / cold / all users separately
        logger.info(
            "################################ Validation data metric report ################################"
        )
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
        metric_calculator.report_metric_with_warm_cold_all_users(
            metric_dict=metric_dict, data_type="val"
        )

        # calculate metric for **test data** with warm / cold / all users separately
        logger.info(
            "################################ Test data metric report ################################"
        )
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

        if args.save_candidate:
            # generate candidates and zip related files
            zip_path = ZIP_PATH.format(test=test_flag, model="item_based_cf", dt=dt)
            os.makedirs(zip_path, exist_ok=True)

            logger.info("Generating candidates for each user...")
            candidates_list = []

            # Get all user IDs
            user_ids = list(data["user_mapping"].keys())
            top_k = config.post_training.candidate_generation.top_k

            for user_id in user_ids:
                recommendations = model.recommend_for_user(
                    user_id=user_id,
                    top_k=top_k,
                    method=args.method,
                )
                for rec in recommendations:
                    candidates_list.append(
                        {
                            "reviewer_id": user_id,
                            "diner_idx": rec["item_id"],
                            "score": rec["predicted_score"],
                        }
                    )

            import pandas as pd

            candidates_df = pd.DataFrame(candidates_list)

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
            if args.reusable_token_path:
                manager = GoogleDriveManager(
                    reusable_token_path=args.reusable_token_path,
                    reuse_auth_info=True,
                )
                file_id = manager.upload_result(
                    model_name="item_based_cf",
                    file_path=os.path.join(zip_path, f"{dt}.zip"),
                    download_file_type="candidates",
                )
                logger.info(
                    f"Successfully uploaded candidate results to google drive. "
                    f"File id: {file_id}"
                )

    except:
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    args = parse_args_item_based_cf()
    main(args)
