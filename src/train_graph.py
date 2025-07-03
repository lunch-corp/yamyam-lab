import importlib
import os
import pickle
import traceback
from argparse import ArgumentParser
from datetime import datetime

import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from data.dataset import DataConfig, DatasetLoader
from evaluation.metric_calculator import EmbeddingMetricCalculator
from preprocess.preprocess import prepare_networkx_undirected_graph
from tools.config import load_yaml
from tools.google_drive import GoogleDriveManager
from tools.logger import setup_logger
from tools.parse_args import parse_args_embedding, save_command_to_file
from tools.plot import plot_metric_at_k
from tools.zip import zip_files_in_directory

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/models/graph/{model}.yaml")
RESULT_PATH = os.path.join(ROOT_PATH, "./result/{test}/{model}/{dt}")
ZIP_PATH = os.path.join(ROOT_PATH, "./zip/{test}/{model}/{dt}")


def main(args: ArgumentParser.parse_args) -> None:
    # set result path
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    test_flag = "test" if args.test else "untest"
    result_path = RESULT_PATH.format(test=test_flag, model=args.model, dt=dt)
    os.makedirs(result_path, exist_ok=True)
    # load config
    config = load_yaml(CONFIG_PATH.format(model=args.model))
    # save command used in argparse
    save_command_to_file(result_path)

    # set multiprocessing start method to spawn
    mp.set_start_method("spawn", force=True)

    # predefine config
    top_k_values_for_pred = config.training.evaluation.top_k_values_for_pred
    top_k_values_for_candidate = config.training.evaluation.top_k_values_for_candidate
    file_name = config.post_training.file_name
    fe = config.preprocess.feature_engineering

    logger = setup_logger(os.path.join(result_path, file_name.log))

    try:
        logger.info(f"embedding model: {args.model}")
        logger.info(f"device: {args.device}")
        logger.info(f"batch size: {args.batch_size}")
        logger.info(f"learning rate: {args.lr}")
        logger.info(f"epochs: {args.epochs}")
        logger.info(f"embedding dimension: {args.embedding_dim}")
        logger.info(f"walks per node: {args.walks_per_node}")
        logger.info(f"walk length: {args.walk_length}")
        logger.info(f"num neg samples: {args.num_negative_samples}")
        logger.info(f"weighted edge: {args.weighted_edge}")
        if args.model == "node2vec":
            logger.info(f"p: {args.p}")
            logger.info(f"q: {args.q}")
        elif args.model == "metapath2vec":
            logger.info(f"defined meta_path: {args.meta_path}")
            logger.info(
                f"category column for node meta: {args.category_column_for_meta}"
            )
        elif args.model == "graphsage":
            logger.info(f"number of sage layers: {args.num_sage_layers}")
            logger.info(f"aggregator functions: {args.aggregator_funcs}")

        elif args.model == "lightgcn":
            logger.info(f"number of layers: {args.num_layers}")
            logger.info(f"drop ratio: {args.drop_ratio}")

        logger.info(f"result path: {result_path}")
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
                category_column_for_meta=args.category_column_for_meta,
                user_engineered_feature_names=fe.user_engineered_feature_names,
                diner_engineered_feature_names=fe.diner_engineered_feature_names,
                is_timeseries_by_time_point=config.preprocess.data.is_timeseries_by_time_point,
                train_time_point=config.preprocess.data.train_time_point,
                val_time_point=config.preprocess.data.val_time_point,
                test_time_point=config.preprocess.data.test_time_point,
                end_time_point=config.preprocess.data.end_time_point,
                is_graph_model=True,
                test=args.test,
            ),
        )
        data = data_loader.prepare_train_val_dataset(use_metadata=args.use_metadata)
        train_graph, val_graph = prepare_networkx_undirected_graph(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
            diner=data["diner"],
            user_mapping=data["user_mapping"],
            diner_mapping=data["diner_mapping"],
            meta_mapping=data["meta_mapping"] if args.use_metadata else None,
            weighted=args.weighted_edge,
            use_metadata=args.use_metadata,
        )

        logger.info("######## Number of reviews statistics ########")
        logger.info(f"Number of reviews in train: {data['X_train'].size(0)}")
        logger.info(f"Number of reviews in val: {data['X_val'].size(0)}")
        logger.info(f"Number of reviews in test: {data['X_test'].size(0)}")

        logger.info("######## Train data statistics ########")
        logger.info(f"Number of users in train: {len(data['train_user_ids'])}")
        logger.info(f"Number of diners in train: {len(data['train_diner_ids'])}")
        logger.info(f"Number of feedbacks in train: {data['X_train'].size(0)}")
        train_density = round(
            100
            * data["X_train"].size(0)
            / (len(data["train_user_ids"]) * len(data["train_diner_ids"])),
            4,
        )
        logger.info(f"Train data density: {train_density}%")

        logger.info("######## Validation data statistics ########")
        logger.info(f"Number of users in val: {len(data['val_user_ids'])}")
        logger.info(f"Number of diners in val: {len(data['val_diner_ids'])}")
        logger.info(f"Number of feedbacks in val: {data['X_val'].size(0)}")
        val_density = round(
            100
            * data["X_val"].size(0)
            / (len(data["val_user_ids"]) * len(data["val_diner_ids"])),
            4,
        )
        logger.info(f"Validation data density: {val_density}%")

        logger.info("######## Test data statistics ########")
        logger.info(f"Number of users in test: {len(data['test_user_ids'])}")
        logger.info(f"Number of diners in test: {len(data['test_diner_ids'])}")
        logger.info(f"Number of feedbacks in test: {data['X_test'].size(0)}")
        test_density = round(
            100
            * data["X_test"].size(0)
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
            f"Number of warm start users in test: {len(data['test_warm_start_user_ids'])}"
        )
        logger.info(
            f"Number of cold start users in test: {len(data['test_cold_start_user_ids'])}"
        )

        # for qualitative eval
        pickle.dump(data, open(os.path.join(result_path, file_name.data_object), "wb"))

        num_nodes = data["num_users"] + data["num_diners"]
        if args.model == "metapath2vec":
            num_nodes += data["num_metas"]
        top_k_values = (
            config.training.evaluation.top_k_values_for_pred
            + config.training.evaluation.top_k_values_for_candidate
        )

        # import embedding module
        model_path = f"model.graph.{args.model}"
        model_module = importlib.import_module(model_path).Model
        model = model_module(
            user_ids=torch.tensor(list(data["user_mapping"].values())).to(args.device),
            diner_ids=torch.tensor(list(data["diner_mapping"].values())).to(
                args.device
            ),
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
            device=args.device,
            recommend_batch_size=config.training.evaluation.recommend_batch_size,
            num_workers=4,  # can be tuned based on server spec
            meta_path=args.meta_path,  # metapath2vec parameter
            num_sage_layers=args.num_sage_layers,  # graphsage parameter
            aggregator_funcs=args.aggregator_funcs,  # graphsage parameter
            num_neighbor_samples=args.num_neighbor_samples,  # graphsage parameter
            user_raw_features=data["user_feature"].to(
                args.device
            ),  # graphsage parameter
            diner_raw_features=data["diner_feature"].to(
                args.device
            ),  # graphsage parameter
            num_layers=args.num_layers,  # lightgcn parameter
            drop_ratio=args.drop_ratio,  # lightgcn parameter
        ).to(args.device)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)

        # metric calculator for validation data
        metric_calculator = EmbeddingMetricCalculator(
            diner_ids=list(data["diner_mapping"].values()),
            top_k_values=top_k_values,
            all_embeds=model._embedding
            if args.model in ["graphsage", "lightgcn"]
            else model._embedding.weight,
            filter_already_liked=True,
            recommend_batch_size=config.training.evaluation.recommend_batch_size,
            device=args.device,
            logger=logger,
        )

        loader = model.loader(
            batch_size=args.batch_size,
            shuffle=True,
        )
        for epoch in range(args.epochs):
            logger.info(f"################## epoch {epoch} ##################")
            total_loss = 0
            batch_len = len(loader)
            for batch_idx, (pos_rw, neg_rw) in enumerate(loader):
                optimizer.zero_grad()
                loss = model.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if batch_idx % 500 == 0:
                    logger.info(f"current batch index: {batch_idx} out of {batch_len}")

            # when training graphsage or lightgcn for every epoch,
            # propagation should be run to store embeddings for each node
            if args.model in ["graphsage", "lightgcn"]:
                for batch_nodes in DataLoader(
                    torch.tensor([node for node in train_graph.nodes()]),
                    batch_size=args.batch_size,
                    shuffle=True,
                ):
                    model.propagate_and_store_embedding(batch_nodes.to(args.device))

            total_loss /= len(loader)
            model.tr_loss.append(total_loss)

            logger.info(f"epoch {epoch}: train loss {total_loss:.4f}")

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
            logger.info(f"successfully saved node2vec torch model: epoch {epoch}")

        # calculate metric for test data with warm / cold / all users separately
        metric_dict_test = (
            metric_calculator.generate_recommendations_and_calculate_metric(
                X_train=pd.DataFrame(
                    data["X_train"], columns=["diner_idx", "reviewer_id"]
                ),
                X_val_warm_users=pd.DataFrame(
                    data["X_test_warm_users"], columns=["diner_idx", "reviewer_id"]
                ),
                X_val_cold_users=pd.DataFrame(
                    data["X_test_cold_users"], columns=["diner_idx", "reviewer_id"]
                ),
                most_popular_diner_ids=data["most_popular_diner_ids"],
                filter_already_liked=True,
            )
        )

        # for each user type, the metric is not yet averaged but summed, so calculate mean
        for user_type, metric in metric_dict_test.items():
            metric_calculator.calculate_mean_metric(metric)

        # for each user type, report map, ndcg, recall
        logger.info(
            "################################ Test data metric report ################################"
        )
        metric_calculator.report_metric_with_warm_cold_all_users(
            metric_dict=metric_dict_test, data_type="test"
        )

        # plot metrics
        plot_metric_at_k(
            metric=model.metric_at_k_total_epochs,
            tr_loss=model.tr_loss,
            parent_save_path=result_path,
            top_k_values_for_pred=top_k_values_for_pred,
            top_k_values_for_candidate=top_k_values_for_candidate,
        )

        if args.save_candidate:
            # generate candidates and zip related files
            zip_path = ZIP_PATH.format(test=test_flag, model=args.model, dt=dt)
            os.makedirs(zip_path, exist_ok=True)
            candidates_df = model.generate_candidates_for_each_user(
                top_k_value=config.post_training.candidate_generation.top_k
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
                model_name=args.model,
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
    args = parse_args_embedding()
    main(args)
