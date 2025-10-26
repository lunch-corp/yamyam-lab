import copy
import importlib
import os
import pickle
import traceback
from argparse import ArgumentParser

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from yamyam_lab.data.config import DataConfig
from yamyam_lab.data.graph import GraphDatasetLoader
from yamyam_lab.evaluation.metric_calculator import EmbeddingMetricCalculator
from yamyam_lab.tools.config import generate_result_path, load_configs
from yamyam_lab.tools.logger import (
    logging_data_statistics,
    logging_experiment_config,
    setup_logger,
)
from yamyam_lab.tools.parse_args import parse_args_graph, save_command_to_file
from yamyam_lab.tools.plot import plot_metric_at_k


def main(args: ArgumentParser.parse_args) -> None:
    # load configs
    config, preprocess_config = load_configs(args.model, args.config_root_path)
    result_path = generate_result_path(args.model, args.test, args.result_path)
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
        logging_experiment_config(logger, args, result_path)

        data_loader = GraphDatasetLoader(
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
                use_unique_mapping_id=True,
                test=args.test,
            ),
        )
        data = data_loader.prepare_graph_dataset(
            is_networkx_graph=True,
            use_metadata=args.use_metadata,
            weighted_edge=args.weighted_edge,
            filter_config=preprocess_config.filter,
        )
        train_graph, val_graph = data["train_graph"], data["val_graph"]  # noqa

        logging_data_statistics(
            config=config,
            data=data,
            logger=logger,
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
        model_path = f"yamyam_lab.model.graph.{args.model}"
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
            num_layers=args.num_lightgcn_layers,  # lightgcn parameter
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

        best_val_ndcg = -float("inf")
        best_val_ndcg_epoch = -1
        early_stopping = False
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
                    X_train=data["X_train"],
                    X_val_warm_users=data["X_val_warm_users"],
                    X_val_cold_users=data["X_val_cold_users"],
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

            # we track last validation ndcg@3,
            # because we do not calculate validation loss
            val_ndcg = model.metric_at_k_total_epochs[3]["ndcg"][-1]

            if val_ndcg == 0:
                logger.info(
                    "Validation ndcg@3 is still ZERO... Going to train again..."
                )
                continue

            # when validation ndcg@3 is greater than 0 and previous best value
            if best_val_ndcg < val_ndcg:
                best_val_ndcg_epoch = epoch
                prev_best_val_ndcg = best_val_ndcg
                best_val_ndcg = round(val_ndcg, 6)
                best_model_weights = copy.deepcopy(model.state_dict())
                patience = args.patience
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
                logger.info(
                    f"Best validation ndcg@3: {best_val_ndcg} at epoch {best_val_ndcg_epoch}, Previous best validation ndcg@3: {prev_best_val_ndcg}"
                )
                logger.info(f"successfully saved {args.model} model: epoch {epoch}")
            else:
                patience -= 1
                logger.info(
                    f"Validation ndcg@3 did not decrease. Patience {patience} left."
                )
                if patience == 0:
                    logger.info(
                        f"Patience over. Early stopping at epoch {epoch} with {best_val_ndcg} validation ndcg@3"
                    )
                    early_stopping = True

            # if patience is over, we early stop training
            if early_stopping:
                break

        # Load the best model weights with highest validation ndcg@3
        model.load_state_dict(best_model_weights)
        logger.info("Load weight with best validation ndcg@3")

        torch.save(
            model.state_dict(),
            str(os.path.join(result_path, file_name.weight)),
        )
        logger.info("Save final model with best validation ndcg@3")

        # calculate metric for test data with warm / cold / all users separately
        metric_dict_test = (
            metric_calculator.generate_recommendations_and_calculate_metric(
                X_train=data["X_train"],
                X_val_warm_users=data["X_test_warm_users"],
                X_val_cold_users=data["X_test_cold_users"],
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

    except:
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    args = parse_args_graph()
    main(args)
