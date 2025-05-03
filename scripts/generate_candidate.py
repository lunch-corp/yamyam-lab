import argparse
import importlib
import logging
import os
import pickle
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from datetime import datetime

import torch

from preprocess.preprocess import prepare_networkx_undirected_graph
from tools.config import load_yaml
from tools.google_drive import GoogleDriveManager
from tools.zip import zip_files_in_directory

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/embedding/{model}.yaml")
RESULT_PATH = os.path.join(ROOT_PATH, "./result/{test}/{model}/{dt}")
ZIP_PATH = os.path.join(ROOT_PATH, "./zip/{test}/{model}/{dt}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=["node2vec", "metapath2vec", "graphsage"]
    )
    parser.add_argument("--embedding_dim", type=int, required=True)
    parser.add_argument("--data_obj_path", type=str, required=True)
    parser.add_argument("--model_pt_path", type=str, required=True)
    parser.add_argument("--weighted_edge", action="store_true")
    parser.add_argument("--use_metadata", action="store_true")
    parser.add_argument("--num_sage_layers", type=int, required=False, default=2)
    parser.add_argument(
        "--aggregator_funcs", type=str, nargs="*", default=["mean", "mean"]
    )
    parser.add_argument("--num_neighbor_samples", type=int, default=3)
    parser.add_argument("--candidate_top_k", type=int, required=False, default=100)
    parser.add_argument("--device", type=str, required=False, default="cpu")
    parser.add_argument("--reusable_token_path", type=str, required=False, default="")
    parser.add_argument("--upload_candidate_to_google_drive", action="store_true")
    return parser.parse_args()


def main(args):
    config = load_yaml(CONFIG_PATH.format(model=args.model))
    if args.upload_candidate_to_google_drive and args.reusable_token_path == "":
        raise ValueError(
            "path for google credential token.json should be specified in reusable_token_path argument."
        )
    file_name = config.post_training.file_name
    data = pickle.load(open(args.data_obj_path, "rb"))
    # Note: train_graph, val_graph will be loaded from saved files.
    # At this time, we do not save those graphs, so we instead make it using the same training pipeline.
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
    num_nodes = data["num_users"] + data["num_diners"]
    if args.model == "metapath2vec":
        num_nodes += data["num_metas"]

    model_path = f"embedding.{args.model}"
    model_module = importlib.import_module(model_path).Model
    model = model_module(
        user_ids=torch.tensor(list(data["user_mapping"].values())),
        diner_ids=torch.tensor(list(data["diner_mapping"].values())),
        graph=train_graph,
        embedding_dim=args.embedding_dim,
        walk_length=1,  # dummay value
        walks_per_node=1,  # dummay value
        num_nodes=num_nodes,
        num_negative_samples=1,  # dummay value
        q=1,  # dummay value
        p=1,  # dummay value
        top_k_values=[1],  # dummay value
        model_name=args.model,
        device="cpu",
        recommend_batch_size=2000,
        num_workers=4,  # can be tuned based on server spec
        meta_path=[],  # metapath2vec parameter
        num_layers=args.num_sage_layers,  # graphsage parameter
        aggregator_funcs=args.aggregator_funcs,  # graphsage parameter
        num_neighbor_samples=args.num_neighbor_samples,  # graphsage parameter
        user_raw_features=data["user_feature"],  # graphsage parameter
        diner_raw_features=data["diner_feature"],  # graphsage parameter
        inference=True,
    )

    model.load_state_dict(
        torch.load(
            args.model_pt_path,
            weights_only=True,
            map_location=torch.device(args.device),
        )
    )
    model.eval()
    logging.info("Done loading pre-trained weights")

    # generate candidates and zip related files
    candidates_df = model.generate_candidates_for_each_user(
        top_k_value=args.candidate_top_k
    )
    logging.info("Done generating candidates from pre-trained weights")

    test_flag = "untest"
    dt = datetime.now().strftime("%Y%m%d%H%M")
    zip_path = ZIP_PATH.format(test=test_flag, model=args.model, dt=dt)
    os.makedirs(zip_path, exist_ok=True)
    # save files to zip
    pickle.dump(
        data["user_mapping"],
        open(os.path.join(zip_path, file_name.user_mapping), "wb"),
    )
    pickle.dump(
        data["diner_mapping"],
        open(os.path.join(zip_path, file_name.diner_mapping), "wb"),
    )
    candidates_df.to_parquet(os.path.join(zip_path, file_name.candidate), index=False)
    # zip file
    zip_files_in_directory(
        dir_path=zip_path,
        zip_file_name=f"{dt}.zip",
        allowed_type=[".pkl", ".parquet"],
        logger=logging,
    )

    if args.upload_candidate_to_google_drive:
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
        logging.info(
            f"Successfully uploaded candidate results to google drive.File id: {file_id}"
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
