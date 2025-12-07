import argparse
import importlib
import logging
import os
import pickle
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from datetime import datetime

import torch

from yamyam_lab.tools.config import load_yaml
from yamyam_lab.tools.google_drive import GoogleDriveManager
from yamyam_lab.tools.zip import zip_files_in_directory

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/models/{model_type}/{model}.yaml")
RESULT_PATH = os.path.join(ROOT_PATH, "./result/{test}/{model}/{dt}")
ZIP_PATH = os.path.join(ROOT_PATH, "./zip/{test}/{model}/{dt}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["node2vec", "metapath2vec", "graphsage", "lightgcn"],
    )
    parser.add_argument("--data_obj_path", type=str, required=True)
    parser.add_argument("--model_pt_path", type=str, required=True)
    parser.add_argument("--candidate_top_k", type=int, required=False, default=100)
    parser.add_argument("--reusable_token_path", type=str, required=False, default="")
    parser.add_argument("--upload_candidate_to_google_drive", action="store_true")
    return parser.parse_args()


def main(args):
    # set model type
    model_type = (
        "graph"
        if args.model in ["node2vec", "metapath2vec", "graphsage", "lightgcn"]
        else "mf"
    )
    config = load_yaml(CONFIG_PATH.format(model=args.model, model_type=model_type))
    if args.upload_candidate_to_google_drive and args.reusable_token_path == "":
        raise ValueError(
            "path for google credential token.json should be specified in reusable_token_path argument."
        )
    file_name = config.post_training.file_name
    data = pickle.load(open(args.data_obj_path, "rb"))

    # load model module
    model_path = f"yamyam_lab.model.{model_type}.{args.model}"
    model_module = importlib.import_module(model_path).Model

    checkpoint = torch.load(args.model_pt_path, map_location="cpu")
    config = checkpoint["config"]
    config.inference = True  # inference mode, not training
    config.device = "cpu"  # force cpu mode even if originally set to cuda

    # Initialize model with config
    model = model_module(config=config).to("cpu")

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
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
