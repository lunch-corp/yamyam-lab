import argparse
import os
import sys
from typing import List


def parse_nested_list(value: str) -> List[str]:
    # Split the outer list by semicolon and inner lists by comma
    try:
        return [inner_list.split(",") for inner_list in value.split(";")]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid format for nested list: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["svd_bias"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def parse_args_graph():
    parser = argparse.ArgumentParser()
    # common parameter
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["node2vec", "metapath2vec", "graphsage", "lightgcn"],
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--walks_per_node", type=int, default=10)
    parser.add_argument("--num_negative_samples", type=int, default=1)
    parser.add_argument("--weighted_edge", action="store_true")
    parser.add_argument("--use_metadata", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--test", action="store_true")

    # node2vec parameter
    parser.add_argument("--walk_length", type=int, default=20)
    parser.add_argument("--p", type=float, default=1)
    parser.add_argument("--q", type=float, default=1)

    # metapath2vec parameter
    parser.add_argument("--meta_path", type=parse_nested_list, default=[])
    parser.add_argument(
        "--category_column_for_meta", type=str, default="diner_category_large"
    )

    # graphsage parameter
    parser.add_argument("--num_sage_layers", type=int, default=2)
    parser.add_argument(
        "--aggregator_funcs", type=str, nargs="*", default=["mean", "mean"]
    )
    parser.add_argument("--num_neighbor_samples", type=int, default=3)

    # lightgcn parameter
    parser.add_argument("--num_lightgcn_layers", type=int, default=3)
    parser.add_argument("--drop_ratio", type=float, default=0.1)

    # candidate generation parameter for two-stage reco
    parser.add_argument("--save_candidate", action="store_true", required=False)
    parser.add_argument("--reusable_token_path", type=str, required=False)

    return parser.parse_args()


def parse_args_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_obj_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--embedding_dim", type=int, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--user_ids", type=int, nargs="*", required=True)
    # gangnam station: (37.497992, 127.027614)
    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--longitude", type=float, default=None)
    parser.add_argument("--near_dist", type=float, default=0.5)
    return parser.parse_args()


def parse_args_als():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--factors", type=int, default=100)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--calculate_training_loss", action="store_true")
    parser.add_argument("--test", action="store_true")

    # candidate generation parameter for two-stage reco
    parser.add_argument("--save_candidate", action="store_true", required=False)
    parser.add_argument("--reusable_token_path", type=str, required=False)
    return parser.parse_args()


def save_command_to_file(save_path):
    os.makedirs(save_path, exist_ok=True)

    # Get the command that was used to run the script
    command = " ".join(sys.argv)
    full_command = f"poetry run python3 {command}"
    command_file_path = os.path.join(save_path, "command.txt")
    with open(command_file_path, "w") as f:
        f.write(full_command)
