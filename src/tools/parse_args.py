import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["svd_bias"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_factors", type=int, default=32)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def parse_args_embedding():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["node2vec"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--walk_length", type=int, default=20)
    parser.add_argument("--walks_per_node", type=int, default=10)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=float, default=1)
    parser.add_argument("--num_negative_samples", type=int, default=1)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--weighted_edge", action="store_true")
    parser.add_argument("--use_metadata", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()


def parse_args_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_obj_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--embedding_dim", type=int)
    parser.add_argument("--log_path", type=str)
    return parser.parse_args()