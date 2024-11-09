import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_factors", type=int, default=32)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--model_path", type=str, default="./model.pt")
    parser.add_argument("--log_path", type=str, default="./log.log")
    return parser.parse_args()