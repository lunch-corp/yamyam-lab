"""Unified training entry point using Factory and Template Method patterns."""

import argparse
import sys

from yamyam_lab.engine import TrainerFactory
from yamyam_lab.tools.parse_args import (
    parse_args,
    parse_args_als,
    parse_args_graph,
)


def main(args: argparse.Namespace):
    """Main training entry point."""
    # Create appropriate trainer using Factory pattern
    trainer = TrainerFactory.create_trainer(args)

    # Train using Template Method pattern
    trainer.train()


if __name__ == "__main__":
    # Check if specific model argument is provided
    if len(sys.argv) > 1 and any(arg.startswith("--model") for arg in sys.argv):
        model = None
        for i, arg in enumerate(sys.argv):
            if arg.startswith("--model"):
                if "=" in arg:
                    model = arg.split("=")[1]
                elif i + 1 < len(sys.argv):
                    model = sys.argv[i + 1]
                break

        # Parse arguments based on model type
        if model in ["node2vec", "graphsage", "metapath2vec", "lightgcn"]:
            args = parse_args_graph()
        elif model == "als":
            args = parse_args_als()
        elif model in ["svd_bias"]:
            args = parse_args()
        else:
            # Default to ALS parser for unknown models
            args = parse_args_als()
    else:
        # Default to ALS
        args = parse_args_als()

    main(args)
