try:
    import os
    import sys

    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src")
    )

except ModuleNotFoundError:
    raise Exception("Module not found")

import argparse

from train_embedding import main


def run_model(model):
    args = argparse.ArgumentParser()
    args.model = model
    args.batch_size = 128
    args.lr = 0.01
    args.epochs = 1
    args.test_ratio = 0.3
    args.embedding_dim = 32
    args.walk_length = 20
    args.walks_per_node = 10
    args.num_negative_samples = 20
    args.p = 1
    args.q = 0.5
    args.result_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"../result/{args.model}"
    )
    args.weighted_edge = True
    args.use_metadata = False
    args.test = True

    main(args)


def test_run_node2vec():
    run_model("node2vec")
