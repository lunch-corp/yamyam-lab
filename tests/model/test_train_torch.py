import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src"))

from train_torch import main


def run_model(model):
    args = argparse.ArgumentParser()
    args.model = model
    args.batch_size = 128
    args.lr = 0.01
    args.regularization = 1e-4
    args.epochs = 1
    args.num_factors = 32
    args.test_ratio = 0.3
    args.random_state = 42
    args.patience = 5
    args.result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../result/{args.model}")
    args.test = True

    main(args)


def test_run_svd_bias():
    run_model("svd_bias")