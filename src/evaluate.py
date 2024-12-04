from __future__ import annotations

from pathlib import Path

import hydra
import lightgbm as lgb
import pandas as pd
from omegaconf import DictConfig
from prettytable import PrettyTable

from data import load_train_dataset
from evaluation import hit_at_k, recall_at_k


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):

    # load dataset
    X_train, _, X_test, _ = load_train_dataset(cfg)
    # dataset = pd.concat([X_train, X_test])

    # train model
    ranker = lgb.Booster(model_file=Path(cfg.models.model_path) / f"{cfg.models.results}.txt")

    # evaluate model
    ground_truth = X_train.groupby("reviewer_id")["diner_category_middle"].apply(list)

    X_test["preds"] = ranker.predict(X_test[cfg.data.features])
    X_test = X_test.sort_values(by=["reviewer_id", "preds"], ascending=[True, False])
    recommendations = X_test.groupby(["reviewer_id"])["diner_category_middle"].apply(list)

    # print results
    table = PrettyTable()
    table.field_names = ["K", "Hit@K", "Recall@K"]

    for k in [1, 3, 5, 10]:
        hit = hit_at_k(ground_truth, recommendations, k)
        recall = recall_at_k(ground_truth, recommendations, k)
        table.add_row([k, hit, recall])

    print(table)


if __name__ == "__main__":
    _main()
