from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from prettytable import PrettyTable

from data import load_train_dataset
from evaluation import map_at_k, ndcg_at_k
from model.boosting import LightGBMTrainer


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):

    # load dataset
    X_train, y_train, X_test, y_test = load_train_dataset(cfg)
    dataset = pd.concat([X_train, X_test])

    # train model
    trainer = LightGBMTrainer(cfg)

    # train model
    ranker = trainer.fit(X_train, y_train, X_test, y_test)

    # evaluate model
    ground_truth = dataset.groupby("reviewer_id")["diner_idx"].apply(list)

    X_test["preds"] = ranker.predict(X_test[cfg.data.features])
    X_test = X_test.sort_values(by=["reviewer_id", "preds"], ascending=[True, False])
    recommendations = X_test.groupby(["reviewer_id"])["diner_idx"].apply(list)

    # print results
    table = PrettyTable()
    table.field_names = ["K", "MAP@K", "NDCG@K"]

    for k in [1, 3, 5, 10]:
        map_at_k_score = map_at_k(ground_truth, recommendations, k=k)
        ndcg_at_k_score = ndcg_at_k(ground_truth, recommendations, k=k)
        table.add_row([k, f"{map_at_k_score:.4f}", f"{ndcg_at_k_score:.4f}"])

    print(table)

    # save model
    ranker.save_model(Path(cfg.models.model_path) / f"{cfg.models.results}.txt")


if __name__ == "__main__":
    _main()
