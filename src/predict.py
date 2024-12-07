from __future__ import annotations

from pathlib import Path

import hydra
import lightgbm as lgb
import numpy as np
from omegaconf import DictConfig
from prettytable import PrettyTable

from data import load_test_dataset
from evaluation import ranking_metrics_at_k


def evaluate_metrics_at_k(
    already_reviewed: list[str], candidates: list[str]
) -> PrettyTable:
    if len(already_reviewed) < len(candidates):
        append_list = np.full(
            fill_value=-1, shape=(len(candidates) - len(already_reviewed))
        )
        already_reviewed.extend(list(append_list))

    # Evaluate metrics
    top_k = [3, 5, 10, 20]
    table = PrettyTable()
    table.field_names = ["k", "Recall", "mAP", "nDCG"]

    for k in top_k:
        metrics = ranking_metrics_at_k(
            np.array(already_reviewed)[:k], np.array(candidates)[:k]
        )
        table.add_row([k, metrics["recall"], metrics["ap"], metrics["ndcg"]])

    return table


@hydra.main(config_path="../config/", config_name="predict", version_base="1.3.1")
def _main(cfg: DictConfig):
    test, already_reviewed = load_test_dataset(cfg)
    X_test = test[cfg.data.features]

    ranker = lgb.Booster(
        model_file=Path(cfg.models.model_path) / f"{cfg.models.results}.txt"
    )

    predictions = ranker.predict(X_test)

    test["prediction"] = predictions
    test = test.sort_values(by=["prediction"], ascending=False)

    test = test.head(cfg.top_n)
    candidates = (
        test.groupby("reviewer_id")["diner_category_middle"].apply(list).values[0]
    )

    table = evaluate_metrics_at_k(already_reviewed, candidates)

    print(f"Evaluation metrics:\n{table}")

    table = PrettyTable()
    table.field_names = [
        "diner_name",
        "diner_address_constituency",
        "diner_category_middle",
        "prediction",
    ]

    for _, row in test.iterrows():
        table.add_row(
            [
                row["diner_name"],
                row["diner_address_constituency"],
                row["diner_category_middle"],
                row["prediction"],
            ]
        )

    print(f"{cfg.user_name}님을 위한 추천 식당 리스트를 알립니다.\n{table}")


if __name__ == "__main__":
    _main()
