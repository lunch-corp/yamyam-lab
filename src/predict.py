from __future__ import annotations

from pathlib import Path

import hydra
import lightgbm as lgb
from omegaconf import DictConfig
from prettytable import PrettyTable

from data import load_test_dataset


@hydra.main(config_path="../config/", config_name="predict", version_base="1.3.1")
def _main(cfg: DictConfig):
    test, already_reviewed = load_test_dataset(cfg)
    X_test = test[cfg.data.features]

    ranker = lgb.Booster(
        model_file=Path(cfg.models.model_path) / f"{cfg.models.results}.model"
    )

    predictions = ranker.predict(X_test)

    test["prediction"] = predictions
    test = test.sort_values(by=["prediction"], ascending=False)
    test = test.head(cfg.top_n)

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
