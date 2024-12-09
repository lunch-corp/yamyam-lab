from __future__ import annotations

from pathlib import Path

import hydra
import lightgbm as lgb
import numpy as np
from omegaconf import DictConfig
from prettytable import PrettyTable

from data import load_test_dataset
from evaluation import ranking_metrics_at_k


@hydra.main(config_path="../config/", config_name="predict", version_base="1.2.0")
def _main(cfg: DictConfig):
    test, already_reviewed = load_test_dataset(cfg)
    X_test = test[cfg.data.features]

    ranker = lgb.Booster(model_file=Path(cfg.models.model_path) / f"{cfg.models.results}.model")
    predictions = ranker.predict(X_test)
    test["prediction"] = predictions
    test = test.sort_values(by=["prediction"], ascending=False)
    candidates = test.groupby("reviewer_id")["diner_category_middle"].apply(list).values[0]

    if len(already_reviewed) < len(candidates):
        append_list = np.full(fill_value=-1, shape=(len(candidates) - len(already_reviewed)))
        already_reviewed.extend(list(append_list))

    # Evaluate metrics
    top_k = [3, 5, 10, 20]
    tabel = PrettyTable()
    tabel.field_names = ["k", "Recall", "mAP", "nDCG"]

    for k in top_k:
        metrics = ranking_metrics_at_k(np.array(already_reviewed)[:k], np.array(candidates)[:k])
        tabel.add_row([k, metrics["recall"], metrics["ap"], metrics["ndcg"]])

    print(tabel)


if __name__ == "__main__":
    _main()
