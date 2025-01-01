from __future__ import annotations

import glob
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

# Load data (same as your current implementation)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
REVIEW_DATA_PATHS = glob.glob(os.path.join(DATA_PATH, "review", "*.csv"))


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    # load data
    diner = pd.read_csv(os.path.join(DATA_PATH, "diner/diner_df_20241219_yamyam.csv"))

    review = pd.concat(
        [pd.read_csv(review_data_path) for review_data_path in REVIEW_DATA_PATHS]
    )
    review["reviewer_review_cnt"] = review["reviewer_review_cnt"].apply(
        lambda x: np.int32(str(x).replace(",", ""))
    )
    review = pd.merge(review, diner, on="diner_idx", how="inner")
    review = review.drop_duplicates(subset=["reviewer_id", "diner_idx"])

    review["target"] = np.where(
        (review["real_good_review_percent"] > review["real_bad_review_percent"])
        & (review["reviewer_review_score"] - review["reviewer_avg"] > 0.5),
        1,
        0,
    )

    review = review.sort_values(by="reviewer_id")

    label = review[review["target"] == 1]

    label = label.groupby("reviewer_id")["diner_idx"].apply(list).reset_index()
    label.to_csv(Path(DATA_PATH) / "label.csv", index=False)


if __name__ == "__main__":
    _main()
