from __future__ import annotations

import hydra
from omegaconf import DictConfig

from data import load_and_prepare_lightgbm_data
from model.boosting import LightGBMTrainer


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):

    # load dataset
    X_train, y_train, X_test, y_test = load_and_prepare_lightgbm_data(
        test_size=cfg.data.test_size,
        min_reviews=cfg.data.min_reviews,
        X_columns=cfg.data.X_columns,
        y_columns=cfg.data.y_columns,
    )

    # train model
    trainer = LightGBMTrainer(cfg)

    # train model
    ranker = trainer.fit(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    _main()
