from __future__ import annotations

from pathlib import Path

import hydra
import lightgbm as lgb
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from data import load_and_prepare_lightgbm_data
from model.boosting import LightGBMTrainer


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):

    # load dataset
    X_train, y_train, X_test, y_test = load_and_prepare_lightgbm_data(cfg)

    # train model
    trainer = LightGBMTrainer(cfg)

    # train model
    ranker = trainer.fit(X_train, y_train, X_test, y_test)

    # save model
    ranker.save_model(Path(cfg.models.model_path) / f"{cfg.models.results}.model")

    # save feature importance
    fig, ax = plt.subplots(figsize=(15, 10))
    lgb.plot_importance(ranker, ax=ax)
    plt.savefig(
        Path(cfg.models.model_path) / f"{cfg.models.results}_feature_importance.png"
    )


if __name__ == "__main__":
    _main()
