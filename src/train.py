from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from data import load_train_dataset
from model.boosting import LightGBMTrainer


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):

    # load dataset
    X_train, y_train, X_test, y_test = load_train_dataset(cfg)

    # train model
    trainer = LightGBMTrainer(cfg)

    # train model
    ranker = trainer.fit(X_train, y_train, X_test, y_test)

    # save model
    ranker.save_model(Path(cfg.models.model_path) / f"{cfg.models.results}.txt")


if __name__ == "__main__":
    _main()
