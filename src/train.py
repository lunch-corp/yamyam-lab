from __future__ import annotations

import hydra
from omegaconf import DictConfig

from data import load_and_prepare_lightgbm_data
from model import build_model


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):

    # load dataset
    X_train, y_train, X_test, y_test = load_and_prepare_lightgbm_data(cfg)

    # train model
    trainer = build_model(cfg)

    # train model
    ranker = trainer.fit(X_train, y_train, X_test, y_test)

    # save model
    trainer.save_model(ranker)

    # plot feature importance
    trainer.plot_feature_importance(ranker)


if __name__ == "__main__":
    _main()
