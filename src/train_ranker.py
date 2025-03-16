from __future__ import annotations

import hydra
from omegaconf import DictConfig

from model.rank import build_model
from preprocess.preprocess import train_test_split_stratify


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def main(cfg: DictConfig):
    # load dataset
    data = train_test_split_stratify(
        test_size=cfg.data.test_size,
        min_reviews=cfg.data.min_reviews,
        is_rank=True,
        diner_engineered_feature_names=cfg.data.diner_engineered_feature_names[0],
    )

    X_train, y_train, X_test, y_test = (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
    )

    # train model
    trainer = build_model(cfg)

    # train model
    trainer.fit(X_train, y_train, X_test, y_test)

    # save model
    trainer.save_model()

    # plot feature importance
    trainer.plot_feature_importance()


if __name__ == "__main__":
    main()
