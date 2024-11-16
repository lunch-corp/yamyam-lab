from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Self

from model.base import BaseModel


class LightGBMTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> lgb.Booster:
        params = OmegaConf.to_container(self.cfg.models.params)
        params["seed"] = self.cfg.models.seed
        train_groups = X_train.groupby("reviewer_id").size().to_numpy()
        valid_groups = X_valid.groupby("reviewer_id").size().to_numpy()

        X_train = X_train[self.cfg.data.features]
        X_valid = X_valid[self.cfg.data.features]

        train_set = lgb.Dataset(
            X_train, y_train, params=params, group=train_groups, feature_name=self.cfg.data.features
        )
        valid_set = lgb.Dataset(
            X_valid, y_valid, params=params, group=valid_groups, feature_name=self.cfg.data.features
        )

        model = lgb.train(
            params=params,
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            num_boost_round=self.cfg.models.num_boost_round,
            callbacks=[
                lgb.log_evaluation(self.cfg.models.verbose_eval),
                lgb.early_stopping(self.cfg.models.early_stopping_rounds),
            ],
        )

        return model
        