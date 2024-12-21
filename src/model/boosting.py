from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
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
        # set params
        params = OmegaConf.to_container(self.cfg.models.params)
        params["seed"] = self.cfg.models.seed

        # set group
        train_groups = X_train.groupby("reviewer_id").size().to_numpy()
        valid_groups = X_valid.groupby("reviewer_id").size().to_numpy()

        # select features
        X_train, X_valid = (
            X_train[self.cfg.data.features],
            X_valid[self.cfg.data.features],
        )

        train_set = lgb.Dataset(
            X_train,
            y_train,
            params=params,
            group=train_groups,
            categorical_feature=self.cfg.data.cat_features,
            feature_name=self.cfg.data.features,
        )
        valid_set = lgb.Dataset(
            X_valid,
            y_valid,
            params=params,
            group=valid_groups,
            categorical_feature=self.cfg.data.cat_features,
            feature_name=self.cfg.data.features,
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

    def plot_feature_importance(self: Self, model: lgb.Booster) -> None:
        fig, ax = plt.subplots(figsize=(15, 10))
        lgb.plot_importance(model, ax=ax)
        plt.savefig(
            Path(self.cfg.models.model_path)
            / f"{self.cfg.models.results}_feature_importance.png"
        )


class XGBoostTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> xgb.Booster:
        params = OmegaConf.to_container(self.cfg.models.params)
        params["seed"] = self.cfg.models.seed

        # set group
        train_group = X_train.groupby("reviewer_id").size().to_numpy()
        valid_group = X_valid.groupby("reviewer_id").size().to_numpy()

        # select features
        X_train, X_valid = (
            X_train[self.cfg.data.features],
            X_valid[self.cfg.data.features],
        )

        train_set = xgb.DMatrix(X_train, y_train)
        train_set.set_group(train_group)
        valid_set = xgb.DMatrix(X_valid, y_valid)
        valid_set.set_group(valid_group)

        model = xgb.train(
            params=params,
            dtrain=train_set,
            num_boost_round=self.cfg.models.num_boost_round,
            evals=[(train_set, "train"), (valid_set, "valid")],
            early_stopping_rounds=self.cfg.models.early_stopping_rounds,
            verbose_eval=self.cfg.models.verbose_eval,
        )

        return model

    def plot_feature_importance(self: Self, model: xgb.Booster) -> None:
        fig, ax = plt.subplots(figsize=(15, 10))
        xgb.plot_importance(model, ax=ax)
        plt.savefig(
            Path(self.cfg.models.model_path)
            / f"{self.cfg.models.results}_feature_importance.png"
        )
