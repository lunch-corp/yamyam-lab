from __future__ import annotations

from pathlib import Path
from typing import Self

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostRanker, Pool
from omegaconf import DictConfig, OmegaConf

from rank.base import BaseModel


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


class CatBoostTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self: Self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> CatBoostRanker:
        train_groups = X_train["reviewer_id"].to_numpy()
        valid_groups = X_valid["reviewer_id"].to_numpy()

        X_train, X_valid = (
            X_train[self.cfg.data.features],
            X_valid[self.cfg.data.features],
        )

        train_set = Pool(
            X_train,
            y_train,
            cat_features=self.cfg.data.cat_features,
            group_id=train_groups,
        )
        valid_set = Pool(
            X_valid,
            y_valid,
            cat_features=self.cfg.data.cat_features,
            group_id=valid_groups,
        )

        params = OmegaConf.to_container(self.cfg.models.params)
        params["random_seed"] = self.cfg.models.seed

        model = CatBoostRanker(
            **params,
            verbose=self.cfg.models.verbose_eval,
        )

        model.fit(
            train_set,
            eval_set=valid_set,
            verbose_eval=self.cfg.models.verbose_eval,
            early_stopping_rounds=self.cfg.models.early_stopping_rounds,
        )

        # save train_set for feature importance
        self.train_set = train_set

        return model

    def plot_feature_importance(self: Self, model: CatBoostRanker) -> None:
        # 피처 중요도 계산
        importances = model.get_feature_importance(
            type="FeatureImportance", data=self.train_set
        )

        # 중요도 데이터프레임 생성
        feature_importances = pd.DataFrame(
            {
                "Feature": self.cfg.data.features,  # 피처 이름
                "Importance": importances,  # 중요도 값
            }
        )

        # 중요도 내림차순 정렬
        feature_importances = feature_importances.sort_values(
            by="Importance", ascending=False
        )

        # 중요도 시각화
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x="Importance", y="Feature", data=feature_importances, palette="viridis"
        )
        plt.title("CatBoost Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")

        plt.savefig(
            Path(self.cfg.models.model_path)
            / f"{self.cfg.models.results}_feature_importance.png"
        )
