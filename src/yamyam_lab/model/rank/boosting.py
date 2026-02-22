from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Self

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from omegaconf import OmegaConf
from tqdm import tqdm

from yamyam_lab.model.rank.base import BaseModel


class LightGBMTrainer(BaseModel):
    def __init__(
        self,
        model_path: str,
        results: str,
        params: dict[str, Any],
        early_stopping_rounds: int,
        num_boost_round: int,
        verbose_eval: int,
        seed: int,
        features: list[str],
        cat_features: list[str],
    ) -> None:
        super().__init__(
            model_path,
            results,
            params,
            early_stopping_rounds,
            num_boost_round,
            verbose_eval,
            seed,
            features,
        )
        self.cat_features = cat_features

    def _get_groups(self: Self, X_train: pd.DataFrame | np.ndarray) -> np.ndarray:
        return X_train.groupby("reviewer_id").size().to_numpy()

    def _fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> lgb.Booster:
        # set params
        params = OmegaConf.to_container(self.params)
        params["seed"] = self.seed

        train_groups = self._get_groups(X_train)
        valid_groups = self._get_groups(X_valid)

        train_set = lgb.Dataset(
            X_train[self.features],
            y_train,
            params=params,
            group=train_groups,
            categorical_feature=self.cat_features,
            feature_name=self.features,
        )
        valid_set = lgb.Dataset(
            X_valid[self.features],
            y_valid,
            params=params,
            group=valid_groups,
            categorical_feature=self.cat_features,
            feature_name=self.features,
        )

        model = lgb.train(
            params=params,
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            num_boost_round=self.num_boost_round,
            callbacks=[
                lgb.log_evaluation(self.verbose_eval),
                lgb.early_stopping(self.early_stopping_rounds),
            ],
        )

        # save train_set for feature importance
        self.model = model

        return model

    def _predict(self: Self, X_test: pd.DataFrame | np.ndarray) -> np.ndarray:
        self.model = self.load_model()
        with tqdm(total=1, desc="Predicting") as pbar:
            result = self.model.predict(X_test)
            pbar.update(1)
        return result

    def save_model(self: Self) -> None:
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.model.save_model(Path(self.model_path) / f"{self.results}.model")

    def load_model(self: Self) -> lgb.Booster:
        return lgb.Booster(model_file=Path(self.model_path) / f"{self.results}.model")

    def plot_feature_importance(self: Self) -> None:
        importance = self.model.feature_importance(importance_type="gain")
        if sum(importance) == 0:
            # Test code passed
            return

        _, ax = plt.subplots(figsize=(15, 10))
        lgb.plot_importance(self.model, ax=ax)
        plt.savefig(Path(self.model_path) / f"{self.results}_feature_importance.png")


class CatBoostRankerTrainer(BaseModel):
    def __init__(
        self,
        model_path: str,
        results: str,
        params: dict[str, Any],
        early_stopping_rounds: int,
        num_boost_round: int,
        verbose_eval: int,
        seed: int,
        features: list[str],
        cat_features: list[str],
    ) -> None:
        super().__init__(
            model_path,
            results,
            params,
            early_stopping_rounds,
            num_boost_round,
            verbose_eval,
            seed,
            features,
        )
        self.cat_features = cat_features

    def _get_groups(self: Self, X_train: pd.DataFrame | np.ndarray) -> np.ndarray:
        return X_train["reviewer_id"].to_numpy()

    def _prepare_cat_features(self: Self, X: pd.DataFrame) -> pd.DataFrame:
        """CatBoost cat_features는 int/str만 허용. float·NaN은 str로 통일."""
        X = X.copy()
        for col in self.cat_features:
            if col in X.columns:
                X[col] = X[col].astype(int)
        return X

    def _fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> CatBoostRanker:
        params = OmegaConf.to_container(self.params)
        params["random_seed"] = self.seed
        params["iterations"] = self.num_boost_round

        train_groups = self._get_groups(X_train)
        valid_groups = self._get_groups(X_valid)
        X_train = self._prepare_cat_features(X_train)
        X_valid = self._prepare_cat_features(X_valid)

        train_set = Pool(
            X_train[self.features],
            y_train,
            group_id=train_groups,
            cat_features=self.cat_features,
        )
        valid_set = Pool(
            X_valid[self.features],
            y_valid,
            group_id=valid_groups,
            cat_features=self.cat_features,
        )

        model = CatBoostRanker(
            **params,
            cat_features=self.cat_features,
        )
        model.fit(
            train_set,
            eval_set=valid_set,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose_eval,
        )
        self.model = model
        return model

    def save_model(self: Self) -> None:
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model.save_model(Path(self.model_path) / f"{self.results}.cbm")

    def load_model(self: Self) -> CatBoostRanker:
        model = CatBoostRanker()
        model.load_model(Path(self.model_path) / f"{self.results}.cbm")
        return model

    def _predict(self: Self, X_test: pd.DataFrame | np.ndarray) -> np.ndarray:
        self.model = self.load_model()
        if isinstance(X_test, pd.DataFrame):
            X_test = self._prepare_cat_features(X_test)
        with tqdm(total=1, desc="Predicting") as pbar:
            result = self.model.predict(X_test)
            pbar.update(1)
        return result
