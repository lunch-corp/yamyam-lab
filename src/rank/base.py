from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self, TypeVar

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from tqdm import tqdm

TreeModel = TypeVar("TreeModel", lgb.Booster, xgb.Booster)


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: dict[str, TreeModel]


class BaseModel(ABC):
    def __init__(self: Self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.model = None

    @abstractmethod
    def save_model(self: Self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_model(self: Self) -> TreeModel:
        # return model
        raise NotImplementedError

    @abstractmethod
    def _fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ):
        raise NotImplementedError

    def fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> TreeModel:
        model = self._fit(X_train, y_train, X_valid, y_valid)

        return model

    @abstractmethod
    def _predict(self: Self, X: pd.DataFrame | np.ndarray):
        raise NotImplementedError

    def predict(self: Self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self._predict(X)

    def run_cv_training(self: Self, X: pd.DataFrame, y: pd.Series) -> Self:
        oof_preds = np.zeros(X.shape[0])
        models = {}
        kfold = KFold(
            n_splits=self.cfg.data.n_splits,
            shuffle=True,
            random_state=self.cfg.data.seed,
        )

        with tqdm(kfold.split(X), total=self.cfg.data.n_splits, desc="cv", leave=False) as pbar:
            for fold, (train_idx, valid_idx) in enumerate(pbar, 1):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

                model = self.fit(X_train, y_train, X_valid, y_valid)
                oof_preds[valid_idx] = self.predict(X_valid)
                models[f"fold_{fold}"] = model

        del X_train, X_valid, y_train, y_valid
        gc.collect()

        self.result = ModelResult(oof_preds=oof_preds, models=models)

        return self
