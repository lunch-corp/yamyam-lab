from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self, TypeVar

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from tqdm import tqdm

TreeModel = TypeVar("TreeModel", lgb.Booster, xgb.Booster)


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: dict[str, TreeModel]


class BaseModel(ABC):
    def __init__(
        self: Self,
        model_path: str,
        results: str,
        params: dict[str, Any],
        early_stopping_rounds: int,
        num_boost_round: int,
        verbose_eval: int,
        seed: int,
        recommend_batch_size: int,
        features: list[str],
    ) -> None:
        self.model_path = model_path
        self.results = results
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.verbose_eval = verbose_eval
        self.seed = seed
        self.model = None
        self.recommend_batch_size = recommend_batch_size
        self.features = features

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

        with tqdm(
            kfold.split(X), total=self.cfg.data.n_splits, desc="cv", leave=False
        ) as pbar:
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

    def calculate_rank(self: Self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rank for candidates.

        Args:
            candidates (pd.DataFrame): Candidates to calculate rank.

        Returns:
            pd.DataFrame: Candidates with rank.
        """
        predictions = np.zeros(len(candidates))

        num_batches = (
            len(candidates) + self.recommend_batch_size - 1
        ) // self.recommend_batch_size
        for i in tqdm(range(num_batches)):
            start_idx = i * self.recommend_batch_size
            end_idx = min((i + 1) * self.recommend_batch_size, len(candidates))
            batch = candidates[self.features].iloc[start_idx:end_idx]
            predictions[start_idx:end_idx] = self.predict(batch)

        candidates["pred_score"] = predictions
        candidates = candidates.sort_values(
            by=["reviewer_id", "pred_score"], ascending=[True, False]
        )

        return candidates
