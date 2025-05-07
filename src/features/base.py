from abc import ABC, abstractmethod
from typing import Any, Dict, List, Self

import pandas as pd

from .diner import DinerFeatureStore
from .reviewer import UserFeatureStore


class BaseFeatureStore(ABC):
    def __init__(
        self: Self,
        review: pd.DataFrame,
        diner: pd.DataFrame,
        feature_param_pair: Dict[str, Dict[str, Any]],
    ):
        """
        Feature engineering on diner data or user data.
        This class gets `feature_param_pair` indicating which features to make with corresponding parameters.
        Unimplemented feature name will raise error with `self.feature_methods`.

        Args:
            review (pd.DataFrame): Review data.
            diner (pd.DataFrame): Diner data.
            feature_param_pair (Dict[str, Dict[str, Any]]): Key is name of engineered feature and
                values are its corresponding parameters.
        """
        self.review = review
        self.diner = diner
        self.feature_param_pair = feature_param_pair
        self.feature_methods = {}

    @abstractmethod
    def make_features(self: Self) -> None:
        """
        Feature engineer using `self.features`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def engineered_features(self) -> pd.DataFrame:
        """
        Get engineered features only without original features with primary key.

        Returns (pd.DataFrame):
            Engineered features dataframe.
        """
        raise NotImplementedError


def build_feature(
    review: pd.DataFrame,
    diner: pd.DataFrame,
    all_user_ids: List[int],
    all_diner_ids: List[int],
    user_engineered_feature_names: Dict[str, Dict[str, Any]],
    diner_engineered_feature_names: Dict[str, Dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # user feature engineering
    user_fs = UserFeatureStore(
        review=review,
        diner=diner,
        all_user_ids=all_user_ids,
        feature_param_pair=user_engineered_feature_names,
    )
    user_fs.make_features()
    user_feature = user_fs.engineered_features

    # diner feature engineering
    diner_fs = DinerFeatureStore(
        review=review,
        diner=diner,
        all_diner_ids=all_diner_ids,
        feature_param_pair=diner_engineered_feature_names,
    )
    diner_fs.make_features()
    diner_feature = diner_fs.engineered_features
    diner_meta_feature = diner_fs.engineered_meta_features
    return user_feature, diner_feature, diner_meta_feature
