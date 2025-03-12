from abc import ABC, abstractmethod
from typing import Any, Dict, List, Self

import pandas as pd


class BaseFeatureStore(ABC):
    def __init__(
        self: Self, review: pd.DataFrame, diner: pd.DataFrame, feature_param_pair: Dict[str, Dict[str, Any]]
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

    @abstractmethod
    def get_engineered_features(self) -> pd.DataFrame:
        """
        Get engineered features only without original features with primary key.

        Returns (pd.DataFrame):
            Engineered features dataframe.
        """
        raise NotImplementedError