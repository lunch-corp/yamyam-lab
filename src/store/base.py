from abc import ABC, abstractmethod
from typing import List, Self

import pandas as pd


class BaseFeatureStore(ABC):
    def __init__(
        self: Self, review: pd.DataFrame, diner: pd.DataFrame, features: List[str]
    ):
        """
        Feature engineering on diner data.
        This class gets `features` indicating which features to make.
        Unimplemented feature name will raise error with `self.feature_methods`.

        Args:
            review (pd.DataFrame): Review data.
            diner (pd.DataFrame): Diner data.
            features (List[str]): List of features to make.
        """
        self.review = review
        self.diner = diner
        self.features = features
        self.feature_methods = {}

    @abstractmethod
    def make_features(self: Self) -> None:
        """
        Feature engineer using `self.features`.
        """
        raise NotImplementedError
