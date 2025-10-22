from typing import Any, Dict, List

import pandas as pd

from .base import BaseFeatureStore
from .diner import DinerFeatureStore
from .reviewer import UserFeatureStore

__all__ = ["BaseFeatureStore", "DinerFeatureStore", "UserFeatureStore"]


def build_feature(
    review: pd.DataFrame,
    diner: pd.DataFrame,
    all_user_ids: List[int],
    all_diner_ids: List[int],
    user_engineered_feature_names: Dict[str, Dict[str, Any]],
    diner_engineered_feature_names: Dict[str, Dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build features for user and diner.

    Args:
        review (pd.DataFrame): Review data.
        diner (pd.DataFrame): Diner data.
        all_user_ids (List[int]): All user ids.
        all_diner_ids (List[int]): All diner ids.
        user_engineered_feature_names (Dict[str, Dict[str, Any]]): User engineered feature names.
        diner_engineered_feature_names (Dict[str, Dict[str, Any]]): Diner engineered feature names.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: User feature, diner feature, diner meta feature.
    """
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
