from .base import *
from .diner import *
from .reviewer import *


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
