try:
    import os
    import sys

    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src")
    )

except ModuleNotFoundError:
    raise Exception("Module not found")

import pandas as pd

from constant.evaluation.qualitative import QualitativeReviewerId
from data.dataset import DatasetLoader, load_test_dataset


def test_loader_dataset():
    data_loader = DatasetLoader(
        test_size=0.3,
        min_reviews=3,
        X_columns=["diner_idx", "reviewer_id"],
        y_columns=["reviewer_review_score"],
        random_state=42,
        stratify="reviewer_id",
        user_engineered_feature_names={
            "categorical_feature_count": {
                "categorical_feature_names": ["diner_category_large"]
            }
        },
        diner_engineered_feature_names={
            "all_review_cnt": {},
            "diner_review_tags": {},
            "diner_menu_price": {},
        },
        test=True,
    )
    data = data_loader.prepare_train_val_dataset()

    assert data["X_train"].shape[0] > 10000
    assert data["X_val"].shape[0] > 10000
    assert data["y_train"].shape[0] > 10000
    assert data["y_val"].shape[0] > 10000
    assert data["num_diners"] > 0
    assert data["num_users"] > 0
    assert data["diner_mapping"] is not None
    assert data["user_mapping"] is not None

    rank_data = data_loader.prepare_train_val_dataset(
        is_rank=True, is_candidate_dataset=True
    )

    assert rank_data["X_train"].shape[0] > 10000
    assert rank_data["X_val"].shape[0] > 10000
    assert rank_data["y_train"].shape[0] > 10000
    assert rank_data["y_val"].shape[0] > 10000
    assert rank_data["X_train"].shape[0] == rank_data["y_train"].shape[0]
    assert rank_data["X_val"].shape[0] == rank_data["y_val"].shape[0]
    assert rank_data["num_diners"] > 0
    assert rank_data["num_users"] > 0
    assert rank_data["diner_mapping"] is not None
    assert rank_data["user_mapping"] is not None


def test_load_test_dataset():
    reviewer_id = QualitativeReviewerId.ROCKY.value
    test, already_reviewed = load_test_dataset(
        reviewer_id=reviewer_id,
        user_feature_param_pair={
            "categorical_feature_count": {
                "categorical_feature_names": ["diner_category_large"]
            }
        },
        diner_feature_param_pair={
            "all_review_cnt": {},
            "diner_review_tags": {},
            "diner_menu_price": {},
        },
    )
    assert test is not None
    assert already_reviewed is not None
    assert len(test) > 0
    assert len(already_reviewed) >= 0
    assert isinstance(test, pd.DataFrame)
    assert isinstance(already_reviewed, list)
