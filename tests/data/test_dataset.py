try:
    import os
    import sys

    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src")
    )

except ModuleNotFoundError:
    raise Exception("Module not found")

import pandas as pd
import pytest

from data.base import BaseDatasetLoader
from data.config import DataConfig
from data.ranker import load_test_dataset
from tools.utils import get_kakao_lat_lng


def test_loader_dataset():
    data_loader = BaseDatasetLoader(
        data_config=DataConfig(
            X_columns=["diner_idx", "reviewer_id"],
            y_columns=["reviewer_review_score"],
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
            is_timeseries_by_time_point=True,
            train_time_point="2024-09-01",
            val_time_point="2024-12-01",
            test_time_point="2025-01-01",
            end_time_point="2025-02-01",
            random_state=42,
            stratify="reviewer_id",
            test=True,
            candidate_type="node2vec",
        ),
    )
    data = data_loader.prepare_train_val_dataset(is_tensor=True)

    assert data["train"].shape[0] > 0
    assert data["val"].shape[0] > 0
    assert data["test"].shape[0] > 0
    assert data["user_feature"].shape[0] > 0
    assert data["diner_feature"].shape[0] > 0
    assert data["diner_meta_feature"].shape[0] > 0
    assert data["mapped_res"] is not None


@pytest.mark.parametrize(
    "setup_data_config", [("lightgbm", {}, 1)], indirect=["setup_data_config"]
)
def test_load_test_dataset(setup_data_config):
    test = load_test_dataset(setup_data_config)
    assert test is not None
    assert len(test) > 0
    assert isinstance(test, pd.DataFrame)


def test_get_kakao_lat_lng():
    location = get_kakao_lat_lng("서울 강남구 강남대로 324")
    assert location is not None
    assert location["lat"] is not None
    assert location["lng"] is not None
