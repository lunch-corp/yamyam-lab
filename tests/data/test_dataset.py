try:
    import os
    import sys

    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src")
    )

except ModuleNotFoundError:
    raise Exception("Module not found")

from unittest.mock import patch

import pandas as pd
import pytest

from yamyam_lab.data.base import BaseDatasetLoader
from yamyam_lab.data.config import DataConfig
from yamyam_lab.data.ranker import load_test_dataset


def test_loader_dataset(mock_load_dataset):
    with patch(
        "yamyam_lab.data.base.BaseDatasetLoader.load_dataset",
        return_value=mock_load_dataset,
    ):
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
def test_load_test_dataset(
    setup_data_config, mock_load_dataset, mock_diner, mock_diner_with_raw_category
):
    mock_data_paths = {
        "diner": "/path/to/diner.csv",
        "category": "/path/to/category.csv",
    }

    with patch(
        "yamyam_lab.data.base.BaseDatasetLoader.load_dataset",
        return_value=mock_load_dataset,
    ):
        with patch("pandas.read_csv") as mock_read_csv:
            # Configure mock to return different dataframes based on the file being read
            def read_csv_side_effect(filepath, **kwargs):
                if "category" in str(filepath):
                    return mock_diner_with_raw_category
                else:  # diner file
                    return mock_diner

            mock_read_csv.side_effect = read_csv_side_effect

            # Patch the data_paths attribute at the point where it's accessed
            with patch(
                "yamyam_lab.data.ranker.RankerDatasetLoader.data_paths",
                mock_data_paths,
                create=True,
            ):
                test = load_test_dataset(setup_data_config)
                assert test is not None
                assert len(test) > 0
                assert isinstance(test, pd.DataFrame)
