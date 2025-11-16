import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, List, Self

import pandas as pd
import yaml


class DataSource(StrEnum):
    LOCAL = "local"
    GOOGLE_DRIVE = "google_drive"


@dataclass
class DataConfig:
    user_engineered_feature_names: Dict[str, Dict[str, Any]] = None
    diner_engineered_feature_names: Dict[str, Dict[str, Any]] = None
    X_columns: List[str] = None
    y_columns: List[str] = None
    category_column_for_meta: str = "diner_category_large"
    num_neg_samples: int = 10
    sampling_type: str = "popularity"
    test_size: float = 0.4
    min_reviews: int = 3
    random_state: int = 42
    stratify: str = "reviewer_id"
    is_timeseries_by_users: bool = False
    is_timeseries_by_time_point: bool = True
    train_time_point: str = "2024-01-01"
    test_time_point: str = "2024-06-01"
    val_time_point: str = "2024-03-01"
    end_time_point: str = "2024-12-31"
    use_unique_mapping_id: bool = False
    test: bool = False
    candidate_type: str = "node2vec"
    config_root_path: str = None
    data_source: DataSource = DataSource.GOOGLE_DRIVE
    # below parameters are used when data are directly passed in fastapi server
    review: pd.DataFrame = None
    reviewer: pd.DataFrame = None
    diner: pd.DataFrame = None
    category: pd.DataFrame = None

    def __post_init__(self: Self):
        self.user_engineered_feature_names = self.user_engineered_feature_names or {}
        self.diner_engineered_feature_names = self.diner_engineered_feature_names or {}
        self.X_columns = self.X_columns or ["diner_idx", "reviewer_id"]
        self.y_columns = self.y_columns or ["reviewer_review_score"]
        self.config_root_path = self.config_root_path or "config/"
        self.additional_reviews_path = os.path.join(
            self.config_root_path, "data/additional_reviews.yaml"
        )

        if isinstance(self.data_source, str):
            self.data_source = DataSource(self.data_source)

    @classmethod
    def from_yaml(cls, path: str) -> "DataConfig":
        """
        Create DataConfig from a YAML file. Unknown fields are ignored.
        """
        with open(path, "r") as f:
            obj = yaml.safe_load(f) or {}

        # Only keep keys that exist in DataConfig
        valid_keys = {
            "user_engineered_feature_names",
            "diner_engineered_feature_names",
            "X_columns",
            "y_columns",
            "category_column_for_meta",
            "num_neg_samples",
            "sampling_type",
            "test_size",
            "min_reviews",
            "random_state",
            "stratify",
            "is_timeseries_by_users",
            "is_timeseries_by_time_point",
            "train_time_point",
            "test_time_point",
            "val_time_point",
            "end_time_point",
            "use_unique_mapping_id",
            "test",
            "candidate_type",
            "additional_reviews_path",
        }
        filtered = {k: v for k, v in obj.items() if k in valid_keys}
        return cls(**filtered)
