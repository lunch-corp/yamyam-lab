import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Self, Tuple

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from features import build_feature
from preprocess.preprocess import (
    meta_mapping,
    prepare_networkx_undirected_graph,
    prepare_torch_dataloader,
    preprocess_common,
    reviewer_diner_mapping,
)
from tools.config import load_yaml
from tools.google_drive import ensure_data_files
from tools.utils import reduce_mem_usage


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
    additional_reviews_path: str = "config/data/additional_reviews.yaml"

    def __post_init__(self: Self):
        self.user_engineered_feature_names = self.user_engineered_feature_names or {}
        self.diner_engineered_feature_names = self.diner_engineered_feature_names or {}
        self.X_columns = self.X_columns or ["diner_idx", "reviewer_id"]
        self.y_columns = self.y_columns or ["reviewer_review_score"]


class DatasetLoader:
    def __init__(self: Self, data_config: DataConfig = DataConfig()):
        """
        Initialize the DatasetLoader class.

        There are 3 types of strategies when splitting reviews into train / test.
            Case 1) is_timeseries_by_users == False & is_timeseries_by_time_point == False
                -> Split reviews stratified with reviewer_id without considering timeseries
            Case 2) is_timeseries_by_users == True & is_timeseries_by_time_point == False
                -> Split reviews into train / test considering timeseries within each user
            Case 3) is_timeseries_by_users == False & is_timeseries_by_time_point == True
                -> Split reviews into train / test based on a specific time point
                if val_time_point is not None:
                    train_time_point <= dt < val_time_point : train dataset
                    val_time_point <= dt < test_time_point : val dataset
                    test_time_point <= dt < end_time_point : test dataset
                else:
                    train_time_point <= dt < test_time_point : train dataset
                    test_time_point <= dt < end_time_point : test dataset
            Case 4) is_timeseries_by_users == True & is_timeseries_by_time_point == True
                -> Will raise error

        Args:
            data_config: Configuration for dataset loading including features, splitting strategy,
                        sampling parameters and model settings
        """
        self.data_config = data_config

        # Set instance attributes for backward compatibility
        self.test_size = self.data_config.test_size
        self.min_reviews = self.data_config.min_reviews
        self.user_engineered_feature_names = (
            self.data_config.user_engineered_feature_names
        )
        self.diner_engineered_feature_names = (
            self.data_config.diner_engineered_feature_names
        )
        self.X_columns = self.data_config.X_columns
        self.y_columns = self.data_config.y_columns
        self.num_neg_samples = self.data_config.num_neg_samples
        self.random_state = self.data_config.random_state
        self.stratify = self.data_config.stratify
        self.sampling_type = self.data_config.sampling_type
        self.is_timeseries_by_users = self.data_config.is_timeseries_by_users
        self.is_timeseries_by_time_point = self.data_config.is_timeseries_by_time_point
        self.train_time_point = self.data_config.train_time_point
        self.test_time_point = self.data_config.test_time_point
        self.val_time_point = self.data_config.val_time_point
        self.end_time_point = self.data_config.end_time_point
        self.use_unique_mapping_id = self.data_config.use_unique_mapping_id
        self.category_column_for_meta = self.data_config.category_column_for_meta
        self.test = self.data_config.test
        self.additional_reviews = load_yaml(self.data_config.additional_reviews_path)
        self.diner_ids_from_additional_reviews = (
            self.get_diner_ids_from_additional_reviews()
        )

        self.data_paths = ensure_data_files()
        self.candidate_paths = Path(f"candidates/{self.data_config.candidate_type}")

        self._validate_input_params()

    def _validate_input_params(self):
        match (self.is_timeseries_by_users, self.is_timeseries_by_time_point):
            # Error case
            case (True, True):
                raise ValueError(
                    "is_timeseries_by_users and is_timeseries cannot be set to True simultaneously."
                )

            # Case 1) Stratified split
            case (False, False):
                if self.test_size is None:
                    raise ValueError(
                        "test_size should be set when splitting train / test with stratified option"
                    )
                if self.min_reviews is None:
                    raise ValueError(
                        "min_reviews should be set when splitting train / test with stratified option"
                    )

            # Case 2) Timeseries by users
            case (True, False):
                if self.test_size is None:
                    raise ValueError(
                        "test_size should be set when splitting train / test with timeseries by users"
                    )

            # Case 3) Timeseries by time point
            case (False, True):
                if (
                    self.train_time_point is None
                    or self.test_time_point is None
                    or self.end_time_point is None
                ):
                    raise ValueError(
                        "All of train_time_point, test_time_point and end_time_point should not be None when is_timeseries_by_time_point is True"
                    )

                time_points = [
                    self.train_time_point,
                    self.val_time_point,
                    self.test_time_point,
                    self.end_time_point,
                ]
                names = [
                    "train_time_point",
                    "val_time_point",
                    "test_time_point",
                    "end_time_point",
                ]
                for name, time_point in zip(names, time_points):
                    if time_point is None and name == "val_time_point":
                        continue
                    if not self.is_valid_date_format(time_point):
                        raise ValueError(
                            f"{name} is invalid date format, expected YYYY-MM-DD but got {time_point}"
                        )

                if self.train_time_point >= self.test_time_point:
                    raise ValueError(
                        "time point for train data should not be greater or equal than time point for test data"
                    )
                if self.test_time_point >= self.end_time_point:
                    raise ValueError(
                        "time point for test data should not be greater or equal than end time point"
                    )
                if self.val_time_point is not None:
                    if self.train_time_point >= self.val_time_point:
                        raise ValueError(
                            "time point for train data should not be greater or equal than time point for val data"
                        )
                    if self.val_time_point >= self.test_time_point:
                        raise ValueError(
                            "time point for val data should not be greater or equal than time point for test data"
                        )

    def _validate_additional_reviews(
        self, reviewer_mapping: Dict, diner_mapping: Dict
    ) -> None:
        """
        Validate additional_reviews.yaml
        1. Validate whether reviewer_id from yaml file already exists in original review dataset or not.
        2. Validate whether diner_ids from yaml file exist in original reviewe dataset or not.
        """
        for member_name, info in self.additional_reviews.items():
            if info["reviewer_id"] in reviewer_mapping:
                raise ValueError(
                    f"There is already reviewer_id {info['reviewer_id']}, so please use another reviewer_id."
                )
            reviews = info["reviews"]["train"] + info["reviews"]["test"]
            for review in reviews:
                if review["diner_id"] not in diner_mapping:
                    raise ValueError(
                        f"diner_idx {review['diner_id']} does not exist in review data, please check if you write correct diner_idx."
                    )

    def load_dataset(self: Self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and merge review, diner, and category data.

        Returns (Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]):
            review, diner, diner_with_raw_category
        """
        review = pd.read_csv(self.data_paths["review"])
        reviewer = pd.read_csv(self.data_paths["reviewer"])
        review = pd.merge(review, reviewer, on="reviewer_id", how="left")

        diner = pd.read_csv(self.data_paths["diner"], low_memory=False)
        diner_with_raw_category = pd.read_csv(self.data_paths["category"])

        if self.test:
            yongsan_diners = diner[
                diner["diner_road_address"].str.startswith("서울 용산구", na=False)
            ]["diner_idx"].unique()[:100]
            review = review[
                review["diner_idx"].isin(yongsan_diners)
                | review["diner_idx"].isin(self.diner_ids_from_additional_reviews)
            ]

        return review, diner, diner_with_raw_category

    def create_target_column(self: Self, review: pd.DataFrame) -> pd.DataFrame:
        """
        Create the target column for classification using temporal reviewer_avg.

        이 메서드는 데이터 리키지(Data Leakage)를 방지하기 위해 시점별 조정된
        reviewer_avg를 사용합니다.

        기존 방식의 문제점:
        - 전체 기간의 reviewer_avg 사용 → 미래 정보 포함
        - 실제 서비스 환경과 불일치

        개선된 방식:
        - 각 리뷰 작성 시점까지의 정보만 사용
        - temporal_reviewer_avg 기반 타겟 생성
        - 실제 추천 시스템 환경과 일치

        Returns:
            pd.DataFrame: temporal 기반 target 컬럼이 포함된 리뷰 데이터
        """

        # 시점별 reviewer_avg 계산
        avg_adjusted_review = self._calculate_temporal_reviewer_avg(review)

        # 시점별 target 계산
        avg_adjusted_review["target"] = (
            avg_adjusted_review["reviewer_review_score"]
            >= avg_adjusted_review["temporal_reviewer_avg"]
        ).astype(np.int8)

        # review_avg 컬럼 생성 (temporal_reviewer_avg를 복사)
        if "review_avg" in avg_adjusted_review.columns:
            avg_adjusted_review.drop(columns=["review_avg"], inplace=True)
        avg_adjusted_review["review_avg"] = avg_adjusted_review["temporal_reviewer_avg"]

        # temporal_reviewer_avg 컬럼 제거
        avg_adjusted_review = avg_adjusted_review.drop(
            columns=["temporal_reviewer_avg"]
        )

        return avg_adjusted_review

    def _calculate_temporal_reviewer_avg(
        self: Self, review: pd.DataFrame
    ) -> pd.DataFrame:
        """
        각 리뷰 작성 시점에서의 reviewer_avg를 계산합니다.

        Args:
            review (pd.DataFrame): 리뷰 데이터프레임

        Returns:
            pd.DataFrame: temporal_reviewer_avg 컬럼이 추가된 데이터프레임
        """
        # 리뷰 데이터를 복사하여 작업
        df = review.copy()

        # 날짜 컬럼을 datetime으로 변환
        df["reviewer_review_date"] = pd.to_datetime(df["reviewer_review_date"])

        # 각 리뷰어별로 날짜순으로 정렬
        df = df.sort_values(["reviewer_id", "reviewer_review_date"])

        # 각 리뷰 시점에서의 평균 계산
        df["temporal_reviewer_avg"] = df.groupby("reviewer_id")[
            "reviewer_review_score"
        ].transform(lambda x: x.expanding().mean())

        # shift를 사용하되, 첫 번째 리뷰는 자기 자신의 점수를 사용
        df["temporal_reviewer_avg_shifted"] = df.groupby("reviewer_id")[
            "reviewer_review_score"
        ].transform(lambda x: x.expanding().mean().shift(1))

        first_review_mask = df.groupby("reviewer_id").cumcount() == 0

        df["temporal_reviewer_avg"] = df.groupby("reviewer_id")[
            "reviewer_review_score"
        ].transform(lambda x: x.expanding().mean().shift(1))

        df.loc[first_review_mask, "temporal_reviewer_avg"] = df.loc[
            first_review_mask, "reviewer_review_score"
        ]

        df = df.drop("temporal_reviewer_avg_shifted", axis=1)

        return df

    # TODO: badge_level를 이용할 경우
    # def _estimate_temporal_badge_level(
    #     self: Self, review: pd.DataFrame
    # ) -> pd.DataFrame:
    #     """
    #     시점별 badge_level을 추정합니다.
    #     리뷰 수 진행률에 비례하여 최종 badge_level을 추정합니다.

    #     Args:
    #         review (pd.DataFrame): 리뷰 데이터프레임

    #     Returns:
    #         pd.DataFrame: temporal_badge_level 컬럼이 추가된 데이터프레임
    #     """
    #     # 리뷰 데이터를 복사하여 작업
    #     df = review.copy()

    #     # 날짜 컬럼을 datetime으로 변환
    #     df["reviewer_review_date"] = pd.to_datetime(df["reviewer_review_date"])

    #     # 각 리뷰어별로 날짜순으로 정렬
    #     df = df.sort_values(["reviewer_id", "reviewer_review_date"])

    #     # 각 리뷰어별 총 리뷰 수 계산
    #     total_reviews = df.groupby("reviewer_id").size().to_dict()

    #     # 각 리뷰 시점까지의 누적 리뷰 수 계산
    #     df["cumulative_review_count"] = df.groupby("reviewer_id").cumcount() + 1

    #     # 진행률 계산 (현재 리뷰 수 / 총 리뷰 수)
    #     df["review_progress_ratio"] = df.apply(
    #         lambda row: row["cumulative_review_count"]
    #         / total_reviews[row["reviewer_id"]],
    #         axis=1,
    #     )

    #     # 시점별 badge_level 추정 (진행률 × 최종 badge_level)
    #     df["temporal_badge_level"] = df["review_progress_ratio"] * df["badge_level"]

    #     # badge_level은 최소 1 이상이어야 한다고 가정
    #     df["temporal_badge_level"] = df["temporal_badge_level"].clip(lower=1.0)

    #     return df

    def train_test_split_stratify(
        self: Self, review: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and validation sets, stratified by a specified column.

        Args:
            review: pd.DataFrame

        Returns (Tuple[pd.DataFrame, pd.DataFrame]):
            train, val
        """

        train, val = train_test_split(
            review,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=review[self.stratify],
        )

        assert np.array_equal(
            np.sort(train["reviewer_id"].unique()), np.sort(val["reviewer_id"].unique())
        )
        return train, val

    def train_test_split_timeseries_by_users(
        self: Self, review: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and validation sets based on time series order.
        This ensures that earlier reviews are in the training set and later reviews are in the validation set.

        Args:
            review: pd.DataFrame

        Returns (Tuple[pd.DataFrame, pd.DataFrame]):
            train, val
        """
        # Convert review date to datetime
        review["reviewer_review_date"] = pd.to_datetime(review["reviewer_review_date"])

        # Group by reviewer and sort by date within each group
        review = review.sort_values(["reviewer_id", "reviewer_review_date"])

        # Create train/val splits for each reviewer based on time
        train_indices = []
        val_indices = []

        for reviewer_id in review["reviewer_id"].unique():
            reviewer_mask = review["reviewer_id"] == reviewer_id
            reviewer_reviews = review[reviewer_mask]

            # Calculate split point for this reviewer
            split_idx = int(len(reviewer_reviews) * (1 - self.test_size))

            # Get indices for train/val split
            reviewer_indices = reviewer_reviews.index
            train_indices.extend(reviewer_indices[:split_idx])
            val_indices.extend(reviewer_indices[split_idx:])

        # Split the data using the collected indices
        train = review.loc[train_indices]
        val = review.loc[val_indices]

        return train, val

    def train_test_split_timeseries_by_time_point(
        self: Self,
        review: pd.DataFrame,
        train_time_point: str,
        test_time_point: str,
        end_time_point: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_time_point = datetime.strptime(train_time_point, "%Y-%m-%d")
        test_time_point = datetime.strptime(test_time_point, "%Y-%m-%d")
        end_time_point = datetime.strptime(end_time_point, "%Y-%m-%d")
        train = review[
            lambda x: (train_time_point <= x["reviewer_review_date"])
            & (x["reviewer_review_date"] < test_time_point)
        ]
        test = review[
            lambda x: (test_time_point <= x["reviewer_review_date"])
            & (x["reviewer_review_date"] < end_time_point)
        ]
        return train, test

    def _apply_negative_sampling_if_needed(
        self: Self,
        df: pd.DataFrame,
        sampling_type: str,
        num_neg_samples: int,
        random_state: int,
    ) -> pd.DataFrame:
        """
        Apply negative sampling to the dataframe if num_neg_samples is greater than 0.

        Args:
            df: DataFrame containing positive samples
            sampling_type: Type of sampling to use ('popularity' or 'random')
            num_neg_samples: Number of negative samples to generate
            random_state: Random seed for reproducibility

        Returns:
            DataFrame with negative samples added if num_neg_samples > 0, otherwise original DataFrame
        """
        if num_neg_samples <= 0:
            return df

        return self.negative_sampling(
            sampling_type=sampling_type,
            df=df,
            num_neg_samples=num_neg_samples,
            random_state=random_state,
        )

    def prepare_train_val_dataset(
        self: Self,
        filter_config: Dict[str, Any] = None,
        is_rank: bool = False,
        is_csr: bool = False,
        is_networkx_graph: bool = False,
        is_tensor: bool = False,
        use_metadata: bool = False,
        weighted_edge: bool = False,
    ) -> Dict[str, Any]:
        """
        Load and process training data.

        Args:
            filter_config (Dict[str, Any]): Filter config used when filtering reviews.
            is_rank (bool): Indicator if it is ranking model or not.
            is_csr (bool): Indicator if csr format or not for als model.
            is_networkx_graph (bool): Indicator if using metworkx graph object or not.
            is_tensor (bool): Indicator if using tensor or not.
            use_metadata (bool): Indicator if using metadata or not in graph model, especially for metapath2vec
            weighted_edge (bool): Indicator if using weighted edge or not.

        Returns (Dict[str, Any]):
            A dictionary containing the training and validation sets.
        """
        # Load data
        review, diner, diner_with_raw_category = self.load_dataset()

        assert self.category_column_for_meta in diner_with_raw_category.columns

        review, diner = preprocess_common(
            review=review,
            diner=diner,
            diner_with_raw_category=diner_with_raw_category,
            min_reviews=self.min_reviews,
            is_timeseries_by_time_point=self.is_timeseries_by_time_point,
            filter_config=filter_config,
        )

        # Map reviewer and diner data
        mapped_res = reviewer_diner_mapping(
            review=review, diner=diner, use_unique_mapping_id=self.use_unique_mapping_id
        )
        review, diner = mapped_res["review"], mapped_res["diner"]
        mapped_res = {
            k: v for k, v in mapped_res.items() if k not in ["review", "diner"]
        }

        # Split data into train and validation
        if self.is_timeseries_by_users and not self.is_timeseries_by_time_point:
            train, test = self.train_test_split_timeseries_by_users(review)
            train, val = self.train_test_split_timeseries_by_users(train)

        elif not self.is_timeseries_by_users and self.is_timeseries_by_time_point:
            # split train and test by time point
            train, test = self.train_test_split_timeseries_by_time_point(
                review=review,
                train_time_point=self.train_time_point,
                test_time_point=self.test_time_point,
                end_time_point=self.end_time_point,
            )
            # split train and val by time point
            train, val = self.train_test_split_timeseries_by_time_point(
                review=review,
                train_time_point=self.train_time_point,
                test_time_point=self.val_time_point,
                end_time_point=self.test_time_point,
            )

        else:
            train, test = self.train_test_split_stratify(review)
            train, val = self.train_test_split_stratify(train)

        # integrate additional reviews
        # first validate whether additional_reviews.yaml is incorrectly written or not
        self._validate_additional_reviews(
            reviewer_mapping=mapped_res["user_mapping"],
            diner_mapping=mapped_res["diner_mapping"],
        )
        # then, merge additional reviews into train / test dataset
        mapped_res, train, test = (
            self.integrate_additional_reviews_to_train_test_dataset(
                train=train,
                test=test,
                mapped_res=mapped_res,
            )
        )

        # Feature engineering
        user_feature, diner_feature, diner_meta_feature = build_feature(
            review=train,
            diner=diner,
            all_user_ids=list(mapped_res["user_mapping"].values()),
            all_diner_ids=list(mapped_res["diner_mapping"].values()),
            user_engineered_feature_names=self.user_engineered_feature_names,
            diner_engineered_feature_names=self.diner_engineered_feature_names,
        )

        if is_rank:
            # reduce memory usage
            train = reduce_mem_usage(train)
            val = reduce_mem_usage(val)
            test = reduce_mem_usage(test)
            user_feature = reduce_mem_usage(user_feature)
            diner_feature = reduce_mem_usage(diner_feature)

            # Identify cold start users
            train_users = set(train["reviewer_id"].unique())
            val_cold_users = set(val["reviewer_id"].unique()) - train_users
            test_cold_users = set(test["reviewer_id"].unique()) - train_users

            val_cold_start_user = val[val["reviewer_id"].isin(val_cold_users)]
            val_warm_start_user = val[~val["reviewer_id"].isin(val_cold_users)]
            test_cold_start_user = test[test["reviewer_id"].isin(test_cold_users)]
            test_warm_start_user = test[~test["reviewer_id"].isin(test_cold_users)]

            # Apply negative sampling if configured
            train = self.create_target_column(train)
            train = self._apply_negative_sampling_if_needed(
                df=train[train["target"] == 1],
                sampling_type=self.sampling_type,
                num_neg_samples=self.num_neg_samples,
                random_state=self.random_state,
            )

            val = self.create_target_column(val)
            val = self._apply_negative_sampling_if_needed(
                df=val[val["target"] == 1],
                sampling_type=self.sampling_type,
                num_neg_samples=self.num_neg_samples,
                random_state=self.random_state,
            )
            val_warm_start_user = self.create_target_column(val_warm_start_user)
            val_cold_start_user = self.create_target_column(val_cold_start_user)

            test = self.create_target_column(test)
            test_cold_start_user = self.create_target_column(test_cold_start_user)
            test_warm_start_user = self.create_target_column(test_warm_start_user)

            # sort by reviewer_id
            train = train.sort_values(by=["reviewer_id"])
            val = val.sort_values(by=["reviewer_id"])
            val_cold_start_user = val_cold_start_user.sort_values(by=["reviewer_id"])
            val_warm_start_user = val_warm_start_user.sort_values(by=["reviewer_id"])
            test = test.sort_values(by=["reviewer_id"])
            test_cold_start_user = test_cold_start_user.sort_values(by=["reviewer_id"])
            test_warm_start_user = test_warm_start_user.sort_values(by=["reviewer_id"])

            # 순위 관련 특성 병합
            train = self.merge_rank_features(train, user_feature, diner_feature)
            val = self.merge_rank_features(val, user_feature, diner_feature)
            val_cold_start_user = self.merge_rank_features(
                val_cold_start_user, user_feature, diner_feature
            )
            val_warm_start_user = self.merge_rank_features(
                val_warm_start_user, user_feature, diner_feature
            )
            test = self.merge_rank_features(test, user_feature, diner_feature)

            user_mapping = mapped_res["user_mapping"]
            diner_mapping = mapped_res["diner_mapping"]

            candidates, candidate_user_mapping, candidate_diner_mapping = (
                self.load_candidate_dataset(user_feature, diner_feature)
            )

            # 후보군 생성 모델과 재순위화 모델의 사용자 ID 매핑 검증
            self._validate_user_mappings(
                candidate_user_mapping=candidate_user_mapping,
                candidate_diner_mapping=candidate_diner_mapping,
                user_mapping=user_mapping,
                diner_mapping=diner_mapping,
            )

            # rank dataset
            data = self.create_rank_dataset(
                train,
                val,
                test,
                val_cold_start_user,
                val_warm_start_user,
                test_cold_start_user,
                test_warm_start_user,
                mapped_res,
            )
            data["candidates"] = candidates
            data["candidate_user_mapping"] = candidate_user_mapping
            data["candidate_diner_mapping"] = candidate_diner_mapping
            data["user_feature"] = user_feature
            data["diner_feature"] = diner_feature

            return data

        val_warm_start_user_ids, val_cold_start_user_ids = (
            self.get_warm_cold_start_user_ids(
                train_review=train,
                test_review=val,
            )
        )
        test_warm_start_user_ids, test_cold_start_user_ids = (
            self.get_warm_cold_start_user_ids(
                train_review=train,
                test_review=test,
            )
        )
        train_user_ids = train["reviewer_id"].unique()
        val_user_ids = val["reviewer_id"].unique()
        test_user_ids = test["reviewer_id"].unique()
        train_diner_ids = train["diner_idx"].unique()
        val_diner_ids = val["diner_idx"].unique()
        test_diner_ids = test["diner_idx"].unique()
        val_warm_users = val[lambda x: x["reviewer_id"].isin(val_warm_start_user_ids)]
        val_cold_users = val[lambda x: x["reviewer_id"].isin(val_cold_start_user_ids)]
        test_warm_users = test[
            lambda x: x["reviewer_id"].isin(test_warm_start_user_ids)
        ]
        test_cold_users = test[
            lambda x: x["reviewer_id"].isin(test_cold_start_user_ids)
        ]

        if is_csr:
            return self.create_csr_dataset(
                train=train,
                val=val,
                test=test,
                train_user_ids=train_user_ids,
                val_user_ids=val_user_ids,
                test_user_ids=test_user_ids,
                train_diner_ids=train_diner_ids,
                val_diner_ids=val_diner_ids,
                test_diner_ids=test_diner_ids,
                val_warm_start_user_ids=val_warm_start_user_ids,
                val_cold_start_user_ids=val_cold_start_user_ids,
                test_warm_start_user_ids=test_warm_start_user_ids,
                test_cold_start_user_ids=test_cold_start_user_ids,
                val_warm_users=val_warm_users,
                val_cold_users=val_cold_users,
                test_warm_users=test_warm_users,
                test_cold_users=test_cold_users,
                mapped_res=mapped_res,
            )

        if use_metadata:
            meta_mapping_info = meta_mapping(
                diner=diner_meta_feature,
                num_users=mapped_res["num_users"],
                num_diners=mapped_res["num_diners"],
            )
            mapped_res.update(meta_mapping_info)

        if is_networkx_graph:
            return self.create_networkx_graph_dataset(
                train=train,
                val=val,
                test=test,
                train_user_ids=train_user_ids,
                val_user_ids=val_user_ids,
                test_user_ids=test_user_ids,
                train_diner_ids=train_diner_ids,
                val_diner_ids=val_diner_ids,
                test_diner_ids=test_diner_ids,
                val_warm_start_user_ids=val_warm_start_user_ids,
                val_cold_start_user_ids=val_cold_start_user_ids,
                test_warm_start_user_ids=test_warm_start_user_ids,
                test_cold_start_user_ids=test_cold_start_user_ids,
                user_feature=user_feature,
                diner_feature=diner_feature,
                diner_meta_feature=diner_meta_feature,
                val_warm_users=val_warm_users,
                val_cold_users=val_cold_users,
                test_warm_users=test_warm_users,
                test_cold_users=test_cold_users,
                mapped_res=mapped_res,
                use_metadata=use_metadata,
                weighted_edge=weighted_edge,
            )

        if is_tensor:
            return self.create_torch_dataset(
                train=train,
                val=val,
                test=test,
                train_user_ids=train_user_ids,
                val_user_ids=val_user_ids,
                test_user_ids=test_user_ids,
                train_diner_ids=train_diner_ids,
                val_diner_ids=val_diner_ids,
                test_diner_ids=test_diner_ids,
                val_warm_start_user_ids=val_warm_start_user_ids,
                val_cold_start_user_ids=val_cold_start_user_ids,
                test_warm_start_user_ids=test_warm_start_user_ids,
                test_cold_start_user_ids=test_cold_start_user_ids,
                val_warm_users=val_warm_users,
                val_cold_users=val_cold_users,
                test_warm_users=test_warm_users,
                test_cold_users=test_cold_users,
                diner_meta_feature=diner_meta_feature,
                mapped_res=mapped_res,
            )

    def _validate_user_mappings(
        self: Self,
        candidate_user_mapping: Dict[str, Any],
        candidate_diner_mapping: Dict[str, Any],
        user_mapping: Dict[str, Any],
        diner_mapping: Dict[str, Any],
    ) -> None:
        """
        Validate user mappings between candidate generation and reranking models.
        """
        # validates user mapping
        for cand_asis_id, cand_tobe_id in candidate_user_mapping.items():
            if cand_asis_id not in user_mapping:
                continue
            if cand_tobe_id != user_mapping[cand_asis_id]:
                raise ValueError(
                    f"For original user_id={cand_asis_id}, expected {cand_tobe_id} but got {user_mapping[cand_asis_id]}."
                )

        # validates diner mapping
        for cand_asis_id, cand_tobe_id in candidate_diner_mapping.items():
            if cand_asis_id not in diner_mapping:
                continue
            if cand_tobe_id != diner_mapping[cand_asis_id]:
                raise ValueError(
                    f"For original diner_id={cand_asis_id}, expected {cand_tobe_id} but got {diner_mapping[cand_asis_id]}."
                )

    def merge_rank_features(
        self: Self,
        df: pd.DataFrame,
        user_feature: pd.DataFrame,
        diner_feature: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge rank-specific features to the train and validation sets.

        Args:
            train: pd.DataFrame
            val: pd.DataFrame
            user_feature: pd.DataFrame
            diner_feature: pd.DataFrame
            diner_meta_feature: pd.DataFrame

        Returns Tuple[pd.DataFrame, pd.DataFrame]:
            tuple: A tuple containing the training and validation sets
        """
        df = df.merge(user_feature, on="reviewer_id", how="left").fillna(0)
        df = df.merge(diner_feature, on="diner_idx", how="left").fillna(0)

        return df

    def negative_sampling(
        self: Self,
        sampling_type: str,
        df: pd.DataFrame,
        num_neg_samples: int,
        random_state: int,
    ):
        """
        Negative sampling for ranking task.

        Args:
            df: pd.DataFrame
            n_samples: int
            random_state: int

        Returns (pd.DataFrame):
            A DataFrame with negative samples
        """
        # set random seed
        np.random.seed(random_state)

        # Get list of restaurants reviewed by each user
        user_2_diner_df = df.groupby("reviewer_id").agg({"diner_idx": list})
        user_2_diner_map = dict(
            zip(user_2_diner_df.index, user_2_diner_df["diner_idx"])
        )

        # Get all unique diners and users
        candidate_pool = df["diner_idx"].unique().tolist()
        all_users = list(user_2_diner_map.keys())

        # Generate negative samples using popularity-based sampling
        diner_popularity = df["diner_idx"].value_counts()

        neg_samples_list = []
        batch_size = 1000

        # load diner category
        diner_category = pd.read_csv(self.data_paths["category"])
        diner_category = diner_category[
            diner_category["diner_category_large"].isin(
                ["한식", "중식", "양식", "일식", "아시안", "패스트푸드", "치킨", "술집"]
            )
        ]

        # group by category
        category_groups = diner_category.groupby("diner_category_large")[
            "diner_idx"
        ].apply(list)

        for i in tqdm(range(0, len(all_users), batch_size), desc="sampling"):
            batch_users = all_users[i : i + batch_size]
            batch_neg_diners = []
            for user_id in batch_users:
                user_diners = set(user_2_diner_map[user_id])
                available_diners = list(set(candidate_pool) - user_diners)

                if sampling_type == "popularity":
                    # Get popularity scores for available diners
                    available_probs = diner_popularity[available_diners]

                    # Sort diners by popularity and get top 50% most popular diners
                    sorted_diners = sorted(
                        zip(available_diners, available_probs),
                        key=lambda x: x[1],
                        reverse=True,
                    )

                    top_n = int(len(sorted_diners) * 0.5)
                    popular_diners = [d[0] for d in sorted_diners[:top_n]]

                    # Randomly sample from popular diners
                    sampled_diners = np.random.choice(
                        popular_diners,
                        size=num_neg_samples,
                        replace=len(popular_diners) < num_neg_samples,
                    )

                elif sampling_type == "random":
                    sampled_diners = np.random.choice(
                        available_diners,
                        size=num_neg_samples,
                        replace=len(available_diners) < num_neg_samples,
                    )
                elif sampling_type == "diversity":
                    sampled_diners = []
                    categories = list(category_groups.keys())

                    # 사용자가 리뷰하지 않은 레스토랑만 필터링
                    available_diners = list(set(candidate_pool) - user_diners)

                    # 각 카테고리에서 사용 가능한 레스토랑만 필터링
                    available_category_groups = {}
                    for category in categories:
                        category_diners = category_groups[category]
                        available_in_category = list(
                            set(category_diners) & set(available_diners)
                        )
                        if available_in_category:  # 사용 가능한 레스토랑이 있는 경우만
                            available_category_groups[category] = available_in_category

                    if available_category_groups:
                        categories = list(available_category_groups.keys())
                        samples_per_category = num_neg_samples // len(categories)
                        remaining_samples = num_neg_samples % len(categories)

                        for i, category in enumerate(categories):
                            category_diners = available_category_groups[category]

                            # basic sample + remaining sample
                            n_samples = samples_per_category + (
                                1 if i < remaining_samples else 0
                            )
                            n_samples = min(n_samples, len(category_diners))

                            if n_samples > 0:
                                category_samples = np.random.choice(
                                    category_diners,
                                    size=n_samples,
                                    replace=len(category_diners) < n_samples,
                                )
                                sampled_diners.extend(category_samples)

                        # 부족한 경우 랜덤으로 보충
                        if len(sampled_diners) < num_neg_samples:
                            remaining_diners = list(
                                set(available_diners) - set(sampled_diners)
                            )
                            if remaining_diners:
                                additional_samples = np.random.choice(
                                    remaining_diners,
                                    size=num_neg_samples - len(sampled_diners),
                                    replace=True,
                                )
                                sampled_diners.extend(additional_samples)
                    else:
                        # 사용 가능한 카테고리가 없는 경우 기본 샘플링
                        sampled_diners = np.random.choice(
                            available_diners,
                            size=num_neg_samples,
                            replace=len(available_diners) < num_neg_samples,
                        )
                else:
                    raise ValueError(f"Invalid sampling type: {sampling_type}")

                batch_neg_diners.extend(sampled_diners)

            batch_user_ids = np.repeat(batch_users, num_neg_samples)
            batch_df = pd.DataFrame(
                {
                    "reviewer_id": batch_user_ids,
                    "diner_idx": batch_neg_diners,
                    "target": 0,
                }
            )
            neg_samples_list.append(batch_df)

        neg_samples = pd.concat(neg_samples_list, ignore_index=True)

        # Combine positive and negative samples
        neg_df = pd.DataFrame(neg_samples)
        all_data = pd.concat([df, neg_df], ignore_index=True)

        return all_data

    def create_rank_dataset(
        self: Self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        val_cold_start_user: pd.DataFrame,
        val_warm_start_user: pd.DataFrame,
        test_cold_start_user: pd.DataFrame,
        test_warm_start_user: pd.DataFrame,
        mapped_res: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepare the output for ranking tasks.

        Args:
            train: pd.DataFrame
            val: pd.DataFrame
            test: pd.DataFrame
            val_cold_start_user: pd.DataFrame
            val_warm_start_user: pd.DataFrame
            test_cold_start_user: pd.DataFrame
            test_warm_start_user: pd.DataFrame
            mapped_res: Dict[str, Any]

        Returns (Dict[str, Any]):
            A dictionary containing the training and validation sets.
        """
        return {
            "X_train": train.drop(columns=["target"]),
            "y_train": train["target"],
            "X_val": val.drop(columns=["target"]),
            "y_val": val["target"],
            "X_test": test.drop(columns=["target"]),
            "y_test": test["target"],
            "X_val_cold_start_user": val_cold_start_user.drop(columns=["target"]),
            "y_val_cold_start_user": val_cold_start_user["target"],
            "X_val_warm_start_user": val_warm_start_user.drop(columns=["target"]),
            "y_val_warm_start_user": val_warm_start_user["target"],
            "X_test_cold_start_user": test_cold_start_user.drop(columns=["target"]),
            "y_test_cold_start_user": test_cold_start_user["target"],
            "X_test_warm_start_user": test_warm_start_user.drop(columns=["target"]),
            "y_test_warm_start_user": test_warm_start_user["target"],
            "most_popular_diner_ids": self.get_most_popular_diner_ids(
                train_review=train
            ),
            **mapped_res,
        }

    def create_csr_dataset(
        self: Self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        train_user_ids: NDArray,
        val_user_ids: NDArray,
        test_user_ids: NDArray,
        train_diner_ids: NDArray,
        val_diner_ids: NDArray,
        test_diner_ids: NDArray,
        val_warm_start_user_ids: NDArray,
        val_cold_start_user_ids: NDArray,
        test_warm_start_user_ids: NDArray,
        test_cold_start_user_ids: NDArray,
        val_warm_users: pd.DataFrame,
        val_cold_users: pd.DataFrame,
        test_warm_users: pd.DataFrame,
        test_cold_users: pd.DataFrame,
        mapped_res: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create train / val csr matrix and return other data used in evaluation.
        """
        num_total_users = len(mapped_res.get("user_mapping"))
        num_total_diners = len(mapped_res.get("diner_mapping"))
        train_Cui_csr = self.create_csr_matrix_from_review_data(
            df=train,
            num_total_users=num_total_users,
            num_total_diners=num_total_diners,
        )
        val_Cui_csr = self.create_csr_matrix_from_review_data(
            df=val,
            num_total_users=num_total_users,
            num_total_diners=num_total_diners,
        )
        test_Cui_csr = self.create_csr_matrix_from_review_data(
            df=test,
            num_total_users=num_total_users,
            num_total_diners=num_total_diners,
        )
        return {
            "X_train": train_Cui_csr,
            "X_val": val_Cui_csr,
            "X_test": test_Cui_csr,
            "X_train_df": train,
            "X_val_warm_users": val_warm_users[self.X_columns],
            "y_val_warm_users": val_warm_users[self.y_columns],
            "X_val_cold_users": val_cold_users[self.X_columns],
            "y_val_cold_users": val_cold_users[self.y_columns],
            "X_test_warm_users": test_warm_users[self.X_columns],
            "y_test_warm_users": test_warm_users[self.y_columns],
            "X_test_cold_users": test_cold_users[self.X_columns],
            "y_test_cold_users": test_cold_users[self.y_columns],
            "most_popular_diner_ids": self.get_most_popular_diner_ids(
                train_review=train
            ),
            "val_warm_start_user_ids": val_warm_start_user_ids,
            "val_cold_start_user_ids": val_cold_start_user_ids,
            "test_warm_start_user_ids": test_warm_start_user_ids,
            "test_cold_start_user_ids": test_cold_start_user_ids,
            "train_user_ids": train_user_ids,
            "val_user_ids": val_user_ids,
            "test_user_ids": test_user_ids,
            "train_diner_ids": train_diner_ids,
            "val_diner_ids": val_diner_ids,
            "test_diner_ids": test_diner_ids,
            **mapped_res,
        }

    def load_candidate_dataset(
        self: Self,
        user_feature: pd.DataFrame,
        diner_feature: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """
        Load candidate dataset.
        """
        if self.test:
            # 테스트용 임의 데이터 생성
            np.random.seed(42)
            num_samples = 10

            # candidate 데이터 생성
            candidate = pd.DataFrame(
                {
                    "user_id": np.random.randint(0, 1000, num_samples),
                    "diner_id": np.random.randint(0, 1000, num_samples),
                    "score": np.random.rand(num_samples),
                }
            )

            # 매핑 데이터 생성
            user_mapping = {str(i): i for i in range(1000)}
            diner_mapping = {str(i): i for i in range(1000)}

            candidate_user_mapping = {
                k: v for k, v in user_mapping.items() if v in candidate["user_id"]
            }
            candidate_diner_mapping = {
                k: v for k, v in diner_mapping.items() if v in candidate["diner_id"]
            }

            num_diners = len(diner_mapping)
            min_user_id = min(list(user_mapping.values()))

            # 사용자 ID 변환
            candidate_user_mapping_convert = {
                asis_id: tobe_id - num_diners
                for asis_id, tobe_id in candidate_user_mapping.items()
            }
            candidate["user_id"] = candidate["user_id"] - num_diners

            # 특성 병합
            candidate["reviewer_id"] = candidate["user_id"].copy()
            candidate["diner_idx"] = candidate["diner_id"].copy()

            candidate = candidate.merge(user_feature, on="reviewer_id", how="left")
            candidate = candidate.merge(diner_feature, on="diner_idx", how="left")

            # reduce memory usage
            candidate = reduce_mem_usage(candidate)

            return candidate, candidate_user_mapping_convert, candidate_diner_mapping

        # 데이터 로드
        candidate = pd.read_parquet(self.candidate_paths / "candidate.parquet")

        # 매핑 로드 및 검증
        user_mapping = pd.read_pickle(self.candidate_paths / "user_mapping.pkl")
        diner_mapping = pd.read_pickle(self.candidate_paths / "diner_mapping.pkl")
        user_mapping = (
            {k: v + len(diner_mapping) for k, v in user_mapping.items()}
            if self.data_config.candidate_type == "als"
            else user_mapping
        )

        candidate_user_mapping = {
            k: v for k, v in user_mapping.items() if v in candidate["user_id"]
        }
        candidate_diner_mapping = {
            k: v for k, v in diner_mapping.items() if v in candidate["diner_id"]
        }

        num_diners = len(diner_mapping)
        min_user_id = min(list(user_mapping.values()))
        if num_diners != min_user_id:
            raise ValueError(
                "Mapping ids may not be unique in candidate generation models and should be checked."
            )

        # 사용자 ID 변환
        candidate_user_mapping_convert = {
            asis_id: tobe_id - num_diners
            for asis_id, tobe_id in candidate_user_mapping.items()
        }
        candidate["user_id"] = candidate["user_id"] - num_diners

        # 특성 병합
        candidate["reviewer_id"] = candidate["user_id"].copy()
        candidate["diner_idx"] = candidate["diner_id"].copy()

        candidate = candidate.merge(user_feature, on="reviewer_id", how="left")
        candidate = candidate.merge(diner_feature, on="diner_idx", how="left")

        # reduce memory usage
        candidate = reduce_mem_usage(candidate)

        return candidate, candidate_user_mapping_convert, candidate_diner_mapping

    def create_networkx_graph_dataset(
        self: Self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        train_user_ids: NDArray,
        val_user_ids: NDArray,
        test_user_ids: NDArray,
        train_diner_ids: NDArray,
        val_diner_ids: NDArray,
        test_diner_ids: NDArray,
        val_warm_start_user_ids: NDArray,
        val_cold_start_user_ids: NDArray,
        test_warm_start_user_ids: NDArray,
        test_cold_start_user_ids: NDArray,
        val_warm_users: pd.DataFrame,
        val_cold_users: pd.DataFrame,
        test_warm_users: pd.DataFrame,
        test_cold_users: pd.DataFrame,
        user_feature: pd.DataFrame,
        diner_feature: pd.DataFrame,
        diner_meta_feature: pd.DataFrame,
        mapped_res: Dict[str, Any],
        use_metadata: bool,
        weighted_edge: bool,
    ) -> Dict[str, Any]:
        """
        Create train / val networkx graph object and return other data used in evaluation.
        """
        train_graph, val_graph = prepare_networkx_undirected_graph(
            X_train=train[self.X_columns],
            y_train=train[self.y_columns],
            X_val=val[self.X_columns],
            y_val=val[self.y_columns],
            diner=diner_meta_feature,
            user_mapping=mapped_res["user_mapping"],
            diner_mapping=mapped_res["diner_mapping"],
            meta_mapping=mapped_res["meta_mapping"] if use_metadata else None,
            weighted=weighted_edge,
            use_metadata=use_metadata,
        )

        return {
            "X_train": train[self.X_columns],
            "y_train": train[self.y_columns],
            "X_val": val[self.X_columns],
            "y_val": val[self.y_columns],
            "X_test": test[self.X_columns],
            "y_test": test[self.y_columns],
            "X_val_warm_users": val_warm_users[self.X_columns],
            "y_val_warm_users": val_warm_users[self.y_columns],
            "X_val_cold_users": val_cold_users[self.X_columns],
            "y_val_cold_users": val_cold_users[self.y_columns],
            "X_test_warm_users": test_warm_users[self.X_columns],
            "y_test_warm_users": test_warm_users[self.y_columns],
            "X_test_cold_users": test_cold_users[self.X_columns],
            "y_test_cold_users": test_cold_users[self.y_columns],
            "diner": diner_meta_feature,
            "user_feature": torch.tensor(
                user_feature.sort_values(by="reviewer_id")
                .drop("reviewer_id", axis=1)
                .values,
                dtype=torch.float32,
            ),
            "diner_feature": torch.tensor(
                diner_feature.sort_values(by="diner_idx")
                .drop("diner_idx", axis=1)
                .values,
                dtype=torch.float32,
            ),
            "most_popular_diner_ids": self.get_most_popular_diner_ids(
                train_review=train
            ),
            "val_warm_start_user_ids": val_warm_start_user_ids,
            "val_cold_start_user_ids": val_cold_start_user_ids,
            "test_warm_start_user_ids": test_warm_start_user_ids,
            "test_cold_start_user_ids": test_cold_start_user_ids,
            "train_user_ids": train_user_ids,
            "val_user_ids": val_user_ids,
            "test_user_ids": test_user_ids,
            "train_diner_ids": train_diner_ids,
            "val_diner_ids": val_diner_ids,
            "test_diner_ids": test_diner_ids,
            "train_graph": train_graph,
            "val_graph": val_graph,
            **mapped_res,
        }

    def create_torch_dataset(
        self: Self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        train_user_ids: NDArray,
        val_user_ids: NDArray,
        test_user_ids: NDArray,
        train_diner_ids: NDArray,
        val_diner_ids: NDArray,
        test_diner_ids: NDArray,
        val_warm_start_user_ids: NDArray,
        val_cold_start_user_ids: NDArray,
        test_warm_start_user_ids: NDArray,
        test_cold_start_user_ids: NDArray,
        val_warm_users: pd.DataFrame,
        val_cold_users: pd.DataFrame,
        test_warm_users: pd.DataFrame,
        test_cold_users: pd.DataFrame,
        diner_meta_feature: pd.DataFrame,
        mapped_res: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create torch dataloader and return other data used in evaluation.
        """
        X_train = torch.tensor(train[self.X_columns].values)
        y_train = torch.tensor(train[self.y_columns].values, dtype=torch.float32)
        X_val = torch.tensor(val[self.X_columns].values)
        y_val = torch.tensor(val[self.y_columns].values, dtype=torch.float32)

        train_dataloader, val_dataloader = prepare_torch_dataloader(
            X_train, y_train, X_val, y_val
        )

        return {
            "X_train": train[self.X_columns],
            "y_train": train[self.y_columns],
            "X_val": val[self.X_columns],
            "y_val": val[self.y_columns],
            "X_test": test[self.X_columns],
            "y_test": test[self.y_columns],
            "X_val_warm_users": val_warm_users[self.X_columns],
            "y_val_warm_users": val_warm_users[self.y_columns],
            "X_val_cold_users": val_cold_users[self.X_columns],
            "y_val_cold_users": val_cold_users[self.y_columns],
            "X_test_warm_users": test_warm_users[self.X_columns],
            "y_test_warm_users": test_warm_users[self.y_columns],
            "X_test_cold_users": test_cold_users[self.X_columns],
            "y_test_cold_users": test_cold_users[self.y_columns],
            "diner": diner_meta_feature,
            "most_popular_diner_ids": self.get_most_popular_diner_ids(
                train_review=train
            ),
            "val_warm_start_user_ids": val_warm_start_user_ids,
            "val_cold_start_user_ids": val_cold_start_user_ids,
            "test_warm_start_user_ids": test_warm_start_user_ids,
            "test_cold_start_user_ids": test_cold_start_user_ids,
            "train_user_ids": train_user_ids,
            "val_user_ids": val_user_ids,
            "test_user_ids": test_user_ids,
            "train_diner_ids": train_diner_ids,
            "val_diner_ids": val_diner_ids,
            "test_diner_ids": test_diner_ids,
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            **mapped_res,
        }

    def get_most_popular_diner_ids(
        self: Self, train_review: pd.DataFrame, top_k: int = 2000
    ) -> List[int]:
        """
        Get most popular diner_ids from `train review`.
        It is important that val / test data should not be used due to data leakage.
        top_k argument should be sufficiently large to cover ndcg@k, map@k and recall@k.
        Currently, k is largest when calculating recall@2000, therefore we set it as 2000 as default.
        Args:
            train_review (pd.DataFrame): Train review dataset.
            top_k (int): Top k value to get most popular diner_ids.
        Returns (List[int]):
            List of top_k diner_ids.
        """
        diner_agg = train_review.value_counts("diner_idx")[:top_k]
        return diner_agg.index.tolist()

    def get_warm_cold_start_user_ids(
        self: Self, train_review: pd.DataFrame, test_review: pd.DataFrame
    ) -> Tuple[NDArray, NDArray]:
        """
        Get warm / cold start user_ids given train/val or train/test dataset.
        Args:
            train_review (pd.DataFrame): Review data used when training model.
            test_review (pd.DataFrame): Review data used when validating model or calculating metric for final report.
        Returns (Tuple[NDArray, NDArray]):
            Tuple of list of warm / cold start user ids.
        """
        train_user_ids = set(train_review["reviewer_id"].unique())
        test_user_ids = set(test_review["reviewer_id"].unique())
        warm_start_user_ids = np.array(list(train_user_ids & test_user_ids))
        cold_start_user_ids = np.array(list(test_user_ids - train_user_ids))
        return warm_start_user_ids, cold_start_user_ids

    def create_csr_matrix_from_review_data(
        self: Self,
        df: pd.DataFrame,
        num_total_users: int,
        num_total_diners: int,
    ) -> csr_matrix:
        """
        Create csr matrix from review data for als model.
        """
        Cui_csr = csr_matrix(
            (df["reviewer_review_score"], (df["reviewer_id"], df["diner_idx"])),
            shape=(num_total_users, num_total_diners),
        )
        return Cui_csr

    def get_diner_ids_from_additional_reviews(self: Self):
        """
        Get diner_ids from additional_reviews.yaml which will be used when test mode is set as true.
        """
        diner_ids = []
        for member_name, info in self.additional_reviews.items():
            reviews = info["reviews"]["train"] + info["reviews"]["test"]
            for review in reviews:
                diner_ids.append(review["diner_id"])
        return diner_ids

    def integrate_additional_reviews_to_train_test_dataset(
        self: Self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        mapped_res: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        """
        Integrate additional reviews into split train and test dataset.
        This method does following steps.
            1. Generate mapping id converting original reviewer_id from additional_reviews.yaml
            2. Using mapped reviewer_id and mapped diner_id, generate pseudo review and integrate it to train and test dataset.
        Args:
            train (pd.DataFrame): Original train dataset.
            test (pd.DataFrame): Original test dataset.
            mapped_res (Dict[str, Any]): Original mapped_res.
        Returns (Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]):
            Integrated dataset with user_mapping updated.
        """
        max_mapped_reviewer_id = max(mapped_res["user_mapping"].values())
        for member_name, info in self.additional_reviews.items():
            # newly define reviewer_id and map it to original reviewer_mapping
            max_mapped_reviewer_id += 1
            mapped_res["user_mapping"][info["reviewer_id"]] = max_mapped_reviewer_id

            # train data concatenation
            for review in info["reviews"]["train"]:
                # map diner_id using diner_mapping dict
                mapped_diner_id = mapped_res["diner_mapping"][review["diner_id"]]
                pseudo_review = self._generate_pseudo_review(
                    diner_idx=mapped_diner_id,
                    reviewer_id=max_mapped_reviewer_id,
                    score=review["score"],
                    review_text=review["review_text"],
                )
                train = pd.concat(
                    [train, pd.DataFrame([pseudo_review])], ignore_index=True
                )

            # test data concatenation
            for review in info["reviews"]["test"]:
                # map diner_id using diner_mapping dict
                mapped_diner_id = mapped_res["diner_mapping"][review["diner_id"]]
                pseudo_review = self._generate_pseudo_review(
                    diner_idx=mapped_diner_id,
                    reviewer_id=max_mapped_reviewer_id,
                    score=review["score"],
                    review_text=review["review_text"],
                )
                test = pd.concat(
                    [test, pd.DataFrame([pseudo_review])], ignore_index=True
                )
        # overwrite some values because mapping dictionary is updated
        mapped_res["num_users"] = len(mapped_res["user_mapping"])
        mapped_res["num_diners"] = len(mapped_res["diner_mapping"])
        return mapped_res, train, test

    def _generate_pseudo_review(
        self: Self,
        diner_idx: int,
        reviewer_id: int,
        score: float,
        review_text: str,
    ) -> Dict[str, Any]:
        """
        Generate pseudo review to include train dataset or test dataset.
        Currently, when training models, diner_idx, reviewer_id, reviewer_review, reviewer_review_score are used.
        Therefore, we set those values from `additional_reviews.yaml`.
        `reviewer_review_date` is set arbitrarily because pseudo review will be included train or test manually.
        Args:
            diner_idx (int): Mapped diner_idx.
            reviewer_id (int): Mapped reviewer_id.
            score (float): Review score.
            review_text (str): Review text.
        Returns (Dict[str, Any]):
            Pseudo review with dictionary.
        """
        return {
            "diner_idx": diner_idx,
            "reviewer_id": reviewer_id,
            "review_id": -1,  # pseudo review_id
            "reviewer_review": review_text,
            "reviewer_review_date": "2024-12-31",  # pseudo review_date
            "reviewer_review_score": score,
        }

    @staticmethod
    def is_valid_date_format(date_string: str) -> bool:
        """
        Validates whether given date_string is `0000-00-00` format or not.
        """
        pattern = r"^\d{4}-\d{2}-\d{2}$"
        return bool(re.match(pattern, date_string))


def load_test_dataset(cfg: DictConfig) -> pd.DataFrame:
    """
    Load test dataset for inference
    Args:
        reviewer_id: int
        user_feature_param_pair: dict
        diner_feature_param_pair: dict

    Returns (pd.DataFrame):
        test
    """
    # load dataset
    data_loader = DatasetLoader(data_config=DataConfig(**cfg.data))
    data = data_loader.prepare_train_val_dataset(
        is_rank=True,
        filter_config=cfg.preprocess.filter,
    )
    review = data["X_test"]
    # 원본 reviewer_id를 mapping된 ID로 변환
    mapped_reviewer_id = data["user_mapping"].get(cfg.user_name)

    if mapped_reviewer_id is None:
        if not data_loader.data_config.test:
            raise ValueError(
                f"Test mode is enabled but reviewer ID {cfg.user_name} not found in test dataset."
            )
        else:
            mapped_reviewer_id = 0  # 가짜 유저 ID 생성

    # load data
    diner = pd.read_csv(data_loader.data_paths["diner"], low_memory=False)
    diner_with_raw_category = pd.read_csv(data_loader.data_paths["category"])

    # merge category column
    diner = pd.merge(
        left=diner,
        right=diner_with_raw_category,
        how="left",
        on="diner_idx",
    )

    # diner_mapping을 사용하여 diner_idx를 mapping된 ID로 변환
    # 원본 diner_idx를 mapping된 ID로 변환
    diner["mapped_diner_idx"] = diner["diner_idx"].map(data["diner_mapping"])
    diner = diner.dropna(subset=["mapped_diner_idx"])  # mapping되지 않은 diner 제거

    # 사용자별 리뷰한 레스토랑 ID 목록 생성 (mapping된 ID 사용)
    user_2_diner_df = review.groupby("reviewer_id").agg({"diner_idx": list})
    user_2_diner_map = dict(zip(user_2_diner_df.index, user_2_diner_df["diner_idx"]))

    # 레스토랑 후보군 리스트 (mapping된 ID 사용)
    candidate_pool = diner["mapped_diner_idx"].unique().tolist()

    reviewed_diners = list(set(user_2_diner_map.get(mapped_reviewer_id, [])))
    candidates = [d for d in candidate_pool if d not in reviewed_diners]

    review = review.drop(columns=["diner_idx"])

    # Create test data (mapping된 diner_idx 사용)
    test = pd.DataFrame({"reviewer_id": mapped_reviewer_id, "diner_idx": candidates})

    # user_feature와 diner_feature만 병합 (review는 제외)
    test = test.merge(data["user_feature"], on="reviewer_id", how="left")
    test = test.merge(data["diner_feature"], on="diner_idx", how="left")

    # diner 정보 병합 시 mapping된 ID 사용
    test = test.merge(
        diner[
            [
                "mapped_diner_idx",
                "diner_name",
                "diner_lat",
                "diner_lon",
                "diner_category_large",
                "diner_category_middle",
            ]
        ],
        left_on="diner_idx",
        right_on="mapped_diner_idx",
        how="left",
    )
    test = test.drop(columns=["mapped_diner_idx"])  # 중복 컬럼 제거

    # reduce memory usage
    test = reduce_mem_usage(test)
    # Add diner columns
    diner_cols = [
        "diner_name",
        "diner_lat",
        "diner_lon",
        "diner_category_large",
        "diner_category_middle",
    ]
    for col in diner_cols:
        test[col] = diner[col].loc[diner["mapped_diner_idx"].isin(candidates)]

    return test
