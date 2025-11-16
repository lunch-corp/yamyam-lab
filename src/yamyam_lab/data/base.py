import re
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Self, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from yamyam_lab.data.config import DataConfig
from yamyam_lab.features import build_feature
from yamyam_lab.preprocess.preprocess import preprocess_common, reviewer_diner_mapping
from yamyam_lab.tools.config import load_yaml
from yamyam_lab.tools.google_drive import check_data_and_return_paths


class BaseDatasetLoader(ABC):
    def __init__(self: Self, data_config: DataConfig):
        """
        Initialize the BaseDatasetLoader class.

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

        self.data_paths = check_data_and_return_paths()
        self.candidate_paths = Path(f"candidates/{self.data_config.candidate_type}")

        self._validate_input_params()

    def _validate_input_params(self: Self) -> None:
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
        self: Self, reviewer_mapping: Dict, diner_mapping: Dict
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

    def prepare_train_val_dataset(
        self: Self, filter_config: Dict[str, Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Load and process training data. This method must be implemented by subclasses.
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
            config_root_path=self.data_config.config_root_path,
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
            config_root_path=self.data_config.config_root_path,
        )

        return {
            "train": train,
            "val": val,
            "test": test,
            "user_feature": user_feature,
            "diner_feature": diner_feature,
            "diner_meta_feature": diner_meta_feature,
            "mapped_res": mapped_res,
        }
