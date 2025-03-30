from pathlib import Path
from typing import Any, Dict, List, Self, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from data.validator import DataValidator
from preprocess.preprocess import (
    meta_mapping,
    preprocess_common,
    reviewer_diner_mapping,
)
from store.feature import build_feature
from tools.google_drive import ensure_data_files


class DatasetLoader:
    def __init__(
        self: Self,
        test_size: float,
        min_reviews: int,
        user_engineered_feature_names: Dict[str, Dict[str, Any]] = {},
        diner_engineered_feature_names: Dict[str, Dict[str, Any]] = {},
        X_columns: List[str] = ["diner_idx", "reviewer_id"],
        y_columns: List[str] = ["reviewer_review_score"],
        random_state: int = 42,
        stratify: str = "reviewer_id",
        is_graph_model: bool = False,
        is_candidate_dataset: bool = False,
        category_column_for_meta: str = "diner_category_large",
        test: bool = False,
    ):
        """
        Initialize the DatasetLoader class.

        Args:
            test_size: float
            min_reviews: int
            user_engineered_feature_names: Dict[str, Dict[str, Any]]
            diner_engineered_feature_names: Dict[str, Dict[str, Any]]
            X_columns: List[str]
            y_columns: List[str]
            random_state: int
            stratify: str
            is_graph_model: bool
            category_column_for_meta: str
            test: bool
        """
        self.test_size = test_size
        self.min_reviews = min_reviews
        self.user_engineered_feature_names = user_engineered_feature_names
        self.diner_engineered_feature_names = diner_engineered_feature_names
        self.X_columns = X_columns
        self.y_columns = y_columns
        self.random_state = random_state
        self.stratify = stratify
        self.is_graph_model = is_graph_model
        self.is_candidate_dataset = is_candidate_dataset
        self.category_column_for_meta = category_column_for_meta
        self.test = test

        self.data_paths = ensure_data_files()
        self.candidate_paths = Path("candidates/node2vec")

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
            review = review[review["diner_idx"].isin(yongsan_diners)]

        return review, diner, diner_with_raw_category

    def create_target_column(self: Self, review: pd.DataFrame) -> pd.DataFrame:
        """
        Create the target column for classification.
        """
        review["target"] = (
            review["reviewer_review_score"] - review["reviewer_avg"] > 0.5
        ).astype(int)
        return review

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

    def prepare_train_val_dataset(
        self: Self,
        is_rank: bool = False,
        is_candidate_dataset: bool = False,
        use_metadata: bool = False,
    ) -> Dict[str, Any]:
        """
        Load and process training data.

        Args:
            is_rank: bool
            use_metadata: bool

        Returns (Dict[str, Any]):
            A dictionary containing the training and validation sets.
        """
        # Load data
        review, diner, diner_with_raw_category = self.load_dataset()

        assert self.category_column_for_meta in diner_with_raw_category.columns

        review, diner = preprocess_common(
            review, diner, diner_with_raw_category, self.min_reviews
        )

        # Map reviewer and diner data
        mapped_res = reviewer_diner_mapping(
            review=review, diner=diner, is_graph_model=self.is_graph_model
        )
        review, diner = mapped_res["review"], mapped_res["diner"]
        mapped_res = {
            k: v for k, v in mapped_res.items() if k not in ["review", "diner"]
        }

        # Split data into train and validation
        train, val = self.train_test_split_stratify(review)

        # Feature engineering
        user_feature, diner_feature, diner_meta_feature = build_feature(
            review,
            diner,
            self.user_engineered_feature_names,
            self.diner_engineered_feature_names,
        )

        if is_rank:
            # 순위 관련 특성 병합
            train, val = self.merge_rank_features(
                train, val, user_feature, diner_feature, diner_meta_feature
            )
            user_mapping = mapped_res["user_mapping"]

            if not is_candidate_dataset:
                return self.create_rank_dataset(train, val, mapped_res)

            candidates, candidate_user_mapping, candidate_diner_mapping = (
                self.load_candidate_dataset(
                    user_feature, diner_feature, diner_meta_feature
                )
            )

            # 후보군 생성 모델과 재순위화 모델의 사용자 ID 매핑 검증
            self._validate_user_mappings(candidate_user_mapping, user_mapping)

            return (
                self.create_rank_dataset(train, val, mapped_res),
                candidates,
                candidate_user_mapping,
                candidate_diner_mapping,
            )

        if use_metadata:
            meta_mapping_info = meta_mapping(
                diner=diner_meta_feature,
                num_users=mapped_res["num_users"],
                num_diners=mapped_res["num_diners"],
            )
            mapped_res.update(meta_mapping_info)

        return self.create_graph_dataset(
            train, val, user_feature, diner_feature, diner_meta_feature, mapped_res
        )

    def _validate_user_mappings(
        self: Self,
        candidate_user_mapping: Dict[str, Any],
        user_mapping: Dict[str, Any],
    ) -> None:
        """
        Validate user mappings between candidate generation and reranking models.
        """
        # 매핑 검증
        if not set(candidate_user_mapping.keys()).issubset(set(user_mapping.keys())):
            missing_users = set(candidate_user_mapping.keys()) - set(
                user_mapping.keys()
            )
            raise ValueError(
                f"후보군 생성 모델의 사용자 ID {missing_users}가 재순위화 모델의 매핑에 없습니다."
            )

    def merge_rank_features(
        self: Self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        user_feature: pd.DataFrame,
        diner_feature: pd.DataFrame,
        diner_meta_feature: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        train = train.merge(user_feature, on="reviewer_id", how="left").drop_duplicates(
            subset=["reviewer_id", "diner_idx"]
        )
        train = train.merge(diner_feature, on="diner_idx", how="left").drop_duplicates(
            subset=["reviewer_id", "diner_idx"]
        )
        train = train.merge(
            diner_meta_feature, on="diner_idx", how="left"
        ).drop_duplicates(subset=["reviewer_id", "diner_idx"])

        val = val.merge(user_feature, on="reviewer_id", how="left").drop_duplicates(
            subset=["reviewer_id", "diner_idx"]
        )
        val = val.merge(diner_feature, on="diner_idx", how="left").drop_duplicates(
            subset=["reviewer_id", "diner_idx"]
        )
        val = val.merge(diner_meta_feature, on="diner_idx", how="left").drop_duplicates(
            subset=["reviewer_id", "diner_idx"]
        )

        return train, val

    def create_rank_dataset(
        self: Self, train: pd.DataFrame, val: pd.DataFrame, mapped_res: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare the output for ranking tasks.

        Args:
            train: pd.DataFrame
            val: pd.DataFrame
            mapped_res: Dict[str, Any]

        Returns (Dict[str, Any]):
            A dictionary containing the training and validation sets.
        """
        train = self.create_target_column(train)
        val = self.create_target_column(val)

        train = train.sort_values(by=["reviewer_id"])
        val = val.sort_values(by=["reviewer_id"])

        return {
            "X_train": train.drop(columns=["target"]),
            "y_train": train["target"],
            "X_val": val.drop(columns=["target"]),
            "y_val": val["target"],
            **mapped_res,
        }

    def load_candidate_dataset(
        self: Self,
        user_feature: pd.DataFrame,
        diner_feature: pd.DataFrame,
        diner_meta_feature: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Load candidate dataset.
        """
        if self.test:
            # 필요한 컬럼만 읽어서 메모리 사용량 줄이기
            columns = ["user_id", "diner_id"]
            candidate = pd.read_parquet(
                self.candidate_paths / "candidate.parquet",
                columns=columns,
                engine="pyarrow",
            ).head(100)
            # test 모드일 때는 매핑 파일도 일부만 읽기
            candidate_user_mapping = {}
            candidate_diner_mapping = {}

            # 후보군 데이터셋의 user_id와 diner_id만 매핑에 포함
            for _, row in candidate.iterrows():
                candidate_user_mapping[row["user_id"]] = row["user_id"]
                candidate_diner_mapping[row["diner_id"]] = row["diner_id"]

        else:
            candidate = pd.read_parquet(self.candidate_paths / "candidate.parquet")
            candidate_user_mapping = pd.read_pickle(
                self.candidate_paths / "user_mapping.pkl"
            )
            candidate_diner_mapping = pd.read_pickle(
                self.candidate_paths / "dimer_mapping.pkl"
            )

        candidate["reviewer_id"] = candidate["user_id"].copy()
        candidate["diner_idx"] = candidate["diner_id"].copy()

        candidate = candidate.merge(user_feature, on="reviewer_id", how="left")
        candidate = candidate.merge(diner_feature, on="diner_idx", how="left")
        candidate = candidate.merge(diner_meta_feature, on="diner_idx", how="left")

        return candidate, candidate_user_mapping, candidate_diner_mapping

    def create_graph_dataset(
        self: Self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        user_feature: pd.DataFrame,
        diner_feature: pd.DataFrame,
        diner_meta_feature: pd.DataFrame,
        mapped_res: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepare the standard output for graph embedding.

        Args:
            train: pd.DataFrame
            val: pd.DataFrame
            user_feature: pd.DataFrame
            diner_feature: pd.DataFrame
            diner_meta_feature: pd.DataFrame
            mapped_res: Dict[str, Any]

        Returns (Dict[str, Any]):
            A dictionary containing the training and validation
        """
        return {
            "X_train": torch.tensor(train[self.X_columns].values),
            "y_train": torch.tensor(train[self.y_columns].values, dtype=torch.float32),
            "X_val": torch.tensor(val[self.X_columns].values),
            "y_val": torch.tensor(val[self.y_columns].values, dtype=torch.float32),
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
            **mapped_res,
        }


def load_test_dataset(
    reviewer_id: int,
    user_feature_param_pair: Dict[str, Any],
    diner_feature_param_pair: Dict[str, Any],
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Load test dataset for inference
    Args:
        reviewer_id: int
        user_feature_param_pair: dict
        diner_feature_param_pair: dict

    Returns (Tuple[pd.DataFrame, list[str]]):
        test, already_reviewed
    """

    # 필요한 데이터 다운로드 확인
    data_paths = ensure_data_files()

    # load data
    diner = pd.read_csv(data_paths["diner"], low_memory=False)
    review = pd.read_csv(data_paths["review"])
    reviewer = pd.read_csv(data_paths["reviewer"])

    diner_with_raw_category = pd.read_csv(data_paths["category"])
    data_validator = DataValidator()
    review = data_validator.validate(review, name_of_df="review")
    diner = data_validator.validate(diner, name_of_df="diner")
    reviewer = reviewer[reviewer["reviewer_id"] == reviewer_id]
    review = pd.merge(review, reviewer, on="reviewer_id", how="left")

    # merge category column
    diner = pd.merge(
        left=diner,
        right=diner_with_raw_category,
        how="left",
        on="diner_idx",
    )

    # feature engineering
    user_feature, diner_feature, diner_meta_feature = build_feature(
        review, diner, user_feature_param_pair, diner_feature_param_pair
    )

    # 사용자별 리뷰한 레스토랑 ID 목록 생성
    user_2_diner_df = review.groupby("reviewer_id").agg({"diner_idx": list})
    user_2_diner_map = dict(zip(user_2_diner_df.index, user_2_diner_df["diner_idx"]))

    # 레스토랑 후보군 리스트
    candidate_pool = diner["diner_idx"].unique().tolist()

    reviewed_diners = list(set(user_2_diner_map.get(reviewer_id, [])))
    candidates = [d for d in candidate_pool if d not in reviewed_diners]

    review = review.drop(columns=["diner_idx"])

    # Create test data
    test = pd.DataFrame({"reviewer_id": reviewer_id, "diner_idx": candidates})
    test = test.merge(user_feature, on="reviewer_id", how="left")
    test = test.merge(diner_feature, on="diner_idx", how="left")
    test = test.merge(review, on="reviewer_id", how="left")

    # Add diner columns
    diner_cols = [
        "diner_name",
        "diner_lat",
        "diner_lon",
        "diner_category_large",
        "diner_category_middle",
    ]
    for col in diner_cols:
        test[col] = diner[col].loc[diner["diner_idx"].isin(candidates)]

    already_reviewed = user_2_diner_map.get(reviewer_id, [])

    return test, already_reviewed
