from pathlib import Path
from typing import Any, Dict, List, Self, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data.validator import DataValidator
from preprocess.preprocess import (
    meta_mapping,
    preprocess_common,
    reviewer_diner_mapping,
)
from store.feature import build_feature
from tools.google_drive import ensure_data_files
from tools.utils import reduce_mem_usage


class DatasetLoader:
    def __init__(
        self: Self,
        test_size: float,
        min_reviews: int,
        user_engineered_feature_names: Dict[str, Dict[str, Any]] = {},
        diner_engineered_feature_names: Dict[str, Dict[str, Any]] = {},
        X_columns: List[str] = ["diner_idx", "reviewer_id"],
        y_columns: List[str] = ["reviewer_review_score"],
        n_samples: int = 10,
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
        self.n_samples = n_samples
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
            review["reviewer_review_score"] >= review["reviewer_avg"]
        ).astype(np.int8)
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
            # reduce memory usage
            train = reduce_mem_usage(train)
            val = reduce_mem_usage(val)
            user_feature = reduce_mem_usage(user_feature)
            diner_feature = reduce_mem_usage(diner_feature)

            # negative sampling
            train = self.create_target_column(train)
            pos_train = train[train["target"] == 1]
            train = self.negative_sampling(pos_train, self.n_samples, self.random_state)

            val = self.create_target_column(val)
            pos_val = val[val["target"] == 1]
            val = self.negative_sampling(pos_val, self.n_samples, self.random_state)

            train = train.sort_values(by=["reviewer_id"])
            val = val.sort_values(by=["reviewer_id"])

            # 순위 관련 특성 병합
            train, val = self.merge_rank_features(
                train, val, user_feature, diner_feature
            )

            user_mapping = mapped_res["user_mapping"]
            diner_mapping = mapped_res["diner_mapping"]

            if not is_candidate_dataset:
                return self.create_rank_dataset(train, val, mapped_res)

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
            data = self.create_rank_dataset(train, val, mapped_res)
            data["candidates"] = candidates
            data["candidate_user_mapping"] = candidate_user_mapping
            data["candidate_diner_mapping"] = candidate_diner_mapping

            return data

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
        train: pd.DataFrame,
        val: pd.DataFrame,
        user_feature: pd.DataFrame,
        diner_feature: pd.DataFrame,
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
        val = val.merge(user_feature, on="reviewer_id", how="left").drop_duplicates(
            subset=["reviewer_id", "diner_idx"]
        )
        val = val.merge(diner_feature, on="diner_idx", how="left").drop_duplicates(
            subset=["reviewer_id", "diner_idx"]
        )

        return train, val

    def negative_sampling(
        self: Self, df: pd.DataFrame, n_samples: int, random_state: int
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

        # Get all unique diners
        candidate_pool = df["diner_idx"].unique().tolist()
        all_users = list(user_2_diner_map.keys())

        # Generate negative samples efficiently using vectorized operations
        neg_samples_list = []

        # Process in batches to manage memory
        batch_size = 1000
        for i in tqdm(range(0, len(all_users), batch_size), desc="negative sampling"):
            batch_users = all_users[i : i + batch_size]
            batch_neg_diners = []

            for user_id in batch_users:
                user_diners = set(user_2_diner_map[user_id])
                available_diners = list(set(candidate_pool) - user_diners)

                if len(available_diners) < n_samples:
                    sampled_diners = np.random.choice(
                        available_diners, size=n_samples, replace=True
                    )
                else:
                    sampled_diners = np.random.choice(
                        available_diners, size=n_samples, replace=False
                    )

                batch_neg_diners.extend(sampled_diners)

            batch_user_ids = np.repeat(batch_users, n_samples)
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
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """
        Load candidate dataset.
        """
        # 데이터 로드

        candidate = pd.read_parquet(self.candidate_paths / "candidate.parquet")
        if self.test:
            candidate = candidate.head(100)

        # 매핑 로드 및 검증
        user_mapping = pd.read_pickle(self.candidate_paths / "user_mapping.pkl")
        diner_mapping = pd.read_pickle(self.candidate_paths / "diner_mapping.pkl")

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
        test[col] = diner[col].loc[diner["diner_idx"].isin(candidates)]

    already_reviewed = user_2_diner_map.get(reviewer_id, [])

    return test, already_reviewed
