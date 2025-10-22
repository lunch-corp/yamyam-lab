from typing import Any, Dict, Self, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tools.utils import reduce_mem_usage
from tqdm import tqdm

from data.base import BaseDatasetLoader
from data.config import DataConfig


class RankerDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader specifically for ranking models (CatBoost, LightGBM, XGBoost, etc.)
    """

    def __init__(self: Self, data_config: DataConfig):
        super().__init__(data_config)

    def prepare_ranker_dataset(
        self: Self,
        filter_config: Dict[str, Any] = None,
        is_rank: bool = True,
        is_csr: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Load and process training data for ranking models.

        Args:
            filter_config (Dict[str, Any]): Filter config used when filtering reviews.
            is_rank (bool): Indicator if it is ranking model or not. Defaults to True.
            is_csr (bool): Indicator if csr format or not for als model.
            **kwargs: Additional keyword arguments

        Returns (Dict[str, Any]):
            A dictionary containing the training and validation sets.
        """
        prepared_data = self.prepare_train_val_dataset(
            filter_config=filter_config,
            is_rank=is_rank,
            is_csr=is_csr,
            **kwargs,
        )

        train = prepared_data["train"]
        val = prepared_data["val"]
        test = prepared_data["test"]
        user_feature = prepared_data["user_feature"]
        diner_feature = prepared_data["diner_feature"]
        mapped_res = prepared_data["mapped_res"]

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
            df: pd.DataFrame
            user_feature: pd.DataFrame
            diner_feature: pd.DataFrame

        Returns:
            pd.DataFrame: DataFrame with merged features
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
            sampling_type: str
            num_neg_samples: int
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
    data_loader = RankerDatasetLoader(data_config=DataConfig(**cfg.data))
    data = data_loader.prepare_ranker_dataset(
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
