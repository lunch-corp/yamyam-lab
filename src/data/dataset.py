from typing import Any, Dict, Tuple

import pandas as pd

from data.validator import DataValidator
from tools.google_drive import ensure_data_files


def load_dataset(test: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load review, diner, and diner with raw category data, and optionally filter for pytest.
    In this function, no other preprocessing logic is done but only loading data will be run.

    Args:
        test (bool): When set true, subset of review data will be used.

    Returns (Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]):
        review, diner, diner with raw category in order.
    """
    data_paths = ensure_data_files()

    review = pd.read_csv(data_paths["review"])
    reviewer = pd.read_csv(data_paths["reviewer"])

    review = pd.merge(review, reviewer, on="reviewer_id", how="left")

    diner = pd.read_csv(data_paths["diner"], low_memory=False)
    diner_with_raw_category = pd.read_csv(data_paths["category"])

    if test:
        yongsan_diners = diner[
            lambda x: x["diner_road_address"].str.startswith("서울 용산구", na=False)
        ]["diner_idx"].unique()[:100]
        review = review[
            lambda x: x["diner_idx"].isin(yongsan_diners)
        ]  # about 5000 rows

    return review, diner, diner_with_raw_category


def load_test_dataset(
    reviewer_id: int,
    user_feature_param_pair: Dict[str, Any],
    diner_feature_param_pair: Dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load test dataset for inference
    params:
        reviewer_id: int

    return:
        test: pd.DataFrame
        already_reviewed: list[str]
    """
    # 순환 참조 방지
    from preprocess.preprocess import make_feature

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
    user_feature, diner_feature, _ = make_feature(
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
