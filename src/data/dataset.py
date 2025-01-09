import glob
import os

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder

from preprocess.feature_store import extract_scores_array, extract_statistics

# Load data (same as your current implementation)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device.type)

# Load data
review_data_paths = glob.glob(os.path.join(DATA_PATH, "review", "*.csv"))


def load_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare dataset for training.

    Args:
        cfg (DictConfig): configuration dictionary.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: train features, train target, valid features, valid target.
    """
    # load data
    diner = pd.read_csv(os.path.join(DATA_PATH, "diner/diner_df_20241219_yamyam.csv"))

    # 범주를 정의
    bins = [-1, 0, 10, 50, 200, float("inf")]

    # pd.cut을 사용하여 정수형 범주 생성
    diner["diner_review_cnt_category"] = pd.cut(
        diner["all_review_cnt"], bins=bins, labels=False
    )
    diner["diner_review_cnt_category"] = diner["diner_review_cnt_category"].fillna(0)
    diner["diner_review_cnt_category"] = diner["diner_review_cnt_category"].astype(int)

    # Extract scores
    # Categories for extracting scores
    tag_categories = [
        ("맛", "taste"),
        ("친절", "kind"),
        ("분위기", "mood"),
        ("가성비", "chip"),
        ("주차", "parking"),
    ]

    scores = extract_scores_array(diner["diner_review_tags"], tag_categories)

    # 결과를 DataFrame으로 변환 및 병합
    diner[["taste", "kind", "mood", "chip", "parking"]] = scores

    # 새 컬럼으로 추가 (최소값, 최대값, 평균, 중앙값, 항목 수)
    diner[["min_price", "max_price", "mean_price", "median_price", "menu_count"]] = (
        diner["diner_menu_price"].apply(lambda x: extract_statistics(eval(x)))
    )

    for col in ["min_price", "max_price", "mean_price", "median_price", "menu_count"]:
        diner[col] = diner[col].fillna(diner[col].median())

    review = pd.concat(
        [pd.read_csv(review_data_path) for review_data_path in review_data_paths]
    )
    # review = pd.read_csv(os.path.join(DATA_PATH, "review/review_df_20241219_part_5.csv"))
    review["reviewer_review_cnt"] = review["reviewer_review_cnt"].apply(
        lambda x: np.int32(str(x).replace(",", ""))
    )
    review = pd.merge(review, diner, on="diner_idx", how="inner")
    review = review.drop_duplicates(subset=["reviewer_id", "diner_idx"])

    # label Encoder
    le = LabelEncoder()
    review["badge_grade"] = le.fit_transform(review["badge_grade"])

    # 리뷰어
    review["reviewer_trust_score"] = (
        0.7 * review["reviewer_review_cnt"] + 0.3 * review["badge_level"]
    )

    return review, diner


def load_test_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, list[str]]:
    """
    review_data: DataFrame containing review information
    diner_data: DataFrame containing diner information

    Returns:
        - user_2_diner_map: Mapping of user IDs to reviewed diner IDs
        - candidate_pool: List of all diner IDs
        - diner_id_2_name_map: Mapping of diner IDs to their names
    """
    # load data
    review, diner = load_dataset(cfg)

    reviewer_id = cfg.user_name

    # 사용자별 리뷰한 레스토랑 ID 목록 생성
    user_2_diner_df = review.groupby("reviewer_id").agg({"diner_idx": list})
    user_2_diner_map = dict(zip(user_2_diner_df.index, user_2_diner_df["diner_idx"]))

    # 레스토랑 후보군 리스트
    candidate_pool = diner["diner_idx"].unique().tolist()

    reviewed_diners = list(set(user_2_diner_map.get(reviewer_id, [])))
    candidates = [d for d in candidate_pool if d not in reviewed_diners]
    review = review[review["reviewer_id"] == reviewer_id].iloc[-1:]

    # Create test data
    test = pd.DataFrame({"reviewer_id": reviewer_id, "diner_idx": candidates})
    test = test.merge(diner, on="diner_idx")
    test = test.merge(
        review[
            [
                "reviewer_user_name",
                "reviewer_id",
                "badge_grade",
                "reviewer_trust_score",
                "badge_level",
                "reviewer_review_cnt",
                "reviewer_collected_review_cnt",
            ]
        ],
        on="reviewer_id",
    )
    test = test.drop_duplicates(subset=["reviewer_id", "diner_idx"])

    test["diner_category_small"] = test["diner_category_small"].fillna(
        test["diner_category_middle"]
    )
    already_reviewed = user_2_diner_map.get(reviewer_id, [])

    return test, already_reviewed
