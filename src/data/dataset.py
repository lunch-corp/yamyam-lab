import glob
import os

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data (same as your current implementation)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device.type)

# Load data
review_data_paths = glob.glob(os.path.join(DATA_PATH, "review", "*.csv"))
# Categories for extracting scores
CATEGORIES = [
    ("맛", "taste"),
    ("친절", "kind"),
    ("분위기", "mood"),
    ("가성비", "chip"),
    ("주차", "parking"),
]


# NaN 또는 빈 리스트를 처리할 수 있도록 정의
def extract_statistics(prices: list[int, float]) -> pd.Series:
    if not prices:  # 빈 리스트라면 NaN 반환
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
    return pd.Series(
        [min(prices), max(prices), np.mean(prices), np.median(prices), len(prices)]
    )


# numpy 기반으로 점수 추출 최적화
def extract_scores_array(reviews: str, categories: list[tuple[str, str]]) -> np.ndarray:
    # 리뷰 데이터를 파싱하여 배열로 변환
    parsed = [eval(review) for review in reviews]
    # 카테고리별 점수 초기화 (rows x categories)
    scores = np.zeros((len(reviews), len(categories)), dtype=int)

    # 각 리뷰에서 카테고리 점수 추출
    category_map = {cat: idx for idx, (cat, _) in enumerate(categories)}
    for row_idx, review in enumerate(parsed):
        for cat, score in review:
            if cat in category_map:  # 해당 카테고리가 정의된 경우
                scores[row_idx, category_map[cat]] = score

    return scores


def load_and_prepare_lightgbm_data(
    cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    test_size: ratio of test dataset
    min_reviews: minimum number of reviews for each reviewer
    X_columns: column names for model feature
    y_columns: column names for target value
    use_columns: columns to use in review data
    random_state: random seed for reproducibility
    stratify: column to stratify review data
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
    scores = extract_scores_array(diner["diner_review_tags"], CATEGORIES)

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

    # target value 생성
    review["target"] = np.where(
        (review["real_good_review_percent"] > review["real_bad_review_percent"])
        & (review["reviewer_review_score"] - review["reviewer_avg"] > 0.5),
        1,
        0,
    )

    del diner

    # store unique number of diner and reviewer
    diner_idxs = sorted(list(review["diner_idx"].unique()))
    reviewer_ids = sorted(list(review["reviewer_id"].unique()))

    # mapping diner_idx and reviewer_id
    diner_mapping = {diner_idx: i for i, diner_idx in enumerate(diner_idxs)}
    reviewer_mapping = {reviewer_id: i for i, reviewer_id in enumerate(reviewer_ids)}
    review["diner_idx"] = review["diner_idx"].map(diner_mapping)
    review["reviewer_id"] = review["reviewer_id"].map(reviewer_mapping)

    # filter reviewer who wrote reviews more than min_reviews
    reviewer2review_cnt = review["reviewer_id"].value_counts().to_dict()
    reviewer_id_over = [
        reviewer_id
        for reviewer_id, cnt in reviewer2review_cnt.items()
        if cnt >= cfg.data.min_reviews
    ]
    review_over = review[lambda x: x["reviewer_id"].isin(reviewer_id_over)]
    review_over = review_over.drop_duplicates(subset=["reviewer_id", "diner_idx"])

    # 사용자 ID를 train과 valid로 분리
    user_ids = review_over["reviewer_id"].unique()
    train_user_ids, valid_user_ids = train_test_split(
        user_ids, test_size=cfg.data.test_size, random_state=cfg.data.random_state
    )

    train = review_over[lambda x: x["reviewer_id"].isin(train_user_ids)]
    valid = review_over[lambda x: x["reviewer_id"].isin(valid_user_ids)]
    train = train.sort_values(by=["reviewer_id"])
    valid = valid.sort_values(by=["reviewer_id"])

    X_train, y_train = train.drop(columns=[cfg.data.target]), train[cfg.data.target]
    X_valid, y_valid = valid.drop(columns=[cfg.data.target]), valid[cfg.data.target]

    return X_train, y_train, X_valid, y_valid


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
    scores = extract_scores_array(diner["diner_review_tags"], CATEGORIES)

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
