import os

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

# Load data (same as your current implementation)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device.type)


def load_train_dataset(
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
    review_1 = pd.read_csv(os.path.join(DATA_PATH, "review/review_df_20241122_part_1.csv"))
    review_2 = pd.read_csv(os.path.join(DATA_PATH, "review/review_df_20241122_part_2.csv"))
    review_3 = pd.read_csv(os.path.join(DATA_PATH, "review/review_df_20241122_part_3.csv"))
    review = pd.concat([review_1, review_2, review_3], axis=0)
    review[cfg.data.target] = np.where(review[cfg.data.target] >= 4, 1, 0)
    diner = pd.read_csv(os.path.join(DATA_PATH, "diner/diner_df_20241122_yamyam.csv"))

    review["reviewer_review_cnt"] = review["reviewer_review_cnt"].apply(lambda x: np.int32(str(x).replace(",", "")))
    review = pd.merge(review, diner, on="diner_idx", how="inner")
    # del review_1, review_2, diner

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
    reviewer_id_over = [reviewer_id for reviewer_id, cnt in reviewer2review_cnt.items() if cnt >= cfg.data.min_reviews]
    review_over = review[lambda x: x["reviewer_id"].isin(reviewer_id_over)]

    # 사용자 ID를 고유값으로 추출
    unique_users = review_over["reviewer_id"].unique()

    # 사용자 ID를 train과 valid로 분리
    train_users, valid_users = train_test_split(unique_users, test_size=0.2, random_state=42)

    # 사용자 ID를 기준으로 데이터 나누기
    train = review_over[review_over["reviewer_id"].isin(train_users)]
    valid = review_over[review_over["reviewer_id"].isin(valid_users)]
    # train, val = train_test_split(review_over, test_size=cfg.data.test_size, random_state=cfg.data.random_state)
    X_train, y_train = train.drop(columns=[cfg.data.target]), train[cfg.data.target]
    X_valid, y_valid = valid.drop(columns=[cfg.data.target]), valid[cfg.data.target]

    return X_train, y_train, X_valid, y_valid


def load_test_dataset(cfg: DictConfig) -> pd.DataFrame:
    # load data
    review_1 = pd.read_csv(os.path.join(DATA_PATH, "review/review_df_20241122_part_1.csv"))
    review_2 = pd.read_csv(os.path.join(DATA_PATH, "review/review_df_20241122_part_2.csv"))
    review_3 = pd.read_csv(os.path.join(DATA_PATH, "review/review_df_20241122_part_3.csv"))
    review = pd.concat([review_1, review_2, review_3], axis=0)

    diner = pd.read_csv(os.path.join(DATA_PATH, "diner/diner_df_20241122_yamyam.csv"))
    diner = diner.loc[diner["diner_address_constituency"] == "서울 강남구"]

    review["reviewer_review_cnt"] = review["reviewer_review_cnt"].apply(lambda x: np.int32(str(x).replace(",", "")))

    reviewer_id = cfg.user_name
    review = pd.merge(review, diner, on="diner_idx", how="inner")
    review = review[review["reviewer_id"] == reviewer_id].iloc[-1:]

    # 사용자별 리뷰한 레스토랑 ID 목록 생성
    user_2_diner_df = review.groupby("reviewer_id").agg({"diner_idx": lambda x: list(set(x))})
    user_2_diner_map = dict(zip(user_2_diner_df.index, user_2_diner_df["diner_idx"]))

    # 레스토랑 후보군 리스트
    candidate_pool = diner["diner_idx"].unique().tolist()

    reviewed_diners = list(set(user_2_diner_map.get(reviewer_id, [])))
    candidates = [d for d in candidate_pool if d not in reviewed_diners]
    candidates = np.random.choice(candidates, size=cfg.data.size)  # candidate choice

    # Create test data
    test = pd.DataFrame({"reviewer_id": "이욱", "diner_idx": candidates})
    test["badge_level"] = 0
    test["reviewer_review_cnt"] = 0
    test["reviewer_collected_review_cnt"] = 0

    test = test.merge(diner, on="diner_idx")
    # test = test.merge(
    #     review[["reviewer_id", "badge_level", "reviewer_review_cnt", "reviewer_collected_review_cnt"]],
    #     on="reviewer_id",
    # )

    return test
