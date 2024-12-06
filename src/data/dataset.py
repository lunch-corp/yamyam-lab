import glob
import os

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# Load data (same as your current implementation)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device.type)

# Load data
review_data_paths = glob.glob(os.path.join(DATA_PATH, "review", "*.csv"))


def load_and_prepare_graph_data(test_size, min_reviews):
    review = pd.DataFrame()
    for review_data_path in review_data_paths:
        review = pd.concat([review, pd.read_csv(review_data_path)], axis=0)

    # Map diner and reviewer IDs
    diner_idxs = sorted(list(review["diner_idx"].unique()))
    reviewer_ids = sorted(list(review["reviewer_id"].unique()))

    diner_mapping = {diner_idx: i for i, diner_idx in enumerate(diner_idxs)}
    reviewer_mapping = {reviewer_id: i + len(diner_mapping) for i, reviewer_id in enumerate(reviewer_ids)}

    review["diner_idx"] = review["diner_idx"].map(diner_mapping)
    review["reviewer_id"] = review["reviewer_id"].map(reviewer_mapping)

    # Filter reviewers with minimum reviews
    reviewer2review_cnt = review["reviewer_id"].value_counts()
    reviewer_id_over = [r_id for r_id, cnt in reviewer2review_cnt.items() if cnt >= min_reviews]
    review_over = review[review["reviewer_id"].isin(reviewer_id_over)]

    # Split data
    train, val = train_test_split(review_over, test_size=test_size, stratify=review_over["reviewer_id"])

    # Create edge index
    edge_index = torch.tensor([train["diner_idx"].values, train["reviewer_id"].values], dtype=torch.long)

    # Optional: create node features or use identifiers as features
    num_nodes = len(diner_mapping) + len(reviewer_mapping)
    x = torch.eye(num_nodes, device=device)  # One-hot encoding as example; use embeddings if available

    # Labels (edge attributes)
    y = torch.tensor(train["reviewer_review_score"].values, dtype=torch.float).view(-1, 1)

    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=y).to(device)
    return data


def load_and_prepare_lightgbm_data(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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
    diner = pd.read_csv(os.path.join(DATA_PATH, "diner/diner_df_20241204_yamyam.csv"))

    review = pd.DataFrame()
    for review_data_path in review_data_paths:
        review = pd.concat([review, pd.read_csv(review_data_path)], axis=0)

    review["reviewer_review_cnt"] = review["reviewer_review_cnt"].apply(lambda x: np.int32(str(x).replace(",", "")))
    review = pd.merge(review, diner, on="diner_idx", how="inner")

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
    reviewer_id_over = [reviewer_id for reviewer_id, cnt in reviewer2review_cnt.items() if cnt >= cfg.data.min_reviews]
    review_over = review[lambda x: x["reviewer_id"].isin(reviewer_id_over)]

    # 사용자 ID를 train과 valid로 분리
    train, valid = train_test_split(review_over, test_size=0.2, random_state=42)

    # 사용자 ID를 기준으로 데이터 나누기
    # train, val = train_test_split(review_over, test_size=cfg.data.test_size, random_state=cfg.data.random_state)
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
    diner = pd.read_csv(os.path.join(DATA_PATH, "diner/diner_df_20241204_yamyam.csv"))
    review = pd.DataFrame()

    for review_data_path in review_data_paths:
        review = pd.concat([review, pd.read_csv(review_data_path, index_col=0)], axis=0)

    review["reviewer_review_cnt"] = review["reviewer_review_cnt"].apply(lambda x: np.int32(str(x).replace(",", "")))

    reviewer_id = cfg.user_name
    review = pd.merge(review, diner, on="diner_idx", how="inner")

    # 사용자별 리뷰한 레스토랑 ID 목록 생성
    user_2_diner_df = review.groupby("reviewer_id").agg({"diner_category_middle": list})
    user_2_diner_map = dict(zip(user_2_diner_df.index, user_2_diner_df["diner_category_middle"]))

    # 레스토랑 후보군 리스트
    candidate_pool = diner["diner_idx"].unique().tolist()

    reviewed_diners = list(set(user_2_diner_map.get(reviewer_id, [])))
    candidates = [d for d in candidate_pool if d not in reviewed_diners]
    candidates = np.random.choice(candidates, size=cfg.data.size)  # candidate choice
    review = review[review["reviewer_id"] == reviewer_id].iloc[-1:]  # 마지막 리뷰 정보만 사용

    # Create test data
    test = pd.DataFrame({"reviewer_id": reviewer_id, "diner_idx": candidates})
    test = test.merge(diner, on="diner_idx")
    test = test.merge(
        review[["reviewer_id", "badge_level", "reviewer_review_cnt", "reviewer_collected_review_cnt"]],
        on="reviewer_id",
    )
    already_reviewed = user_2_diner_map.get(reviewer_id, [])

    return test, already_reviewed
