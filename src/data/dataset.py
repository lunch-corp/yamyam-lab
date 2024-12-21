import glob
import os

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

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
    return pd.Series([min(prices), max(prices), np.mean(prices), np.median(prices), len(prices)])


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
    diner["review_cnt_category"] = pd.cut(diner["all_review_cnt"], bins=bins, labels=False)  # 레이블 대신 정수를 반환

    diner["diner_blog_review_cnt"] = np.log1p(diner["diner_blog_review_cnt"])

    # Extract review scores and add columns
    scores = extract_scores_array(diner["diner_review_tags"], CATEGORIES)
    diner[["taste", "kind", "mood", "chip", "parking"]] = scores

    # Extract menu statistics and fill missing values
    stats_columns = ["min_price", "max_price", "mean_price", "median_price", "menu_count"]
    diner[stats_columns] = diner["diner_menu_price"].apply(lambda x: extract_statistics(eval(x)))

    for col in stats_columns:
        diner[col] = diner[col].fillna(diner[col].median())

    # Load and concatenate review data
    review = pd.concat([pd.read_csv(path) for path in review_data_paths], axis=0)

    # Clean and process review data
    review["reviewer_review_cnt"] = review["reviewer_review_cnt"].apply(lambda x: int(str(x).replace(",", "")))
    review = pd.merge(review, diner, on="diner_idx", how="inner")
    review.drop_duplicates(subset=["reviewer_id", "diner_idx"], inplace=True)

    # Encode categorical features
    review["badge_grade"] = LabelEncoder().fit_transform(review["badge_grade"])

    # Create target variable
    review["target"] = np.where(
        (review["reviewer_avg"] > 3)
        & (review["reviewer_review_score"] - review["reviewer_avg"] > -0.5)
        & (review["reviewer_review_score"] - review["reviewer_avg"] < 0.5),
        1,
        0,
    )
    print(review["target"].value_counts())

    # define reviewer trust score
    review["reviewer_trust_score"] = 0.7 * review["reviewer_review_cnt"] + 0.3 * review["badge_level"]

    # Map diner and reviewer IDs to unique indices
    diner_mapping = {d: i for i, d in enumerate(sorted(review["diner_idx"].unique()))}
    reviewer_mapping = {r: i for i, r in enumerate(sorted(review["reviewer_id"].unique()))}
    review["diner_idx"] = review["diner_idx"].map(diner_mapping)
    review["reviewer_id"] = review["reviewer_id"].map(reviewer_mapping)

    # Filter reviewers with enough reviews
    reviewer_counts = review["reviewer_id"].value_counts()
    valid_reviewers = reviewer_counts[reviewer_counts >= cfg.data.min_reviews].index
    review = review[review["reviewer_id"].isin(valid_reviewers)]

    # Split reviewers into train and validation sets
    user_ids = review["reviewer_id"].unique()
    train_users, valid_users = train_test_split(
        user_ids, test_size=cfg.data.test_size, random_state=cfg.data.random_state
    )

    train = review[review["reviewer_id"].isin(train_users)]
    valid = review[review["reviewer_id"].isin(valid_users)]
    # Split features and target
    X_train, y_train = train.drop(columns=["target"]), train["target"]
    X_valid, y_valid = valid.drop(columns=["target"]), valid["target"]

    return X_train, y_train, X_valid, y_valid


def load_test_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, list[str]]:
    """
    Loads and processes test dataset for restaurant recommendation.

    Args:
        cfg: Configuration containing user-specific settings.

    Returns:
        tuple:
            - test: DataFrame containing test dataset with candidate restaurants.
            - already_reviewed: List of restaurant IDs already reviewed by the user.
    """
    # load data
    diner = pd.read_csv(os.path.join(DATA_PATH, "diner/diner_df_20241219_yamyam.csv"))

    # 범주를 정의
    bins = [-1, 0, 10, 50, 200, float("inf")]

    # pd.cut을 사용하여 정수형 범주 생성
    diner["review_cnt_category"] = pd.cut(diner["all_review_cnt"], bins=bins, labels=False)  # 레이블 대신 정수를 반환
    # log 변환환
    diner["diner_blog_review_cnt"] = np.log1p(diner["diner_blog_review_cnt"])

    # Extract review scores and add columns
    scores = extract_scores_array(diner["diner_review_tags"], CATEGORIES)
    diner[["taste", "kind", "mood", "chip", "parking"]] = scores

    # Extract menu statistics and fill missing values
    stats_columns = ["min_price", "max_price", "mean_price", "median_price", "menu_count"]
    diner[stats_columns] = diner["diner_menu_price"].apply(lambda x: extract_statistics(eval(x)))

    for col in stats_columns:
        diner[col] = diner[col].fillna(diner[col].median())

    # Load and concatenate review data
    review = pd.concat([pd.read_csv(path) for path in review_data_paths], axis=0)

    # Clean and process review data
    review["reviewer_review_cnt"] = review["reviewer_review_cnt"].apply(lambda x: int(str(x).replace(",", "")))
    review = pd.merge(review, diner, on="diner_idx", how="inner")
    review.drop_duplicates(subset=["reviewer_id", "diner_idx"], inplace=True)

    # Encode categorical features
    review["badge_grade"] = LabelEncoder().fit_transform(review["badge_grade"])

    # define reviewer trust score
    review["reviewer_trust_score"] = 0.7 * review["reviewer_review_cnt"] + 0.3 * review["badge_level"]

    # Map users to their reviewed diner IDs
    user_2_diner_df = review.groupby("reviewer_id").agg({"diner_idx": list})
    user_2_diner_map = user_2_diner_df["diner_idx"].to_dict()

    # Prepare candidate pool
    candidate_pool = diner["diner_idx"].unique().tolist()

    reviewer_id = cfg.user_name
    already_reviewed = user_2_diner_map.get(reviewer_id, [])
    candidates = [diner_id for diner_id in candidate_pool if diner_id not in already_reviewed]

    # Filter the most recent review by the user
    recent_review = review[review["reviewer_id"] == reviewer_id].iloc[-1:]

    # Create test dataset
    test = (
        pd.DataFrame({"reviewer_id": reviewer_id, "diner_idx": candidates})
        .merge(diner, on="diner_idx")
        .merge(
            recent_review[
                [
                    "reviewer_user_name",
                    "reviewer_id",
                    "badge_grade",
                    "badge_level",
                    "reviewer_trust_score",
                    "reviewer_review_cnt",
                ]
            ],
            on="reviewer_id",
        )
        .drop_duplicates(subset=["reviewer_id", "diner_idx"])
    )

    # Fill missing category data
    test["diner_category_small"] = test["diner_category_small"].fillna(test["diner_category_middle"])

    return test, already_reviewed
