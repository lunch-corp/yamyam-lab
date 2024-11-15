import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# Load data (same as your current implementation)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device.type)


def load_and_prepare_graph_data(test_size, min_reviews):
    # Load data
    review_1 = pd.read_csv(os.path.join(DATA_PATH, "review_df_20241107_071929_yamyam_1.csv"))
    review_2 = pd.read_csv(os.path.join(DATA_PATH, "review_df_20241107_071929_yamyam_2.csv"))
    review = pd.concat([review_1, review_2], axis=0)
    del review_1, review_2

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
    test_size,
    min_reviews,
    X_columns=["diner_idx", "reviewer_id"],
    y_columns=["reviewer_review_score"],
    random_state=42,
    stratify="reviewer_id",
):
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
    review_1 = pd.read_csv(os.path.join(DATA_PATH, "review_df_20241107_071929_yamyam_1.csv"), index_col=0)
    review_2 = pd.read_csv(os.path.join(DATA_PATH, "review_df_20241107_071929_yamyam_2.csv"), index_col=0)
    diner = pd.read_csv(os.path.join(DATA_PATH, "diner_df_20241107_071929_yamyam.csv"), index_col=0)
    review = pd.concat([review_1, review_2], axis=0)

    review = pd.merge(review, diner, on="diner_idx", how="inner")

    del review_1, review_2, diner

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
    reviewer_id_over = [reviewer_id for reviewer_id, cnt in reviewer2review_cnt.items() if cnt >= min_reviews]
    review_over = review[lambda x: x["reviewer_id"].isin(reviewer_id_over)]

    train, val = train_test_split(
        review_over, test_size=test_size, random_state=random_state, stratify=review_over[stratify]
    )

    X_train, y_train = train.drop(columns=y_columns), train[y_columns]
    X_val, y_val = val.drop(columns=y_columns), val[y_columns]

    return X_train, y_train, X_val, y_val
