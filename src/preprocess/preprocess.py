import glob
import os
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from preprocess.feature_store import extract_scores_array, extract_statistics

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")


class TorchData(Dataset):
    def __init__(self, X: Tensor, y: Tensor):
        """
        Make Dataset object especially for pytorch DataLoader.
        This Dataset class will be input for pytorch DataLoader.

        Args:
            X (Tensor): input features.
            y (Tensor): target features.
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(
    review_paths: List[str], diner_path: str, test: bool = False
) -> pd.DataFrame:
    """
    Load review and diner data, and optionally filter for pytest.
    """
    review = pd.concat([pd.read_csv(path) for path in review_paths])

    if test:
        review = review.iloc[:5000, :]

    diner = pd.read_csv(diner_path)
    diner_idx_both_exist = set(review["diner_idx"].unique()) & set(
        diner["diner_idx"].unique()
    )

    review = review[review["diner_idx"].isin(diner_idx_both_exist)]

    return review, diner


def filter_reviewers(review: pd.DataFrame, min_reviews: int) -> pd.DataFrame:
    """
    Filter reviewers who have written more than `min_reviews` reviews.
    """
    reviewer_counts = review["reviewer_id"].value_counts()
    valid_reviewers = reviewer_counts[reviewer_counts >= min_reviews].index
    return review[review["reviewer_id"].isin(valid_reviewers)]


def preprocess_diner_data(diner: pd.DataFrame) -> pd.DataFrame:
    """
    Add categorical and statistical features to the diner dataset.
    """
    bins = [-1, 0, 10, 50, 200, float("inf")]
    diner["diner_review_cnt_category"] = (
        pd.cut(diner["all_review_cnt"], bins=bins, labels=False).fillna(0).astype(int)
    )

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

    return diner


def create_target_column(review: pd.DataFrame) -> pd.DataFrame:
    """
    Create the target column for classification.
    """
    review["target"] = np.where(
        (review["real_good_review_percent"] > review["real_bad_review_percent"])
        & (review["reviewer_review_score"] - review["reviewer_avg"] > 0.5),
        1,
        0,
    )
    return review


def train_test_split_stratify(
    test_size: float,
    min_reviews: int,
    X_columns: List[str] = ["diner_idx", "reviewer_id"],
    y_columns: List[str] = ["reviewer_review_score"],
    random_state: int = 42,
    stratify: str = "reviewer_id",
    pg_model: bool = False,
    test: bool = False,
    is_rank: bool = False,
) -> Union[Dict[str, Any], Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """
    Split review data stratifying by `stratify` column.
    This function aims for using consistent train / validation dataset across coders
    and ensures that each reviewer in validation is included in train dataset.

    Args:
        test_size (float): ratio of test dataset.
        min_reviews (int): minimum number of reviews for each reviewer.
        X_columns (List[str]): column names for model feature.
        y_columns (List[str]): column names for target value.
        random_state (int): random seed for reproducibility.
        stratify (str): reference column when stratifying review data.
        pg_model (bool): indicator whether using torch_geometric model or not.
        test (bool): indicator whether under pytest. when set true, use part of total dataset.

    Returns (Dict[str, Any]):
        Dataset, statistics, and mapping information which could be used when training model.
    """
    # Load and preprocess data
    review_paths = glob.glob(os.path.join(DATA_PATH, "review", "*.csv"))
    diner_path = os.path.join(DATA_PATH, "diner/diner_df_20241219_yamyam.csv")

    review, diner = load_dataset(review_paths, diner_path, test)
    review = filter_reviewers(review, min_reviews)

    # store unique number of diner and reviewer
    diner_idxs = sorted(list(review["diner_idx"].unique()))
    reviewer_ids = sorted(list(review["reviewer_id"].unique()))

    num_diners = len(diner_idxs)
    num_reviewers = len(reviewer_ids)

    # mapping diner_idx and reviewer_id
    diner_mapping = {diner_idx: i for i, diner_idx in enumerate(diner_idxs)}

    if pg_model:
        # each node index in torch_geometric should be unique
        reviewer_mapping = {
            reviewer_id: (i + num_diners) for i, reviewer_id in enumerate(reviewer_ids)
        }

    else:
        reviewer_mapping = {
            reviewer_id: i for i, reviewer_id in enumerate(reviewer_ids)
        }

    review["diner_idx"] = review["diner_idx"].map(diner_mapping)
    review["reviewer_id"] = review["reviewer_id"].map(reviewer_mapping)

    diner["diner_idx"] = diner["diner_idx"].map(diner_mapping)

    train, val = train_test_split(
        review,
        test_size=test_size,
        random_state=random_state,
        stratify=review[stratify],
    )

    if is_rank:
        # label Encoder
        le = LabelEncoder()
        train["badge_grade"] = le.fit_transform(train["badge_grade"])
        val["badge_grade"] = le.transform(val["badge_grade"])

        # Calculate reviewer trust score
        train["reviewer_trust_score"] = (
            0.7 * train["reviewer_review_cnt"] + 0.3 * train["badge_level"]
        )
        val["reviewer_trust_score"] = (
            0.7 * val["reviewer_review_cnt"] + 0.3 * val["badge_level"]
        )

        # Preprocess and merge diner data
        diner = preprocess_diner_data(diner)
        train = train.merge(diner, on="diner_idx", how="inner").drop_duplicates(
            subset=["reviewer_id", "diner_idx"]
        )
        val = val.merge(diner, on="diner_idx", how="inner").drop_duplicates(
            subset=["reviewer_id", "diner_idx"]
        )

        # Create target column
        train = create_target_column(train)
        val = create_target_column(val)

        train = train.sort_values(by=[stratify])
        val = val.sort_values(by=[stratify])

        X_train, y_train = train.drop(columns=["target"]), train["target"]
        X_val, y_val = val.drop(columns=["target"]), val["target"]

        return X_train, y_train, X_val, y_val

    # check whether reviewers from train is equivalent with reviewers from val
    assert np.array_equal(
        np.sort(train["reviewer_id"].unique()), np.sort(val["reviewer_id"].unique())
    )
    # TODO: check whether diners from train is equivalent with diners from val

    return {
        "X_train": torch.tensor(train[X_columns].values),
        "y_train": torch.tensor(train[y_columns].values, dtype=torch.float32),
        "X_val": torch.tensor(val[X_columns].values),
        "y_val": torch.tensor(val[y_columns].values, dtype=torch.float32),
        "num_diners": num_diners,
        "num_users": num_reviewers,
        "diner_mapping": diner_mapping,
        "user_mapping": reviewer_mapping,
    }


def prepare_torch_dataloader(
    X_train: Tensor,
    y_train: Tensor,
    X_val: Tensor,
    y_val: Tensor,
    batch_size: int = 128,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Make train / validation pytorch DataLoader.
    This function gets input from `train_test_split_stratify` function.

    Args:
        X_train (Tensor): input features used when training model.
        y_train (Tensor): target features used when training model.
        X_val (Tensor): input features used when validating model.
        y_val (Tensor): target features used when validating model.
        batch_size (int): batch size for mini-batch gradient descent.
        random_state (int): random seed for reproducibility.

    Returns (Tuple[DataLoader, DataLoader]):
        Train / validation dataloader.
    """
    # seed = torch.Generator(device=device.type).manual_seed(random_state)

    train_dataset = TorchData(X_train, y_train)
    val_dataset = TorchData(X_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader


def prepare_torch_geometric_data(
    X_train: Tensor, X_val: Tensor, num_diners: int, num_reviewers: int
) -> Tuple[Data, Data]:
    """
    Make train / validation Dataset especially for pytorch geometric package.
    This function gets input from `train_test_split_stratify` function.
    Note that currently, only edge relation info is integrated into data

    Args:
        X_train (Tensor): input features used when training pytorch geometric model.
        X_val (Tensor): input features used when validating pytorch geometric model.
        num_diners (int): number of unique diners.
        num_reviewers (int): number of unique reviewers.

    Returns (Tuple[Data, Data]):
        Train / validation pytorch geometric dataset.
    """
    # check feature data has only two columns, e.g., diner_id and reviewer_id
    assert X_train.shape[1] == 2
    assert X_val.shape[1] == 2

    # make edges for undirected graph
    edge_index_train = torch.concat((X_train, X_train[:, [1, 0]]), axis=0).T
    edge_index_val = torch.concat((X_val, X_val[:, [1, 0]]), axis=0).T

    # make pg data
    train = Data(edge_index=edge_index_train, num_nodes=num_diners + num_reviewers)
    val = Data(edge_index=edge_index_val, num_nodes=num_diners + num_reviewers)

    return train, val


def prepare_networkx_data(X_train: Tensor, X_val: Tensor) -> Tuple[nx.Graph, nx.Graph]:
    """
    Make train / validation dataset in nx.Graph object type.

    Args:
        X_train (Tensor): input features used when training model.
        X_val (Tensor): input features used when validating model.

    Returns (Tuple[nx.Graph, nx.Graph]):
        Train / validation dataset in nx.Graph object type.
    """
    train_graph = nx.Graph()
    val_graph = nx.Graph()

    for diner_id, reviewer_id in X_train:
        train_graph.add_edge(diner_id.item(), reviewer_id.item())

    for diner_id, reviewer_id in X_val:
        val_graph.add_edge(diner_id.item(), reviewer_id.item())

    return train_graph, val_graph
