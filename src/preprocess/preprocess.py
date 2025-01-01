from typing import Tuple, List, Dict, Any
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import networkx as nx

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

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


def train_test_split_stratify(
    test_size: float,
    min_reviews: int,
    X_columns: List[str] = ["diner_idx", "reviewer_id"],
    y_columns: List[str] = ["reviewer_review_score"],
    random_state: int = 42,
    stratify: str = "reviewer_id",
    pg_model: bool = False,
) -> Dict[str, Any]:
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

    Returns (Dict[str, Any]):
        Dataset, statistics, and mapping information which could be used when training model.
    """
    # load data
    review_1 = pd.read_csv(
        os.path.join(DATA_PATH, "review/review_df_20241219_part_1.csv")
    )
    review_2 = pd.read_csv(
        os.path.join(DATA_PATH, "review/review_df_20241219_part_2.csv")
    )
    review_3 = pd.read_csv(
        os.path.join(DATA_PATH, "review/review_df_20241219_part_3.csv")
    )
    review_4 = pd.read_csv(
        os.path.join(DATA_PATH, "review/review_df_20241219_part_4.csv")
    )
    review_5 = pd.read_csv(
        os.path.join(DATA_PATH, "review/review_df_20241219_part_5.csv")
    )
    review = pd.concat([review_1, review_2, review_3, review_4, review_5], axis=0)[
        X_columns + y_columns
    ]
    del review_1
    del review_2
    del review_3
    del review_4
    del review_5

    # filter diner in review dataset not existing in diner dataset
    # TODO: add this step as data validation
    diner = pd.read_csv(os.path.join(DATA_PATH, "diner/diner_df_20241219_yamyam.csv"))
    diner_idx_both_exist = np.array(
        list(set(review["diner_idx"].unique()) & set(diner["diner_idx"].unique()))
    )
    review = review[lambda x: x["diner_idx"].isin(diner_idx_both_exist)]

    # filter reviewer who wrote reviews more than min_reviews
    reviewer2review_cnt = review["reviewer_id"].value_counts().to_dict()
    reviewer_id_over = [
        reviewer_id
        for reviewer_id, cnt in reviewer2review_cnt.items()
        if cnt >= min_reviews
    ]
    review = review[lambda x: x["reviewer_id"].isin(reviewer_id_over)]

    # store unique number of diner and reviewer
    diner_idxs = sorted(list(review["diner_idx"].unique()))
    reviewer_ids = sorted(list(review["reviewer_id"].unique()))

    num_diners = len(diner_idxs)
    num_reviewers = len(reviewer_ids)

    # mapping diner_idx and reviewer_id
    diner_mapping = {diner_idx: i for i, diner_idx in enumerate(diner_idxs)}

    if pg_model is True:
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

    train, val = train_test_split(
        review,
        test_size=test_size,
        random_state=random_state,
        stratify=review[stratify],
    )
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
