import os
from typing import Any, Dict, List, Tuple

import networkx as nx
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from data.validator import DataValidator
from preprocess.diner_transform import CategoryProcessor

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


def preprocess_common(
    review: pd.DataFrame,
    diner: pd.DataFrame,
    diner_with_raw_category: pd.DataFrame,
    min_reviews: int,
    is_timeseries_by_time_point: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Common preprocessing identically applied to ranking and candidate generation.

    Args:
        review (pd.DataFrame): Review dataset.
        diner (pd.DataFrame): Diner dataset.
        diner_with_raw_category (pd.DataFrame): Diner dataset with raw category not preprocessed before.
        min_reviews (int): Minimum number of reviews for each reviewers.

    Returns (Tuple[pd.DataFrame, pd.DataFrame]):
        Preprocessed review dataset and diner dataset.
    """
    # step 0: validate data
    data_validator = DataValidator()
    review = data_validator.validate(review, name_of_df="review")
    diner = data_validator.validate(diner, name_of_df="diner")
    diner_with_raw_category = data_validator.validate(
        diner_with_raw_category, name_of_df="category"
    )

    # step 1: filter reviewers writing reviews greater than or equal to `min_reviews`
    if not is_timeseries_by_time_point:
        reviewer_counts = review["reviewer_id"].value_counts()
        valid_reviewers = reviewer_counts[reviewer_counts >= min_reviews].index
        review = review[review["reviewer_id"].isin(valid_reviewers)]

    # step 2: use diner which has at least one review
    diner_idx_both_exist = set(review["diner_idx"].unique()) & set(
        diner["diner_idx"].unique()
    )
    review = review[review["diner_idx"].isin(diner_idx_both_exist)]
    diner = diner[diner["diner_idx"].isin(diner_idx_both_exist)]

    # step 3: preprocess diner categories and merge them with diner data
    category_columns = [
        "diner_category_large",
        "diner_category_middle",
        "diner_category_small",
        "diner_category_detail",
    ]

    if all(col in diner.columns for col in category_columns):
        # `category_columns`을 제외한 컬럼 목록 생성 (집합 연산으로 최적화)
        columns_exclude_category_columns = list(
            set(diner.columns) - set(category_columns)
        )
        diner = diner[columns_exclude_category_columns]

    processor = CategoryProcessor(diner_with_raw_category)
    diner_with_processd_category = processor.process_all().df

    diner = pd.merge(
        left=diner,
        right=diner_with_processd_category,
        how="left",
        on="diner_idx",
    )

    # step 4: temporary na filling
    diner["diner_category_large"] = diner["diner_category_large"].fillna("NA")
    diner["diner_category_middle"] = diner["diner_category_middle"].fillna("NA")

    # step 5: convert reviewer_review_date into datetime object
    review["reviewer_review_date"] = pd.to_datetime(
        review["reviewer_review_date"].fillna("2015-01-01")
    )

    return review, diner


def reviewer_diner_mapping(
    review: pd.DataFrame,
    diner: pd.DataFrame,
    is_graph_model: bool = False,
) -> Dict[str, Any]:
    """
    Map reviewer_id, diner_idx to integer in ascending order.
    In raw data, reviewer_id and diner_idx are integer but their digit lengths are too long.
    When training model, integer in ascending order is easier to handler and more efficient than current id format.

    Args:
        review (pd.DataFrame): Review dataset.
        diner (pd.DataFrame): Diner dataset.
        is_graph_model (bool): Indicator whether target model is graph based model or not.
            When set true, all the mapped id should be unique.

    Returns (Dict[str, Any]):
        Mapped result.
    """
    diner_mapping, diner = map_id_to_ascending_integer(
        id_column="diner_idx",
        data=diner,
        start_number=0,
    )
    num_diners = len(diner_mapping)
    start_number = num_diners if is_graph_model else 0
    reviewer_mapping, review = map_id_to_ascending_integer(
        id_column="reviewer_id",
        data=review,
        start_number=start_number,
    )
    review["diner_idx"] = review["diner_idx"].map(diner_mapping)
    return {
        "review": review,
        "diner": diner,
        "num_diners": num_diners,
        "num_users": len(reviewer_mapping),
        "diner_mapping": diner_mapping,
        "user_mapping": reviewer_mapping,
    }


def meta_mapping(
    diner: pd.DataFrame,
    num_users: int,
    num_diners: int,
) -> Dict[str, Any]:
    """
    Map meta_id in ascending order.

    Args:
        diner (pd.DataFrame): Diner dataset.
        num_users (int): Number of users.
        num_diners (int): Number of diners.

    Returns (Dict[str, Any]):
        Mapped result.
    """
    meta_ids = list(diner["metadata_id"].unique())
    for meta in diner["metadata_id_neighbors"]:
        meta_ids.extend(meta)
    meta_ids = sorted(list(set(meta_ids)))
    meta_mapping, diner = map_id_to_ascending_integer(
        id_column="metadata_id",
        data=diner,
        start_number=num_diners + num_users,
        unique_ids=meta_ids,
    )
    # additional mapping
    diner["metadata_id_neighbors"] = diner["metadata_id_neighbors"].map(
        lambda x: [meta_mapping[meta] for meta in x]
    )
    return {
        "diner": diner,
        "meta_mapping": meta_mapping,
        "num_metas": len(meta_mapping),
    }


def map_id_to_ascending_integer(
    id_column: str,
    data: pd.DataFrame,
    start_number: int = 0,
    unique_ids: List[int] | None = None,
) -> Tuple[Dict[int, int], pd.DataFrame]:
    """
    Maps primary key into ascending integer.
    Kakao manages primary keys for reviewer (reviewer_id) and diner (diner_idx) with big integer numbers.
        - Example of diner_idx values are 1783219, 1382138.
    This function maps those big integer numbers into unique ascending integer sequence.

    Why this process is necessary?
     - Those values are not necessary to maintain their format and for efficient indexing in torch, unique ascending
     integer representation is required. In pytorch, torch.nn.Embedding creates embedding matrix with
     (num_samples x dimension) and kth row in this torch means representation of kth id.
     - When searching values in data for kth id, reduced representation is more efficient in terms of time complexity.

    Args:
        id_column (str): Column name to convert to ascending integer.
        data (pd.DataFrame): Diner or review data.
        start_number (int): If set value larger than 0, say k, mapping will start from k rather than 0.
            This option is useful when there are already unique mapping in data and wants to start right next.
        unique_ids (List[int]): Ids needed to be converted to ascending integer.

    Returns (Tuple[Dict[int, int], pd.DataFrame]):
        Mapping dictionary and converted dataframe.
    """
    if unique_ids is None:
        unique_ids = sorted(data[id_column].dropna().unique().tolist())

    mapping_info = {id_: i + start_number for i, id_ in enumerate(unique_ids)}
    data[id_column] = data[id_column].map(mapping_info)

    return mapping_info, data


def prepare_torch_dataloader(
    X_train: Tensor,
    y_train: Tensor,
    X_val: Tensor,
    y_val: Tensor,
    batch_size: int = 128,
    num_workers: int = 4,
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
        num_workers (int): number of workers when multiprocessing.

    Returns (Tuple[DataLoader, DataLoader]):
        Train / validation dataloader.
    """

    train_dataset = TorchData(X_train, y_train)
    val_dataset = TorchData(X_val, y_val)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )
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


def prepare_networkx_undirected_graph(
    X_train: Tensor,
    y_train: Tensor,
    X_val: Tensor,
    y_val: Tensor,
    diner: pd.DataFrame,
    user_mapping: Dict[int, int],
    diner_mapping: Dict[int, int],
    meta_mapping: Dict[int, int] = None,
    weighted: bool = False,
    use_metadata: bool = False,
) -> Tuple[nx.Graph, nx.Graph]:
    """
    Make train / validation dataset in nx.Graph object type.
    Metadata could be integrated into nx.Graph depending on the argument.

    There are two types of graphs when integrating metadata nodes to model.
        - Directed graph between user and diner.
            - When weighted equals true, rating that user gave to diner is set as weight.
        - Directed graph between diner and metadata.
            - Metadata could be based on a various data source. For example, it could be
            defined as {h3_cell_id}_{diner_middle_category}.
            - Note that when traversing from diner node to metadata node, there is the only one path.
            However, when traversing reversely, there are lots of paths because multiple diners belong to one metadata.

    Args:
        X_train (Tensor): input features used when training model.
        y_train (Tensor): target features, which is usually used for edged weight in training data.
        X_val (Tensor): input features used when validating model.
        y_val (Tensor): target features, which is usually used for edged weight in validation data.
        diner (pd.DataFrame): diner dataset consisting of diner_id and metadata_id.
        user_mapping (Dict[int, int]): dictionary mapping original user_id to descending integer.
        diner_mapping (Dict[int, int]): dictionary mapping original diner_id to descending integer.
        meta_mapping (Dict[int, int]): dictionary mapping original meta_id to descending integer.
        weighted (bool): whether setting edge weight or not.
        use_metadata (bool): whether to use metadata or not.

    Returns (Tuple[nx.Graph, nx.Graph]):
        Train / validation dataset in nx.Graph object type.
    """
    train_graph = nx.Graph()
    val_graph = nx.Graph()

    # Prepare all edges at once
    edges = []

    # Add user-diner edges
    edges.extend(
        (
            diner_id.item(),
            reviewer_id.item(),
            {"weight": rating.item()} if weighted else {},
        )
        for (diner_id, reviewer_id), rating in zip(X_train, y_train)
    )
    edges.extend(
        (
            diner_id.item(),
            reviewer_id.item(),
            {"weight": rating.item()} if weighted else {},
        )
        for (diner_id, reviewer_id), rating in zip(X_val, y_val)
    )

    # Add metadata edges if needed
    if use_metadata:
        edges.extend(
            edge
            for _, row in diner.iterrows()
            for edge in [
                (row["diner_idx"], row["metadata_id"], {}),
                *(
                    (row["diner_idx"], meta, {})
                    for meta in row["metadata_id_neighbors"]
                ),
            ]
        )

    # Add all edges to both graphs at once
    train_graph.add_edges_from(edges)
    val_graph.add_edges_from(edges)

    # Add node attributes if needed
    if use_metadata:
        node_metadata = {
            **{uid: {"meta": "user"} for uid in user_mapping.values()},
            **{did: {"meta": "diner"} for did in diner_mapping.values()},
            **{mid: {"meta": "category"} for mid in meta_mapping.values()},
        }

        nx.set_node_attributes(train_graph, node_metadata)
        nx.set_node_attributes(val_graph, node_metadata)

    return train_graph, val_graph
