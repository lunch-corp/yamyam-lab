import os
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from preprocess.feature_store import extract_scores_array, extract_statistics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tools.google_drive import ensure_data_files
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

    review = pd.read_csv(data_paths["review"]["file_path"])
    reviewer = pd.read_csv(data_paths["reviewer"]["file_path"])

    review = pd.merge(review, reviewer, on="reviewer_id", how="left")

    if test:
        review = review.iloc[:5000, :]


    diner = pd.read_csv(data_paths["diner"]["file_path"])
    diner_with_raw_category = pd.read_csv(data_paths["diner_category_raw"]["file_path"])
    return review, diner, diner_with_raw_category


def preprocess_common(
    review: pd.DataFrame,
    diner: pd.DataFrame,
    diner_with_raw_category: pd.DataFrame,
    min_reviews: int,
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
    # step 1: filter reviewers writing reviews greater than or equal to `min_reviews`
    reviewer_counts = review["reviewer_id"].value_counts()
    valid_reviewers = reviewer_counts[reviewer_counts >= min_reviews].index
    review = review[review["reviewer_id"].isin(valid_reviewers)]

    # step 2: use diner which has at least one review
    diner_idx_both_exist = set(review["diner_idx"].unique()) & set(
        diner["diner_idx"].unique()
    )
    review = review[review["diner_idx"].isin(diner_idx_both_exist)]
    diner = diner[diner["diner_idx"].isin(diner_idx_both_exist)]

    # step 3: replace diner_category with raw, unpreprocessed diner_category
    # this is temporary preprocessing because preprocessed categories will be given
    category_columns = [
        "diner_category_large",
        "diner_category_middle",
        "diner_category_small",
        "diner_category_detail",
    ]
    columns_exclude_category_columns = [
        col for col in diner.columns if col not in category_columns
    ]
    diner = pd.merge(
        left=diner[columns_exclude_category_columns],
        right=diner_with_raw_category,
        how="left",
        on="diner_idx",
    )

    # step 4: temporary na filling
    diner["diner_category_large"] = diner["diner_category_large"].fillna("NA")

    return review, diner


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


def map_id_to_ascending_integer(
    review: pd.DataFrame,
    diner: pd.DataFrame,
    is_graph_model: bool = False,
    category_column_for_meta: str = "diner_category_large",
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
        category_column_for_meta (str): Category column name which will be used to generate meta for each node.

    Returns (Dict[str, Any]):
        Mapped result.
    """
    # store unique number of diner and reviewer
    diner_idxs = sorted(list(review["diner_idx"].unique()))
    reviewer_ids = sorted(list(review["reviewer_id"].unique()))

    num_diners = len(diner_idxs)
    num_users = len(reviewer_ids)

    # mapping diner_idx and reviewer_id
    diner_mapping = {diner_idx: i for i, diner_idx in enumerate(diner_idxs)}

    if is_graph_model:
        # each node index in graph based model should be unique
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

    # metadata preprocessing
    if is_graph_model:
        diner = preprocess_diner_data_for_candidate_generation(
            diner=diner,
            category_column_for_meta=category_column_for_meta,
        )
        meta_ids = list(diner["metadata_id"].unique())
        for meta in diner["metadata_id_neighbors"]:
            meta_ids.extend(meta)
        meta_ids = sorted(list(set(meta_ids)))

        meta_mapping = {
            meta_id: (i + num_diners + num_users) for i, meta_id in enumerate(meta_ids)
        }
        diner["metadata_id"] = diner["metadata_id"].map(meta_mapping)
        diner["metadata_id_neighbors"] = diner["metadata_id_neighbors"].map(
            lambda x: [meta_mapping[meta] for meta in x]
        )
    else:
        meta_mapping = None

    return {
        "review": review,
        "diner": diner,
        "num_diners": num_diners,
        "num_users": num_users,
        "num_metas": len(meta_mapping) if meta_mapping else 0,
        "diner_mapping": diner_mapping,
        "user_mapping": reviewer_mapping,
        "meta_mapping": meta_mapping,
    }


def preprocess_diner_data_for_candidate_generation(
    diner: pd.DataFrame,
    category_column_for_meta: str = None,
) -> pd.DataFrame:
    """
    Additional preprocessing when metadata is integrated to graph based model.

    Args:
        diner (pd.DataFrame): Diner dataset
        category_column_for_meta (str): Category column name which will be used to generate meta for each node.

    Returns (pd.DataFrame):
        Diner dataset with metadata added.
    """
    # get diner's h3_index
    diner["h3_index"] = diner.apply(
        lambda row: get_h3_index(row["diner_lat"], row["diner_lon"], RESOLUTION), axis=1
    )
    # get h3_index neighboring with diner's h3_index and concat with meta field
    diner["metadata_id_neighbors"] = diner.apply(
        lambda row: [
            row[category_column_for_meta] + "_" + h3_index
            for h3_index in get_hexagon_neighbors(row["h3_index"], k=1)
        ],
        axis=1,
    )
    # get current h3_index and concat with meta field
    diner["metadata_id"] = diner.apply(
        lambda row: row[category_column_for_meta] + "_" + row["h3_index"], axis=1
    )
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
    is_graph_model: bool = False,
    category_column_for_meta: str = "diner_category_large",
    test: bool = False,
    is_rank: bool = False,
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
        is_graph_model (bool): indicator whether using graph based model or not.
            When set true, all the mapped index should be unique in ascending order.
        category_column_for_meta (str): Category column name which will be used to generate meta for each node.
        test (bool): indicator whether under pytest. when set true, use part of total dataset.

    Returns (Dict[str, Any]):
        Dataset, statistics, and mapping information which could be used when training model.
    """
    review, diner, diner_with_raw_category = load_dataset(test=test)
    assert category_column_for_meta in diner.columns
    review, diner = preprocess_common(
        review=review,
        diner=diner,
        diner_with_raw_category=diner_with_raw_category,
        min_reviews=min_reviews,
    )
    mapped_res = map_id_to_ascending_integer(
        review=review,
        diner=diner,
        is_graph_model=is_graph_model,
        category_column_for_meta=category_column_for_meta,
    )

    review = mapped_res.get("review")
    diner = mapped_res.get("diner")
    mapped_res = {k: v for k, v in mapped_res.items() if k not in ["review", "diner"]}

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

    if is_rank:
        # label Encoder
        le = LabelEncoder()
        train["badge_grade"] = le.fit_transform(train["badge_grade"])
        val["badge_grade"] = le.transform(val["badge_grade"])

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

        return {
            "X_train": train.drop(columns=["target"]),
            "y_train": train["target"],
            "X_val": val.drop(columns=["target"]),
            "y_val": val["target"],
            **mapped_res,
        }

    return {
        "X_train": torch.tensor(train[X_columns].values),
        "y_train": torch.tensor(train[y_columns].values, dtype=torch.float32),
        "X_val": torch.tensor(val[X_columns].values),
        "y_val": torch.tensor(val[y_columns].values, dtype=torch.float32),
        "diner": diner,
        **mapped_res,
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


def prepare_networkx_undirected_graph(
    X_train: Tensor,
    y_train: Tensor,
    X_val: Tensor,
    y_val: Tensor,
    diner: pd.DataFrame,
    user_mapping: Dict[int, int],
    diner_mapping: Dict[int, int],
    meta_mapping: Dict[int, int],
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

    # add edge between user and diner
    for (diner_id, reviewer_id), rating in zip(X_train, y_train):
        if weighted is True:
            train_graph.add_edge(
                diner_id.item(), reviewer_id.item(), weight=rating.item()
            )
        else:
            train_graph.add_edge(diner_id.item(), reviewer_id.item())

    for (diner_id, reviewer_id), rating in zip(X_val, y_val):
        if weighted is True:
            val_graph.add_edge(
                diner_id.item(), reviewer_id.item(), weight=rating.item()
            )
        else:
            val_graph.add_edge(diner_id.item(), reviewer_id.item())

    # add edge between diner and metadata
    if use_metadata is True:
        for i, row in diner.iterrows():
            diner_idx = row["diner_idx"]
            metadata_id = row["metadata_id"]
            train_graph.add_edge(diner_idx, metadata_id)
            val_graph.add_edge(diner_idx, metadata_id)
            for meta in row["metadata_id_neighbors"]:
                train_graph.add_edge(diner_idx, meta)
                val_graph.add_edge(diner_idx, meta)

    # add node and node attribute (user / diner / meta) to networkx graph
    nodes_metadata = {
        **{user_id: {"meta": "user"} for _, user_id in user_mapping.items()},
        **{diner_id: {"meta": "diner"} for _, diner_id in diner_mapping.items()},
        **{meta_id: {"meta": "category"} for _, meta_id in meta_mapping.items()},
    }
    nx.set_node_attributes(train_graph, nodes_metadata)
    nx.set_node_attributes(val_graph, nodes_metadata)

    return train_graph, val_graph
