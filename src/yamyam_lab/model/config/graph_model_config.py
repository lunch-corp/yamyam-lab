from dataclasses import dataclass
from typing import Any, List, Optional

import torch


@dataclass
class GraphModelConfig:
    """
    Configuration for graph model initialization.

    Args:
        # Required base parameters
        user_ids (Tensor): User ids in data.
        diner_ids (Tensor): Diner ids in data.
        graph (nx.Graph): Networkx graph object generated from train data.
        embedding_dim (int): Dimension of user / diner embedding vector.
        walk_length (int): Length of each random walk.
        walks_per_node (int): Number of generated walks for each node.
        num_nodes (int): Total number of nodes.
        num_negative_samples (int): Number of negative samples for each node.
        top_k_values (List[int]): Top k values used when calculating metric for prediction and candidate generation.
        model_name (str): Model name.
        device (str): Device on which train is run. (cpu or cuda)
        recommend_batch_size (int): Batch size when calculating validation metric.
        num_workers (int): Number of workers used in DataLoader.
        inference (bool): Indicator whether inference mode or not.

        # node2vec parameters
        p (float): Likelihood of immediately revisiting a node in the walk.
        q (float): Control parameter to interpolate between breadth-first strategy and depth-first strategy.

        # metapath2vec parameters
        meta_path (List[List[str]]): List of meta path which controls types of walk sequence.
        meta_field (str): Name of meta field in graph object.

        # graphsage parameters
        num_sage_layers (int): Number of sage layers.
        user_raw_features (torch.Tensor): User raw features whose dimension should be matched with user_ids.
        diner_raw_features (torch.Tensor): Diner raw features whose dimension should be matched with diner_ids.
        aggregator_funcs (str): Aggregation function when combining neighbor embeddings from previous step.
        num_neighbor_samples (int): Number of neighbors.

        # lightgcn parameters
        num_layers (int): Number of layers in lightgcn.
        drop_ratio (float): Ratio used in dropout module.
    """

    # Required base parameters
    user_ids: torch.Tensor
    diner_ids: torch.Tensor
    graph: Any
    embedding_dim: int
    walk_length: int
    walks_per_node: int
    num_nodes: int
    num_negative_samples: int
    top_k_values: List[int]
    model_name: str
    device: str
    recommend_batch_size: int
    num_workers: int
    inference: bool

    # Optional parameters (model-specific)
    # for node2vec
    q: Optional[float] = None
    p: Optional[float] = None

    # for metapath2vec
    meta_path: Optional[List[List[str]]] = None
    meta_field: Optional[str] = "meta"  # constant value

    # for graphsage
    num_sage_layers: Optional[int] = None
    user_raw_features: Optional[torch.Tensor] = None
    diner_raw_features: Optional[torch.Tensor] = None
    aggregator_funcs: Optional[List[str]] = None
    num_neighbor_samples: Optional[List[int]] = None

    # for lightgcn
    num_layers: Optional[int] = None
    drop_ratio: Optional[float] = None
