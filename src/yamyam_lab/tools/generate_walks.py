import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import torch
from constant.embedding.metapath2vec import TransitionKeyMetaPath
from constant.embedding.node2vec import TransitionKey
from numpy.typing import NDArray
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

"""
source: https://github.com/eliorc/node2vec/blob/master/node2vec/parallel.py
"""


def generate_walks(
    node_ids: Union[List[int], NDArray],
    d_graph: Dict[str, Any],
    walk_length: int,
    num_walks: int,
) -> Tensor:
    """
    Generates the random walks which will be used as the skip-gram input.

    Args:
        node_ids (Union[List[int], NDArray]): Node id as starting point.
        d_graph (Dict[str, Any]): Precomputed transition probabilities based
            on parameter `p`, `q`, and edge weights
        walk_length (int): Length of the random walks.
        num_walks (int): Number of biased random walks for each of node id

    Returns (Tensor):
        Concatenated random walks in Tensor.
    """

    walks = list()

    for n_walk in range(num_walks):
        # Start a random walk from input node
        for source in node_ids:
            # Start walk
            walk = [source]

            # Perform walk
            while len(walk) < walk_length:
                # last visited node's neighbors
                walk_options = d_graph[walk[-1]].get(TransitionKey.NEIGHBORS, None)

                # skip dead end nodes which have no neighbors
                if not walk_options:
                    break

                if len(walk) == 1:
                    # for the first step
                    probabilities = d_graph[walk[-1]][TransitionKey.FIRST_PROB]
                    walk_to = random.choices(walk_options, weights=probabilities)[0]
                else:
                    # if not first step, consider previous visited node of last visited node
                    # walk[-1]: last visited node, walk[-2]: previous visited node of last visited node
                    probabilities = d_graph[walk[-1]][TransitionKey.NEXT_PROB][walk[-2]]
                    walk_to = random.choices(walk_options, weights=probabilities)[0]

                walk.append(walk_to)

            walks.append(walk)

    return torch.tensor(walks)


def precompute_probabilities(
    graph: nx.Graph,
    p: float = 1.0,
    q: float = 1.0,
) -> Dict[str, Any]:
    """
    Precomputes transition probabilities for each node.

    Args:
        graph (nx.Graph): Networkx graph containing edge relationship.
        p (float): BFS related hyperparameter.
        q (float): DFS related hyperparameter.

    Returns (Dict[str, Any]):
        Dictionary of keys with node id and values with transition probabilities.
    """
    d_graph = defaultdict(dict)
    # initialize transition matrix
    for node in graph.nodes():
        d_graph[node][TransitionKey.FIRST_PROB] = []
        d_graph[node][TransitionKey.NEIGHBORS] = []
        d_graph[node][TransitionKey.NEXT_PROB] = {}
        for neighbor in graph.neighbors(node):
            d_graph[node][TransitionKey.NEXT_PROB][neighbor] = []

    for source in tqdm(graph.nodes(), desc="Computing transition probabilities"):
        for current_node in graph.neighbors(source):
            unnorm_weights = list()
            d_neighbors = list()

            # Calculate unnormalized weights
            for destination in graph.neighbors(current_node):
                weight = graph[current_node][destination].get("weight", 1)

                if destination == source:  # Backwards probability
                    weight = weight * 1 / p if p > 0 else 0
                elif (
                    destination in graph[source]
                ):  # If the neighbor is connected to the source
                    weight = weight
                else:
                    weight = weight * 1 / q if q > 0 else 0

                # Assign the unnormalized sampling strategy weight, normalize during random walk
                unnorm_weights.append(weight)
                d_neighbors.append(destination)

            # Normalize
            unnorm_weights = np.array(unnorm_weights)
            d_graph[current_node][TransitionKey.NEXT_PROB][source] = (
                unnorm_weights / unnorm_weights.sum()
            )

        # Calculate first_travel weights for source
        first_travel_weights = []

        for destination in graph.neighbors(source):
            first_travel_weights.append(graph[source][destination].get("weight", 1))

        first_travel_weights = np.array(first_travel_weights)
        d_graph[source][TransitionKey.FIRST_PROB] = (
            first_travel_weights / first_travel_weights.sum()
        )

        # Save neighbors preserving order
        d_graph[source][TransitionKey.NEIGHBORS] = list(graph.neighbors(source))
    return d_graph


def precompute_probabilities_metapath(
    graph: nx.Graph,
    meta_field: str,
) -> Dict[int, Dict[str, Dict[str, List[int]]]]:
    """
    Precomputes probability when using metapath.
    Note that there are not any parameters p and q which controls bias in random walk.
    In other words, for metapath2vec, random walk categorized by meta is performed.
    To do this, this function precompute uniform probabilities for each node.

    Args:
        graph (nx.Graph): Networkx graph object passed from preprocessing step.
        meta_field (str): Name of meta field in nx.Graph.

    Returns (Dict[int, Dict[str, Dict[str, List[int | float]]]]):
        Key is node id, corresponding is precomputed probabilities.
        For example,
        {
            0: {
                "user": {
                    "neighbors": [1, 2, 3],
                    "prob": [1/3, 1/3, 1/3],
                }
                "diner": {
                    "neighbors": [4],
                    "prob": [1],
                }
                "category": {
                    "neighbors": [],
                    "prob": [],
                }
            }
        }
        Above example shows precomputed probabilities for node_id = 0.
        For node_id = 0,
        neighbor nodes with `user` meta is [1, 2, 3] and
        neighbor nodes with `diner` meta is [4] and
        neighbor nodes with `category` meta is [].
        Note that uniform probabilities are set.
    """
    nodes_without_meta = [
        node for node in graph.nodes() if meta_field not in graph.nodes[node]
    ]
    assert len(nodes_without_meta) == 0

    node_meta = set([graph.nodes[node][meta_field] for node in graph.nodes()])

    d_graph = defaultdict(dict)
    for node in graph.nodes():
        for meta in node_meta:
            d_graph[node][meta] = {
                TransitionKeyMetaPath.NEIGHBORS: [],
                TransitionKeyMetaPath.PROB: [],
            }

    for node in tqdm(graph.nodes(), desc="Computing transition probabilities"):
        for neighbor in graph.neighbors(node):
            neighbor_meta = graph.nodes[neighbor].get(meta_field)
            d_graph[node][neighbor_meta][TransitionKeyMetaPath.NEIGHBORS].append(
                neighbor
            )
        for meta in node_meta:
            if len(d_graph[node][meta][TransitionKeyMetaPath.NEIGHBORS]) != 0:
                neighbors_meta = d_graph[node][meta][TransitionKeyMetaPath.NEIGHBORS]
                # uniform distribution
                d_graph[node][meta][TransitionKeyMetaPath.PROB] = [
                    1 / len(neighbors_meta) for _ in range(len(neighbors_meta))
                ]
    return d_graph


def generate_walks_metapath(
    node_ids: Union[List[int], NDArray],
    graph: nx.Graph,
    d_graph: Dict[int, Dict[str, Dict[str, List[int]]]],
    meta_path: List[List[str]],
    meta_field: str,
    walks_per_node: int,
    padding_value: int,
) -> Tuple[Tensor, List[Tuple[Tuple[str, str], int]]]:
    """
    Generate walks given meta_path.
    For each of meta_path, generate random walk based on precomputed probabilities.
    If there are not any nodes that satisfy next meta information, do not include this sequence.

    Args:
        node_ids (Union[List[int], NDArray]): List of node_ids.
        graph (nx.Graph): Given networkx graph
        d_graph (Dict[int, Dict[str, Dict[str, List[int]]]]): Precomputed probabilities.
        meta_path (List[List[str]]): List of meta path. For example,
            [ ["user","diner","user","diner"], ["user","diner","category","diner","user"] ] shows
            two meta_paths consisting of different meta values.
        meta_field (str): Name of meta field.
        walks_per_node (int): Number of sequences per node.
        padding_value (int): Predefined padding value.

    Returns (Tuple[Tensor, List[Tuple, int]]):
        Concatenated tensor and meta paths count. Latter is used to unpad positive random walks
        when calculating metapath loss.
    """
    walks = []
    meta_path_count = []
    for path in meta_path:
        cnt = 0
        for node in node_ids:
            start_node_meta = graph.nodes[node][meta_field]
            if path[0] != start_node_meta:
                continue
            for _ in range(walks_per_node):
                walk = [node]
                current_node = node
                for meta in path[1:]:
                    neighbors = d_graph[current_node][meta][
                        TransitionKeyMetaPath.NEIGHBORS
                    ]
                    prob = d_graph[current_node][meta][TransitionKeyMetaPath.PROB]
                    if len(neighbors) == 0:
                        break
                    next_node = random.choices(neighbors, weights=prob)[0]
                    walk.append(next_node)

                    current_node = next_node
                if len(walk) == len(path):
                    walks.append(torch.tensor(walk))
                    cnt += 1
        meta_path_count.append((tuple(path), cnt))
    walks_padded = pad_sequence(walks, batch_first=True, padding_value=padding_value)
    return walks_padded, meta_path_count
