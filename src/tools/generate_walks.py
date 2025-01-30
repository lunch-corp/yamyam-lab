from typing import Dict, Any, List, Union
import random
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
import networkx as nx
from collections import defaultdict

import torch
from torch import Tensor

from constant.embedding.node2vec import TransitionKey

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
                walk_options = d_graph[walk[-1]].get(
                    TransitionKey.NEIGHBORS.value, None
                )

                # skip dead end nodes which have no neighbors
                if not walk_options:
                    break

                if len(walk) == 1:
                    # for the first step
                    probabilities = d_graph[walk[-1]][TransitionKey.FIRST_PROB.value]
                    walk_to = random.choices(walk_options, weights=probabilities)[0]
                else:
                    # if not first step, consider previous visited node of last visited node
                    # walk[-1]: last visited node, walk[-2]: previous visited node of last visited node
                    probabilities = d_graph[walk[-1]][TransitionKey.NEXT_PROB.value][
                        walk[-2]
                    ]
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
        d_graph[node][TransitionKey.FIRST_PROB.value] = []
        d_graph[node][TransitionKey.NEIGHBORS.value] = []
        d_graph[node][TransitionKey.NEXT_PROB.value] = {}
        for neighbor in graph.neighbors(node):
            d_graph[node][TransitionKey.NEXT_PROB.value][neighbor] = []

    for source in tqdm(graph.nodes(), desc="Computing transition probabilities"):
        for current_node in graph.neighbors(source):
            unnorm_weights = list()
            d_neighbors = list()

            # Calculate unnormalized weights
            for destination in graph.neighbors(current_node):
                weight = graph[current_node][destination].get("weight", 1)

                if destination == source:  # Backwards probability
                    weight = weight * 1 / p
                elif (
                    destination in graph[source]
                ):  # If the neighbor is connected to the source
                    weight = weight
                else:
                    weight = weight * 1 / q

                # Assign the unnormalized sampling strategy weight, normalize during random walk
                unnorm_weights.append(weight)
                d_neighbors.append(destination)

            # Normalize
            unnorm_weights = np.array(unnorm_weights)
            d_graph[current_node][TransitionKey.NEXT_PROB.value][source] = (
                unnorm_weights / unnorm_weights.sum()
            )

        # Calculate first_travel weights for source
        first_travel_weights = []

        for destination in graph.neighbors(source):
            first_travel_weights.append(graph[source][destination].get("weight", 1))

        first_travel_weights = np.array(first_travel_weights)
        d_graph[source][TransitionKey.FIRST_PROB.value] = (
            first_travel_weights / first_travel_weights.sum()
        )

        # Save neighbors preserving order
        d_graph[source][TransitionKey.NEIGHBORS.value] = list(graph.neighbors(source))
    return d_graph
