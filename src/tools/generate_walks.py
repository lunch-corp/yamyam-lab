from typing import Dict, Any, List, Union
import random
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
import networkx as nx
from collections import defaultdict

import torch
from torch import Tensor

from constant.node2vec import TransitionKey
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

    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    # pbar = tqdm(total=num_walks, desc=f"Generating walks")

    for n_walk in range(num_walks):

        # Update progress bar
        # pbar.update(1)

        # Start a random walk from input node
        for source in node_ids:

            # Start walk
            walk = [source]

            # Perform walk
            while len(walk) < walk_length:

                # last visited node's neighbors
                walk_options = d_graph[walk[-1]].get(TransitionKey.NEIGHBORS.value, None)

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
                    probabilities = d_graph[walk[-1]][TransitionKey.NEXT_PROB.value][walk[-2]]
                    walk_to = random.choices(walk_options, weights=probabilities)[0]

                walk.append(walk_to)

            walks.append(walk)

    # pbar.close()

    return torch.tensor(walks)

def precompute_probabilities(
        graph: nx.Graph,
        p: float = 1.0,
        q: float = 1.0,
) -> Dict[str, Any]:
    """
    Precomputes transition probabilities for each node.
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
                elif destination in graph[source]:  # If the neighbor is connected to the source
                    weight = weight
                else:
                    weight = weight * 1 / q

                # Assign the unnormalized sampling strategy weight, normalize during random walk
                unnorm_weights.append(weight)
                d_neighbors.append(destination)

            # Normalize
            unnorm_weights = np.array(unnorm_weights)
            d_graph[current_node][TransitionKey.NEXT_PROB.value][source] = unnorm_weights / unnorm_weights.sum()

        # Calculate first_travel weights for source
        first_travel_weights = []

        for destination in graph.neighbors(source):
            first_travel_weights.append(graph[source][destination].get("weight", 1))

        first_travel_weights = np.array(first_travel_weights)
        d_graph[source][TransitionKey.FIRST_PROB.value] = first_travel_weights / first_travel_weights.sum()

        # Save neighbors preserving order
        d_graph[source][TransitionKey.NEIGHBORS.value] = list(graph.neighbors(source))
    return d_graph

# def generate_walks(self, node_ids: List[int]):
#     flatten = lambda l: [item for sublist in l for item in sublist]
#
#     # Split num_walks for each worker
#     num_walks_lists = np.array_split(range(self.walks_per_node), self.workers)
#
#     walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder)(
#         delayed(generate_walks)(
#             node_ids=node_ids,
#             d_graph=self.d_graph,
#             walk_length=self.walk_length,
#             num_walks=len(num_walks),
#             cpu_num=cpu_num
#         ) for
#         cpu_num, num_walks
#         in enumerate(num_walks_lists, 1))
#
#     walks = flatten(walk_results)
#
#     return walks