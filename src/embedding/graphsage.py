from typing import List, Tuple, Union

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from embedding.base_embedding import BaseEmbedding
from loss.custom import basic_contrastive_loss
from tools.generate_walks import precompute_probabilities
from tools.sampling import uniform_sampling_without_replacement_from_small_size_pool


class SageLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(
            in_features=input_size,
            out_features=output_size,
        )

    def forward(self, self_feat: torch.Tensor, agg_feat: torch.Tensor):
        concat_feat = torch.cat([self_feat, agg_feat], dim=1)
        out = F.relu(self.linear(concat_feat))  # (output_size, 1)
        return out


class Model(BaseEmbedding):
    def __init__(
        self,
        # parameters for base_embedding
        user_ids: Tensor,
        diner_ids: Tensor,
        top_k_values: List[int],
        graph: nx.Graph,
        embedding_dim: int,
        walks_per_node: int,
        num_negative_samples: int,
        num_nodes: int,
        model_name: str,
        device: str,
        recommend_batch_size: int,
        num_workers: int,
        # parameters for graphsage
        num_layers: int,
        user_raw_features: torch.Tensor,
        diner_raw_features: torch.Tensor,
        aggregator_funcs: List[str],
        walk_length: int,
        num_neighbor_samples: int,
        **kwargs,
    ):
        """
        Graphsage model which is inductive type.

        Args:
            user_ids (Tensor): User ids in data.
            diner_ids (Tensor): Diner ids in data.
            top_k_values (List[int]): Top k values used when calculating metric for prediction and candidate generation.
            graph (nx.Graph): Networkx graph object generated from train data.
            embedding_dim (int): Dimension of user / diner embedding vector.
            walks_per_node (int): Number of generated walks for each node.
            num_negative_samples (int): Number of negative samples for each node.
            num_nodes (int): Total number of nodes.
            model_name (str): Model name.
            device (str): Device on which train is run. (cpu or cuda)
            recommend_batch_size (int): Batch size when calculating validation metric.
            num_layers (int): Number of sage layers.
            user_raw_features (torch.Tensor): User raw features whose dimension should be matched with user_ids.
            diner_raw_features (torch.Tensor): Diner raw features whose dimension should be matched with diner_ids.
            agg_func (str): Aggregation function when combining neighbor embeddings from previous step.
            walk_length (int): Length of walk.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            user_ids=user_ids,
            diner_ids=diner_ids,
            top_k_values=top_k_values,
            graph=graph,
            embedding_dim=embedding_dim,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            num_nodes=num_nodes,
            model_name=model_name,
            device=device,
            recommend_batch_size=recommend_batch_size,
            num_workers=num_workers,
        )
        self.num_layers = num_layers
        self.user_raw_features = user_raw_features
        self.diner_raw_features = diner_raw_features
        self.walk_length = walk_length
        self.num_neighbor_samples = num_neighbor_samples

        self.aggregators = []
        for func in aggregator_funcs:
            if func == "mean":
                self.aggregators.append(self.mean_aggregator)
            elif func == "max":
                self.aggregators.append(self.max_aggregator)
            else:
                ValueError(f"Unsupported aggregator: {func}")

        _, user_feature_input_size = user_raw_features.shape
        _, diner_feature_input_size = diner_raw_features.shape

        self.user_feature_layer = nn.Linear(user_feature_input_size, embedding_dim)
        self.diner_feature_layer = nn.Linear(diner_feature_input_size, embedding_dim)
        layer_dims = [self.embedding_dim] * (self.num_layers + 1)

        self.sage_layers = nn.ModuleList()

        for k in range(self.num_layers):
            in_dim = layer_dims[k] * 2
            self.sage_layers.append(
                SageLayer(
                    input_size=in_dim,
                    output_size=layer_dims[k + 1],
                )
            )

        self.d_graph = precompute_probabilities(
            graph=graph,
            p=1,  # unbiased random walk
            q=1,  # unbiased random walk
        )

    def forward(self, nodes: Tensor) -> Tensor:
        """
        Forward method in graphsage following `minibatch pseudocode` in paper.

        Args:
            batch_nodes (Tensor): Node ids in current batch.

        Returns (Tensor):
            Propagated embedding vector with node features in inductive way.
        """
        # Convert batch nodes to a set for O(1) lookups
        batch_nodes_set = set(nodes.detach().cpu().numpy())

        # Initialize sets of nodes and store sampled neighbors needed at each layer (B^k in the algorithm)
        layer_nodes = [set() for _ in range(self.num_layers + 1)]
        layer_neighbor_nodes = [{} for _ in range(self.num_layers + 1)]
        layer_nodes[self.num_layers] = batch_nodes_set.copy()

        # Lines 1-7: neighborhood Sampling - identify required nodes at each layer
        for k in range(self.num_layers, 0, -1):
            for u in layer_nodes[k]:
                # Add u to the set of nodes needed at layer k-1
                layer_nodes[k - 1].add(u)

                # Sample neighbors of u and add them to required nodes at layer k-1
                if self.graph.has_node(u):
                    neighbors = list(self.graph.neighbors(u))
                else:
                    neighbors = []
                sampled_neighbors = self.sample_neighbors(
                    neighbors, self.num_neighbor_samples
                )
                layer_nodes[k - 1].update(sampled_neighbors)

                # store neighbor for reproducibility in forward prop
                layer_neighbor_nodes[k][u] = sampled_neighbors

        # Initialize hidden representations for all required nodes (lines 8-16)
        # h^0_v = x_v for all v in B^0
        hidden_reps = [{}]  # List of dictionaries, one per layer

        # Line 8: initialize with input features for layer 0
        features = self._get_raw_features()
        for v in layer_nodes[0]:
            if self.graph.has_node(v):
                hidden_reps[0][v] = features[v]
            else:
                hidden_reps[0][v] = torch.zeros(self.embedding_dim, device=self.device)

        # Lines 9-15: forward propagation through layers
        for k in range(1, self.num_layers + 1):
            # hidden representations for current layer
            hidden_reps.append({})
            sage_layer = self.sage_layers[k - 1]

            # Lines 10-14: process each node in current layer
            nodes_by_neighbor_count = {}
            for node in layer_nodes[k]:
                n_count = len(layer_neighbor_nodes[k][node])
                if n_count not in nodes_by_neighbor_count:
                    nodes_by_neighbor_count[n_count] = []
                nodes_by_neighbor_count[n_count].append(node)

            for n_count, node_group in nodes_by_neighbor_count.items():
                batch_size = min(128, len(node_group))
                for i in range(0, len(node_group), batch_size):
                    batch = node_group[i : i + batch_size]

                    # Pre-allocate tensors for batched processing
                    self_features = torch.zeros(
                        len(batch), self.embedding_dim, device=self.device
                    )
                    neighbor_features = torch.zeros(
                        len(batch), self.embedding_dim, device=self.device
                    )

                    for j, node in enumerate(batch):
                        self_features[j] = hidden_reps[k - 1][node]

                        neighbors = layer_neighbor_nodes[k][node]
                        # Line 11: aggregate features from neighbors
                        neighbor_feats = [hidden_reps[k - 1][v] for v in neighbors]
                        stacked_neighbors = torch.stack(neighbor_feats)
                        neighbor_features[j] = self.aggregators[k - 1](
                            stacked_neighbors
                        )

                    # Line 12: perform forward pass using sage layer
                    h_new_batch = sage_layer(self_features, neighbor_features)
                    # Line 13: normalize the representation
                    h_new_batch = F.normalize(h_new_batch, p=2, dim=1)

                    # Store results
                    for j, node in enumerate(batch):
                        hidden_reps[k][node] = h_new_batch[j]

        # Line 16: final representations for requested nodes
        results = torch.zeros(len(nodes), self.embedding_dim, device=self.device)
        for i, node_id in enumerate(nodes.tolist()):
            results[i] = hidden_reps[self.num_layers][node_id]

        return results

    def pos_sample(self, batch: Tensor) -> Tensor:
        """
        For each of node id, generate biased random walk using `generate_walks` function.
        Based on transition probabilities information (`d_graph`), perform biased random walks.

        Args:
            batch (Tensor): A batch of node ids which are starting points in each biased random walk.

        Returns (Tensor):
            Generated biased random walks. Number of random walks are based on walks_per_node,
            walk_length, and context size. Note that random walks are concatenated row-wise.
        """
        return self._pos_sample(batch)

    def neg_sample(self, batch: Tensor) -> Tensor:
        """
        Sample negative with uniform sampling.
        In word2vec objective function, to reduce computation burden, negative sampling
        is performed and approximate denominator of probability.

        Args:
            batch (Tensor): A batch of node ids.

        Returns (Tensor):
            Negative samples for each of node ids.
        """
        return self._neg_sample(batch)

    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        # forward propagation
        # pool pos, neg node together and calculates embedding in one batch
        # this line may trigger memory error depending on server spec
        all_nodes = torch.unique(torch.concat([pos_rw.view(-1), neg_rw.view(-1)]))
        embeddings = self.forward(all_nodes)

        # get embeddings for pos, neg node
        pos_rw_indices = (
            (all_nodes.unsqueeze(1) == pos_rw.view(-1).unsqueeze(0))
            .long()
            .argmax(dim=0)
        )  # ( len(pos_rw.view(-1)), )
        neg_rw_indices = (
            (all_nodes.unsqueeze(1) == neg_rw.view(-1).unsqueeze(0))
            .long()
            .argmax(dim=0)
        )  # ( len(neg_rw.view(-1)), )
        pos_rw_emb = embeddings[pos_rw_indices].view(
            pos_rw.size(0), -1, self.embedding_dim
        )  # [i][j]: embedding of pos_rw[i][j]
        neg_rw_emb = embeddings[neg_rw_indices].view(
            neg_rw.size(0), -1, self.embedding_dim
        )  # [i][j]: embedding of neg_rw[i][j]

        contrastive_loss = basic_contrastive_loss(
            pos_rw_emb=pos_rw_emb,
            neg_rw_emb=neg_rw_emb,
        )

        return contrastive_loss

    # Neighborhood sampling function (N_k in the algorithm)
    @staticmethod
    def sample_neighbors(neighbors: List[int], sample_size: int) -> List[int]:
        """Uniformly samples 'sample_size' neighbors from the given list."""
        if len(neighbors) <= sample_size:
            return neighbors
        return uniform_sampling_without_replacement_from_small_size_pool(
            pool=neighbors,
            size=sample_size,
        )

    # Aggregator functions
    @staticmethod
    def mean_aggregator(neighbor_features: torch.Tensor) -> torch.Tensor:
        """Mean aggregator: average neighbor features."""
        return torch.mean(neighbor_features, dim=0)

    @staticmethod
    def max_aggregator(neighbor_features: torch.Tensor) -> torch.Tensor:
        """Max pooling aggregator: element-wise maximum of neighbor features."""
        return torch.max(neighbor_features, dim=0)[0]

    def _get_raw_features(self) -> Tensor:
        """
        Helper function for getting raw features from user feature and diner feature.
        For heterogeneous graph, dimensions of user feature and diner feature could be different.
        Therefore, features are passed to each linear layer to match as a single dimension.

        Returns (Tensor):
            Concatenated features with diner feature first and user feature following.
        """
        user_features = self.user_feature_layer(self.user_raw_features)
        diner_features = self.diner_feature_layer(self.diner_raw_features)
        return torch.concat([diner_features, user_features])  # diner index first

    def propagate_and_store_embedding(self, batch_nodes: Tensor):
        """
        Propagates graphsage algorithm to get embeddings for batch_nodes.
        Because graphsage is inductive algorithm, embeddings should be updated at every epoch.

        Args:
            batch_nodes (Tensor): List of nodes to propagate
        """
        with torch.no_grad():
            self._embedding[batch_nodes] = self.forward(batch_nodes)
