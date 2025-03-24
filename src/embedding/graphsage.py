from typing import List, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from embedding.base_embedding import BaseEmbedding
from tools.generate_walks import precompute_probabilities
from tools.sampling import uniform_sampling_without_replacement_from_pool


class SageLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(
            in_features=input_size * 2,
            out_features=output_size,
        )

    def forward(self, self_feat: torch.Tensor, agg_feat: torch.Tensor):
        concat_feat = torch.concat([self_feat, agg_feat], dim=0).unsqueeze(
            0
        )  # (1, input_size*2)
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
        # parameters for graphsage
        num_layers: int,
        user_raw_features: torch.Tensor,
        diner_raw_features: torch.Tensor,
        agg_func: str = "MEAN",
        walk_length: int = 1,
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
        )
        self.num_layers = num_layers
        self.user_raw_features = user_raw_features
        self.diner_raw_features = diner_raw_features
        self.agg_func = agg_func
        self.walk_length = walk_length

        _, user_feature_input_size = user_raw_features.shape
        _, diner_feature_input_size = diner_raw_features.shape

        self.user_feature_layer = nn.Linear(user_feature_input_size, embedding_dim)
        self.diner_feature_layer = nn.Linear(diner_feature_input_size, embedding_dim)

        for index in range(1, num_layers + 1):
            layer_size = embedding_dim if index != 1 else embedding_dim
            setattr(
                self, "sage_layer_" + str(index), SageLayer(layer_size, embedding_dim)
            )

        self.d_graph = precompute_probabilities(
            graph=graph,
            p=1,  # unbiased random walk
            q=1,  # unbiased random walk
        )

    def forward(self, batch_nodes: Tensor) -> Tensor:
        """
        Forward method in graphsage following `minibatch pseudocode` in paper.
        Args:
            batch_nodes (Tensor): Node ids in current batch.

        Returns (Tensor):
            Propagated embedding vector with node features in inductive way.
        """
        B_ks = self._sample_from_batch(batch_nodes=batch_nodes)
        emb = self._get_raw_features()
        for i in range(self.num_layers):
            B_k = B_ks[i]
            sage_layer = getattr(self, f"sage_layer_{i + 1}")
            for node in B_k:
                # sample neighbors from current node
                neighbors = self._sample_neighbors(
                    node.item(), num_samples=self.walks_per_node
                )

                # get neighbor nodes embeddings in previous step
                pre_emb_neighbors = emb[neighbors]  # h^{k-1}_{u'}

                # aggregate neighbor embedding vectors
                agg_emb_neighbors = self.aggregate(
                    pre_emb_neighbors
                )  # AGG_k( h^{k-1}_{u'} )

                # get current node embedding in previous step
                pre_emb_node = emb[node]  # h^{k-1}_{u}

                # pass sage layer and get node embedding in current step
                cur_emb_node = sage_layer(pre_emb_node, agg_emb_neighbors)  # h^{k}_{u}

                # normalize current node embedding
                cur_emb_node = F.normalize(cur_emb_node, p=2, dim=1)

                # store current step embedding
                emb[node] = cur_emb_node
        return emb[batch_nodes]

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
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.forward(start).view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = self.forward(rest.view(-1)).view(
            pos_rw.size(0), -1, self.embedding_dim
        )

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.forward(start).view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = self.forward(rest.view(-1)).view(
            neg_rw.size(0), -1, self.embedding_dim
        )

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss

    def _sample_neighbors(self, node: int, num_samples: int) -> Tensor:
        """
        Uniformly samples neighbor nodes.

        Args:
            node (int): Center node.
            num_samples (int): Number of neighbors to sample.

        Returns (Tensor):
            Tensor containing node_ids of neighbors.
        """
        if not self.graph.has_node(node):
            return torch.tensor([], dtype=torch.long)
        neighbors = list(self.graph.neighbors(node))
        if len(neighbors) <= num_samples:
            return torch.tensor(neighbors)
        else:
            return uniform_sampling_without_replacement_from_pool(
                pool=torch.tensor(neighbors),
                size=num_samples,
            )

    def _sample_from_batch(self, batch_nodes: Tensor) -> List[Tensor]:
        """
        Sampling stage in `minibatch pseudocode` in paper.

        Args:
            batch_nodes (Tensor): List of node_ids in current batch.

        Returns (List[Tensor]):
            List of B^{k}.
        """
        batch_nodes = torch.tensor(
            [
                node_id
                for node_id in torch.unique(batch_nodes).clone()
                if self.graph.has_node(node_id.item())
            ]
        )
        batches = [batch_nodes]
        for _ in range(self.num_layers):
            last_batch_nodes = batches[-1].detach().cpu().numpy()
            neighbors = []
            for node in last_batch_nodes:
                neighbors.append(self._sample_neighbors(node, self.walks_per_node))
            last_batch_nodes_with_neighbors = np.concatenate(
                (last_batch_nodes, np.concatenate(neighbors))
            )
            batches.append(torch.tensor(np.unique(last_batch_nodes_with_neighbors)))
        return batches[::-1]

    def aggregate(self, emb: Tensor) -> Tensor:
        """
        Function for aggregating embeddings of neighbors.
        In paper, GCN, mean, LSTM, pool algorithms are used.
        Those functions will be implemented in the future.

        Args:
            emb (Tensor): Stacked embeddings of neighbors.

        Returns (Tensor):
            Aggregated embeddings using specified agg_func.
        """
        if self.agg_func == "MEAN":
            return emb.mean(dim=0)

    def _get_raw_features(self) -> Tensor:
        """
        Helper function for getting raw features from user feature and diner feature.
        For heterogeneous graph, dimensions of user feature and diner feature could be different.
        Therefore, features are passed to each linear layer to match as a single dimension.

        Returns (Tensor):
            Concatenated features with diner feature first and user feature following.
        """
        user_features = self.user_feature_layer(self.user_raw_features.to(self.device))
        diner_features = self.diner_feature_layer(
            self.diner_raw_features.to(self.device)
        )
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
