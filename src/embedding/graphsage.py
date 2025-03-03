from typing import List, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx.classes import neighbors
from torch import Tensor

from embedding.base_embedding import BaseEmbedding


class SageLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(
            in_features=input_size*2,
            out_features=output_size,
        )

        # self._init_params()

    def _init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feat: torch.Tensor, agg_feat: torch.Tensor):
        concat_feat = torch.concat([self_feat, agg_feat], dim=0).unsqueeze(
            -1
        )  # (1, input_size*2)
        out = F.relu(self.linear(concat_feat))  # (output_size, 1)
        return out


class Model(BaseEmbedding):
    def __init__(
        self,
        user_ids: Tensor,
        diner_ids: Tensor,
        top_k_values: List[int],
        graph: nx.Graph,
        embedding_dim: int,
        num_nodes: int,
        num_layers: int,
        user_raw_features: torch.Tensor,
        diner_raw_features: torch.Tensor,
        agg_func: str = "MEAN",
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        **kwargs,
    ):
        super().__init__(
            user_ids=user_ids,
            diner_ids=diner_ids,
            top_k_values=top_k_values,
            graph=graph,
            embedding_dim=embedding_dim,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            num_nodes=num_nodes,
        )
        self.num_layers = num_layers
        self.user_raw_features = user_raw_features
        self.diner_raw_features = diner_raw_features
        self.agg_func = agg_func

        _, user_feature_input_size = user_raw_features.shape
        _, diner_feature_input_size = diner_raw_features.shape

        self.user_feature_layer = nn.Linear(embedding_dim, user_feature_input_size)
        self.diner_feature_layer = nn.Linear(embedding_dim, diner_feature_input_size)

        for index in range(1, num_layers + 1):
            layer_size = embedding_dim if index != 1 else embedding_dim
            setattr(self, "sage_layer_" + str(index), SageLayer(layer_size, embedding_dim))

    def forward(self, batch_nodes: Tensor):
        B_ks = self._sample_from_batch(batch_nodes=batch_nodes)
        pre_emb = self.raw_features
        for i in range(self.num_layers):
            B_k = B_ks[i]
            cur_emb = torch.empty((len(B_k), self.output_size))
            sage_layer = getattr(f"sage_layer_{i+1}")
            for i,node in enumerate(B_k):
                neighbors = self._sample_neighbors(node, num_samples=4)
                pre_emb_neighbors = pre_emb[neighbors] # h^{k-1}_{u'}
                cur_emb_neighbors = self.aggregate(pre_emb_neighbors)
                pre_emb_node = pre_emb[node] # h^{k-1}_{u}
                cur_emb_node = sage_layer(pre_emb_node, cur_emb_neighbors)
                # normalize
                cur_emb[i] = cur_emb_node
            pre_emb = cur_emb
        return cur_emb

    def pos_sample(self, batch: Tensor) -> Tensor:
        pass

    def neg_sample(self, batch: Tensor) -> Tensor:
        pass

    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        pass

    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        pass

    def _sample_neighbors(self, node: int, num_samples: int):
        neighbors = list(self.graph.neighbors(node))
        if len(neighbors) <= num_samples:
            return neighbors
        else:
            return np.random.choice(neighbors, size=num_samples, replace=False)

    def _sample_from_batch(self, batch_nodes: Tensor):
        batches = [batch_nodes]
        for _ in range(self.num_layers):
            batch_nodes_cp = batch_nodes.detach().cpu().numpy()
            neighbors = []
            for node in batch_nodes_cp:
                neighbors.append(self._sample_neighbors(node, 4))
            batch_nodes_cp = np.concatenate(
                (batch_nodes_cp, np.concatenate((neighbors)))
            )
            batches.append(torch.tensor(batch_nodes_cp))
        return batches[::-1]

    def aggregate(self, emb: Tensor):
        return emb


if __name__ == "__main__":
    sage_layer = SageLayer(input_size=10, output_size=16)
    self_feat = torch.randn(10)
    agg_feat = torch.randn(10)
    sage_layer(self_feat, agg_feat)
