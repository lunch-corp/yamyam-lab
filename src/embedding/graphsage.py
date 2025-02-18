from typing import List, Tuple, Union

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from embedding.base_embedding import BaseEmbedding


class SageLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.FloatTensor(output_size, input_size * 2))

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feat: torch.Tensor, agg_feat: torch.Tensor):
        concat_feat = torch.concat([self_feat, agg_feat], dim=0).unsqueeze(
            -1
        )  # (1, input_size*2)
        out = F.relu(torch.mm(self.weight, concat_feat))  # (output_size, 1)
        return out


class GraphSage(BaseEmbedding):
    def __init__(
        self,
        user_ids: Tensor,
        diner_ids: Tensor,
        top_k_values: List[int],
        graph: nx.Graph,
        embedding_dim: int,
        num_nodes: int,
        num_layers: int,
        input_size: int,
        out_size: int,
        raw_features: torch.Tensor,
        adj_lists,
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
        self.input_size = input_size
        self.output_size = out_size
        self.raw_features = raw_features
        self.adj_lists = adj_lists
        self.agg_func = agg_func

        for index in range(1, num_layers + 1):
            layer_size = out_size if index != 1 else input_size
            setattr(self, "sage_layer_" + str(index), SageLayer(layer_size, out_size))

    def pos_sample(self, batch: Tensor) -> Tensor:
        pass

    def neg_sample(self, batch: Tensor) -> Tensor:
        pass

    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        pass

    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        pass


if __name__ == "__main__":
    sage_layer = SageLayer(input_size=10, output_size=16)
    self_feat = torch.randn(10)
    agg_feat = torch.randn(10)
    sage_layer(self_feat, agg_feat)
