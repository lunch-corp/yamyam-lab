import torch
from torch_geometric.nn import Node2Vec as Node2VecPG

from embedding.base import BaseEmbedding


class Node2Vec(BaseEmbedding):
    def __init__(self):
        super().__init__()

    def initialize(self, **kwargs):
        model = Node2VecPG(
            edge_index=kwargs["edge_index"],
            embedding_dim=kwargs["embedding_dim"],
            walk_length=kwargs["walk_length"],
            context_size=kwargs["context_size"],
            walks_per_node=kwargs["walks_per_node"],
            num_negative_samples=kwargs["num_negative_samples"],
            p=kwargs["p"],
            q=kwargs["q"],
            sparse=kwargs["sparse"],
        )
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
        return model, optimizer