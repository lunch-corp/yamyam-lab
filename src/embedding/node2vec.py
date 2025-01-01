from typing import Union

import networkx as nx
import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

from embedding.base_embedding import BaseEmbedding
from tools.generate_walks import generate_walks, precompute_probabilities


class Model(BaseEmbedding):
    r"""
    This is a customized version of pytorch geometric implementation of node2vec.
    It differs from pg implementation in 2 aspects.
        - class initialization: Does not use any pyg-lib or torch-cluster.
            Make random walks using explicit function.
        - data structure: Uses networkx.Graph.

    Original doc string starts here.
    ----------------------------------
    The Node2Vec model from the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
    length :obj:`walk_length` are sampled in a given graph, and node embeddings
    are learned via negative sampling optimization.

    .. note::

        For an example of using Node2Vec, see `examples/node2vec.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        node2vec.py>`_.

    Args:
        graph (nx.Graph): Graph object.
        embedding_dim (int): The size of each embedding vector.
        walk_length (int): The walk length.
        walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
        num_negative_samples (int, optional): The number of negative samples to
            use for each positive sample. (default: :obj:`1`)
    """

    def __init__(
        self,
        user_ids: Tensor,
        diner_ids: Tensor,
        graph: nx.Graph,
        embedding_dim: int,
        walk_length: int,
        num_nodes: int,
        top_k_values: list[int],
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        inference: bool = False,
    ):
        super().__init__(
            user_ids=user_ids,
            diner_ids=diner_ids,
            top_k_values=top_k_values,
        )
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.EPS = 1e-15
        self.num_nodes = num_nodes

        self.embedding = Embedding(self.num_nodes, embedding_dim)

        if inference is False:
            self.d_graph = precompute_probabilities(
                graph=graph,
                p=p,
                q=q,
            )

    def forward(self, batch: Tensor) -> Tensor:
        """
        Dummy forward pass which actually does not do anything.

        Args:
            batch (Tensor): A batch of node ids.

        Returns (Tensor):
            A batch of node embeddings.
        """
        emb = self.embedding.weight
        return emb if batch is None else emb[batch]

    def loader(self, **kwargs) -> DataLoader:
        """
        Node id generator in pytorch dataloader type.

        Returns (DataLoader):
            DataLoader used when training model.
        """
        return DataLoader(
            torch.tensor([node for node in self.graph.nodes()]),
            collate_fn=self.sample,
            **kwargs
        )

    @torch.jit.export
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
        batch = batch.repeat(self.walks_per_node)
        rw = generate_walks(
            node_ids=batch.detach().cpu().numpy(),
            d_graph=self.d_graph,
            walk_length=self.walk_length,
            num_walks=1,
        )
        return rw

    @torch.jit.export
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
        batch = batch.repeat(self.walks_per_node)

        rw = torch.randint(
            self.num_nodes,
            (batch.size(0), self.num_negative_samples),
            dtype=batch.dtype,
            device=batch.device,
        )
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        return rw

    @torch.jit.export
    def sample(self, batch: Union[list[int], Tensor]) -> tuple[Tensor, Tensor]:
        """
        Wrapper function for positive, negative sampling.
        This function is used as `collate_fn` in pytorch dataloader.

        Args:
            batch (Union[List[int], Tensor]): A batch of node ids.

        Returns (Tuple[Tensor, Tensor]):
            Positive, negative samples.
        """
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    @torch.jit.export
    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        """
        Computes word2vec skip-gram based loss.

        Args:
             pos_rw (Tensor): Node ids of positive samples
             neg_rw (Tensor): Node ids of negative samples

        Returns (Tensor):
            Calculated loss.
        """
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(
            pos_rw.size(0), -1, self.embedding_dim
        )

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(
            neg_rw.size(0), -1, self.embedding_dim
        )

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss
