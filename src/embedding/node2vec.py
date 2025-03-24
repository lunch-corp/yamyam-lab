from typing import List, Tuple, Union

import networkx as nx
import torch
from torch import Tensor

from embedding.base_embedding import BaseEmbedding
from tools.generate_walks import precompute_probabilities


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
        # parameters for node2vec
        walk_length: int,
        p: float = 1.0,
        q: float = 1.0,
        inference: bool = False,
        **kwargs,
    ):
        """
        This is a customized version of pytorch geometric implementation of node2vec.
        It differs from pg implementation in 2 aspects.
            - class initialization: Does not use any pyg-lib or torch-cluster.
                Make random walks using explicit function.
            - data structure: Uses networkx.Graph.

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
            walk_length (int):
            p (float): Likelihood of immediately revisiting a node in the walk.
            q (float): Control parameter to interpolate between breadth-first strategy and depth-first strategy.
            inference (bool): Indicator whether inference mode or not.
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

        self.walk_length = walk_length
        self.p = p
        self.q = q

        if inference is False:
            self.d_graph = precompute_probabilities(
                graph=graph,
                p=p,
                q=q,
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
        return self._pos_sample(batch)

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
        return self._neg_sample(batch)

    @torch.jit.export
    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
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

        h_start = self._embedding(start).view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = self._embedding(rest.view(-1)).view(
            pos_rw.size(0), -1, self.embedding_dim
        )

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self._embedding(start).view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = self._embedding(rest.view(-1)).view(
            neg_rw.size(0), -1, self.embedding_dim
        )

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss
