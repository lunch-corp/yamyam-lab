from collections import defaultdict
from typing import List, Tuple, Union

import networkx as nx
import numpy as np
import torch
from torch import Tensor

from embedding.base_embedding import BaseEmbedding
from tools.generate_walks import (
    generate_walks_metapath,
    precompute_probabilities_metapath,
)
from tools.tensor import unpad_by_mask


class Model(BaseEmbedding):
    def __init__(
        self,
        user_ids: Tensor,
        diner_ids: Tensor,
        top_k_values: List[int],
        graph: nx.Graph,
        embedding_dim: int,
        num_nodes: int,
        meta_path: List[List[str]],
        meta_field: str = "meta",
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        inference: bool = False,
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
        self.meta_path = meta_path
        self.meta_field = meta_field

        given_meta = []
        for path in meta_path:
            given_meta.extend(path)
        given_meta = set(given_meta)
        node_meta = set([graph.nodes[node][meta_field] for node in graph.nodes()])
        for meta in given_meta:
            assert meta in node_meta

        self.meta2node_id = defaultdict(list)
        for node in graph.nodes():
            node_meta = graph.nodes[node][meta_field]
            self.meta2node_id[node_meta].append(node)

        if inference is False:
            self.d_graph = precompute_probabilities_metapath(
                graph=graph,
                meta_field=meta_field,
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
        rw, meta_path_count = generate_walks_metapath(
            node_ids=batch.detach().cpu().numpy(),
            graph=self.graph,
            d_graph=self.d_graph,
            meta_path=self.meta_path,
            meta_field=self.meta_field,
            walks_per_node=1,
        )
        self.meta_path_count = meta_path_count
        return rw

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        """
        Sample negative with uniform sampling.
        In word2vec objective function, to reduce computation burden, negative sampling
        is performed and approximate denominator of probability.
        In metapath2vec, heterogeneous negative sampling is performed reflecting meta value
        in positive samples.

        Args:
            batch (Tensor): A batch of node ids.

        Returns (Tensor):
            Negative samples for each of node ids.
        """
        neg_rw = []
        for start_node in batch:
            start_node_meta = self.graph.nodes[start_node.item()][self.meta_field]
            heterogeneous_neg_rw = np.random.choice(
                a=self.meta2node_id[start_node_meta],
                size=self.num_negative_samples,
                replace=False,
            )
            neg_rw.append(torch.tensor(heterogeneous_neg_rw))
        return torch.stack(neg_rw)

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
        pos_rw = self.pos_sample(batch)
        pos_rw_start_node = pos_rw[:, 0]
        neg_rw = self.neg_sample(pos_rw_start_node)
        return pos_rw, neg_rw

    @torch.jit.export
    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        """
        Computes word2vec skip-gram based loss.
        In metapath2vec, pos_rw is passed with padded values. Therefore, this should be
        removed when calculating loss. This function do this using `self.meta_path_count`.

        Args:
             pos_rw (Tensor): Node ids of positive samples
             neg_rw (Tensor): Node ids of negative samples

        Returns (Tensor):
            Calculated loss.
        """
        start_idx = 0
        loss = torch.tensor(0.0, requires_grad=True)
        for meta_path, count in self.meta_path_count:
            # Positive loss.
            pos_rw_padded = pos_rw[start_idx:count, :]
            pos_rw_unpadded = unpad_by_mask(
                padded_tensor=pos_rw_padded,
                padding_value=-1,
            )
            start, rest = pos_rw_unpadded[:, 0], pos_rw_unpadded[:, 1:].contiguous()

            h_start = self.embedding(start).view(
                pos_rw_unpadded.size(0), 1, self.embedding_dim
            )
            h_rest = self.embedding(rest.view(-1)).view(
                pos_rw_unpadded.size(0), -1, self.embedding_dim
            )

            out = (h_start * h_rest).sum(dim=-1).view(-1)
            pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()
            loss = loss + pos_loss

            # Negative loss.
            neg_rw_sliced = neg_rw[start_idx:count, :]
            start, rest = neg_rw_sliced[:, 0], neg_rw_sliced[:, 1:].contiguous()

            h_start = self.embedding(start).view(
                neg_rw_sliced.size(0), 1, self.embedding_dim
            )
            h_rest = self.embedding(rest.view(-1)).view(
                neg_rw_sliced.size(0), -1, self.embedding_dim
            )

            out = (h_start * h_rest).sum(dim=-1).view(-1)
            neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()
            loss = loss + neg_loss

            start_idx += count
        return loss
