from typing import List, Tuple, Union

import torch
from torch import Tensor

from yamyam_lab.loss.custom import basic_contrastive_loss
from yamyam_lab.model.config.graph_model_config import GraphModelConfig
from yamyam_lab.model.graph.base_embedding import BaseEmbedding
from yamyam_lab.tools.generate_walks import precompute_probabilities


class Model(BaseEmbedding):
    def __init__(
        self,
        config: GraphModelConfig = None,
    ):
        """
        This is a customized version of pytorch geometric implementation of node2vec.
        It differs from pg implementation in 2 aspects.
            - class initialization: Does not use any pyg-lib or torch-cluster.
                Make random walks using explicit function.
            - data structure: Uses networkx.Graph.

        Args:
            config (GraphModelConfig): Configuration object containing all parameters.
        """
        # Pass config to parent
        super().__init__(config=config)

        self.walk_length = config.walk_length
        self.q = config.q
        self.p = config.p

        if self.inference is False:
            self.d_graph = precompute_probabilities(
                graph=self.graph,
                p=self.p,
                q=self.q,
            )

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

    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        """
        Computes word2vec skip-gram based loss.

        Args:
             pos_rw (Tensor): Node ids of positive samples
             neg_rw (Tensor): Node ids of negative samples

        Returns (Tensor):
            Calculated loss.
        """
        pos_rw_emb = self._embedding(pos_rw.view(-1).to(self.device)).view(
            pos_rw.size(0), -1, self.embedding_dim
        )
        neg_rw_emb = self._embedding(neg_rw.view(-1).to(self.device)).view(
            neg_rw.size(0), -1, self.embedding_dim
        )

        contrastive_loss = basic_contrastive_loss(
            pos_rw_emb=pos_rw_emb,
            neg_rw_emb=neg_rw_emb,
        )

        return contrastive_loss
