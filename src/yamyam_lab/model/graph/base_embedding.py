from abc import abstractmethod
from typing import List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from constant.metric.metric import Metric
from numpy.typing import NDArray
from tools.generate_walks import generate_walks
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader


class BaseEmbedding(nn.Module):
    def __init__(
        self,
        user_ids: Tensor,
        diner_ids: Tensor,
        top_k_values: List[int],
        graph: nx.Graph = None,
        embedding_dim: int = 32,
        walks_per_node: int = 3,
        num_negative_samples: int = 3,
        num_nodes: int = None,
        model_name: str = None,
        device: str = None,
        recommend_batch_size: int = 2000,
        num_workers: int = 4,
    ):
        """
        Base module for node embedding model (node2vec, metapath2vec, graphsage)

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
        """
        super().__init__()
        self.user_ids = user_ids
        self.diner_ids = diner_ids
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.num_nodes = num_nodes
        self.model_name = model_name
        self.device = device
        self.recommend_batch_size = recommend_batch_size
        self.num_workers = num_workers
        self.EPS = 1e-15
        self.num_users = len(self.user_ids)
        self.num_diners = len(self.diner_ids)
        self.tr_loss = []

        # store metric value at each epoch
        self.metric_at_k_total_epochs = {
            k: {
                Metric.MAP: [],
                Metric.NDCG: [],
                Metric.RECALL: [],
                Metric.COUNT: 0,
            }
            for k in top_k_values
        }

        if self.model_name in ["node2vec", "metapath2vec"]:
            # trainable parameters
            self._embedding = Embedding(self.num_nodes, self.embedding_dim)
        elif self.model_name in ["graphsage", "lightgcn"]:
            # not trainable parameters, but result tensors from model forwarding
            self._embedding = torch.empty((self.num_nodes, self.embedding_dim)).to(
                self.device
            )
        else:
            # case when svd_bias
            pass

    def forward(self, batch: Tensor) -> Tensor:
        """
        Dummy forward pass which actually does not do anything.

        Args:
            batch (Tensor): A batch of node ids.

        Returns (Tensor):
            A batch of node embeddings.
        """
        emb = self._embedding.weight
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
            num_workers=self.num_workers,  # can be tuned depending on server spec
            pin_memory=True,  # to reduce data transfer btw cpu and gpu
            prefetch_factor=2,  # can be tuned depending on server spec
            **kwargs,
        )

    @abstractmethod
    def pos_sample(self, batch: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def neg_sample(self, batch: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        raise NotImplementedError

    def _pos_sample(self, batch: Tensor) -> Tensor:
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

    def _neg_sample(self, batch: Tensor) -> Tensor:
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

    def _neg_sample_from_train_nodes(self, batch: Tensor) -> Tensor:
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
        train_num_nodes = len(self.graph.nodes)

        indices = torch.randint(
            train_num_nodes,
            (batch.size(0), self.num_negative_samples),
            dtype=batch.dtype,
            device=batch.device,
        )
        negative_samples = torch.index_select(
            input=torch.tensor(list(self.graph.nodes)),
            dim=0,
            index=indices.view(-1),
        ).view(batch.size(0), self.num_negative_samples)

        negative_samples = torch.cat([batch.view(-1, 1), negative_samples], dim=-1)

        return negative_samples

    def _recommend(
        self,
        user_id: Tensor,
        already_liked_item_id: List[int],
        top_k: int = 10,
        near_diner_ids: List[int] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        For qualitative evaluation, calculate score for `one` user.

        Args:
             user_id (Tensor): target user_id.
             already_liked_item_id (List[int]): diner_ids that are already liked by user_id.
             top_k (int): number of diners to recommend to user_id.
             near_diner_ids (List[int]): List of diner_ids within specified distance.

        Returns (Tuple[NDArray, NDArray]):
            top_k diner_ids and associated scores.
        """
        user_embed = self.get_embedding(user_id)
        diner_embeds = self.get_embedding(self.diner_ids)
        score = torch.mm(user_embed, diner_embeds.t()).squeeze(0)
        # filter already liked diner_id
        for diner_id in already_liked_item_id:
            score[diner_id] = -float("inf")
        # filter near diner_id
        if near_diner_ids is not None:
            for diner_id in self.diner_ids:
                if diner_id.item() not in near_diner_ids:
                    score[diner_id] = -float("inf")
        top_k = torch.topk(score, k=top_k)
        pred_liked_item_id = top_k.indices.detach().cpu().numpy()
        pred_liked_item_score = top_k.values.detach().cpu().numpy()
        return pred_liked_item_id, pred_liked_item_score

    def get_embedding(self, batch_tensor: Tensor) -> Tensor:
        if self.model_name in ["node2vec", "metapath2vec"]:
            return self._embedding(batch_tensor)
        elif self.model_name == "graphsage":
            return self._embedding[batch_tensor]
        else:
            raise ValueError(
                f"get_embedding method must be overwritten when model is not one of node2vec, metapath2vec, graphsage, got :{self.mode_name}"
            )

    def generate_candidates_for_each_user(self, top_k_value: int) -> pd.DataFrame:
        start = 0
        diner_embeds = self.get_embedding(self.diner_ids.to(self.device))
        res = torch.tensor([], dtype=torch.float32)

        while start < self.num_users:
            batch_users = self.user_ids[start : start + self.recommend_batch_size]
            user_embeds = self.get_embedding(batch_users.to(self.device))
            scores = torch.mm(user_embeds, diner_embeds.t())
            top_k = torch.topk(scores, k=top_k_value)
            top_k_id = top_k.indices
            top_k_score = top_k.values
            candi = torch.cat(
                (
                    batch_users.repeat_interleave(top_k_value).view(-1, 1),
                    top_k_id.view(-1, 1),
                    top_k_score.view(-1, 1),
                ),
                dim=1,
            )
            res = torch.cat((res, candi), dim=0)
            start += self.recommend_batch_size
        dtypes = {"user_id": np.int64, "diner_id": np.int64, "score": np.float64}
        res = pd.DataFrame(res.detach().numpy(), columns=list(dtypes.keys())).astype(
            dtypes
        )
        return res
