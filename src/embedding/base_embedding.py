from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

from constant.metric.metric import Metric, NearCandidateMetric
from evaluation.metric import fully_vectorized_ranking_metrics_at_k
from tools.generate_walks import generate_walks
from tools.utils import safe_divide


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
                NearCandidateMetric.RANKED_PREC: [],
                NearCandidateMetric.RANKED_PREC_COUNT: 0,
                NearCandidateMetric.NEAR_RECALL: [],
                NearCandidateMetric.RECALL_COUNT: 0,
            }
            for k in top_k_values
        }

        if self.model_name in ["node2vec", "metapath2vec"]:
            # trainable parameters
            self._embedding = Embedding(self.num_nodes, self.embedding_dim)
        elif self.model_name == "graphsage":
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

    def generate_recommendations_and_calculate_metric(
        self,
        X_train: Tensor,
        X_val_warm_users: Tensor,
        X_val_cold_users: Tensor,
        top_k_values: List[int],
        metric_at_k: Dict,
        most_popular_diner_ids: List[int],
        filter_already_liked: bool = True,
    ) -> Dict:
        """
        Generate recommendations for warm / cold start users separately and calculate metric.
        For warm start users, trained user embedding will be used.
        For cold start users, most popular items from train data will be used.

        Args:
            X_train (Tensor): (diner_idx, reviewer_id) tensor from train dataset.
            X_val_warm_users (Tensor): (diner_idx, reviewer_id) tensor from val dataset with warm_start_users only.
            X_val_cold_users (Tensor): (diner_idx, reviewer_id) tensor from val dataset with cold_start_users only.
            top_k_values (List[int]): List of top k values to evaluate.
            metric_at_k (Dict): Dictionary to store metric result.
            most_popular_diner_ids (List[int]): List of most popular diner_ids from train dataset.

        Returns (Dict):
            Dictionary with calculated metric for warm / cold start users.
        """
        metric_at_k_warm_users_updated = (
            self._generate_recommendations_and_calculate_metric_for_warm_start_users(
                X_train=X_train,
                X_val_warm_users=X_val_warm_users,
                top_k_values=top_k_values,
                metric_at_k=metric_at_k,
                filter_already_liked=filter_already_liked,
            )
        )
        metric_at_k_warm_cold_users_updated = (
            self._generate_recommendations_and_calculate_metric_for_cold_start_users(
                X_val_cold_users=X_val_cold_users,
                top_k_values=top_k_values,
                most_popular_diner_ids=most_popular_diner_ids,
                metric_at_k=metric_at_k_warm_users_updated,
            )
        )
        return metric_at_k_warm_cold_users_updated

    def _generate_recommendations_and_calculate_metric_for_warm_start_users(
        self,
        X_train: Tensor,
        X_val_warm_users: Tensor,
        top_k_values: List[int],
        metric_at_k: Dict,
        filter_already_liked: bool = True,
    ) -> Dict:
        """
        Generate recommendations for warm start users separately and calculate metric.
        For warm start users, trained user embedding will be used.

        Args:
            X_train (Tensor): (diner_idx, reviewer_id) tensor from train dataset.
            X_val_warm_users (Tensor): (diner_idx, reviewer_id) tensor from val dataset with warm_start_users only.
            top_k_values (List[int]): List of top k values to evaluate.
            metric_at_k (Dict): Dictionary to store metric result.
            filter_already_liked (bool): Filter already liked item in train dataset when generating recommendations to users.

        Returns (Dict):
            Dictionary with calculated metric for warm start users.
        """
        max_k = max(top_k_values)
        start = 0
        diner_embeds = self.get_embedding(self.diner_ids)

        self.train_liked_series = (
            pd.DataFrame(X_train, columns=["diner_idx", "reviewer_id"])
            .groupby("reviewer_id")["diner_idx"]
            .apply(np.array)
        )
        self.train_liked = (
            self.train_liked_series.to_frame()
            .reset_index()
            .assign(num_liked_items=lambda df: df["diner_idx"].apply(len))
        )
        self.val_liked_series = (
            pd.DataFrame(X_val_warm_users, columns=["diner_idx", "reviewer_id"])
            .groupby("reviewer_id")["diner_idx"]
            .apply(np.array)
        )
        self.val_liked = (
            self.val_liked_series.to_frame()
            .reset_index()
            .assign(num_liked_items=lambda df: df["diner_idx"].apply(len))
            .sort_values(by="reviewer_id")
        )

        liked_items_count2user_ids = (
            self.val_liked.groupby("num_liked_items")["reviewer_id"]
            .apply(np.array)
            .to_dict()
        )

        for count, user_ids in liked_items_count2user_ids.items():
            num_users = len(user_ids)
            start = 0
            user_ids = torch.tensor(user_ids, device=self.device)

            while start < num_users:
                batch_users = user_ids[start : start + self.recommend_batch_size]
                user_embeds = self.get_embedding(batch_users)
                scores = torch.mm(user_embeds, diner_embeds.t())
                liked_items_by_batch_users = []

                # TODO: change for loop to more efficient program
                # filter diner id already liked by user in train dataset
                if filter_already_liked:
                    for i, user_id in enumerate(batch_users):
                        already_liked_ids = self.train_liked_series[user_id.item()]
                        scores[i][already_liked_ids] = -float("inf")

                batch_users_np = batch_users.detach().cpu().numpy()
                liked_items_by_batch_users = np.vstack(
                    self.val_liked[lambda x: x["reviewer_id"].isin(batch_users_np)][
                        "diner_idx"
                    ].values
                )

                max_k = min(scores.shape[1], max_k)  # to prevent index error in pytest
                top_k = torch.topk(scores, k=max_k)
                top_k_id = top_k.indices

                self.calculate_metric_at_current_batch(
                    metric_at_k=metric_at_k,
                    top_k_id=top_k_id.detach().cpu().numpy(),
                    liked_items=liked_items_by_batch_users,
                    top_k_values=top_k_values,
                )

                start += self.recommend_batch_size

        return metric_at_k

    def _generate_recommendations_and_calculate_metric_for_cold_start_users(
        self,
        X_val_cold_users: torch.Tensor,
        most_popular_diner_ids: List[int],
        top_k_values: List[int],
        metric_at_k: Dict,
    ) -> Dict:
        """
        Generate recommendations for cold start users separately and calculate metric.
        For cold start users, most popular items from train data will be used.
        Note that filter_already_liked argument cannot be used because for cold_start_users, there are not any liked items in train dataset.

        Args:
            X_val_cold_users (Tensor): (diner_idx, reviewer_id) tensor from val dataset with cold_start_users only.
            most_popular_diner_ids (List[int]): List of most popular diner_ids from train dataset.
            top_k_values (List[int]): List of top k values to evaluate.
            metric_at_k (Dict): Dictionary to store metric result.

        Returns (Dict):
            Dictionary with calculated metric for cold start users.
        """
        if X_val_cold_users.size(0) == 0:
            return metric_at_k
        val_liked_cold_users = (
            pd.DataFrame(X_val_cold_users, columns=["diner_idx", "reviewer_id"])
            .groupby("reviewer_id")["diner_idx"]
            .apply(np.array)
            .to_frame()
            .reset_index()
            .assign(num_liked_items=lambda df: df["diner_idx"].apply(len))
            .sort_values(by="reviewer_id")
        )
        liked_items_count2user_ids = (
            val_liked_cold_users.groupby("num_liked_items")["reviewer_id"]
            .apply(np.array)
            .to_dict()
        )

        for count, user_ids in liked_items_count2user_ids.items():
            num_users = len(user_ids)
            start = 0
            while start < num_users:
                batch_users = user_ids[start : start + self.recommend_batch_size]
                most_popular_reco_items = np.tile(
                    most_popular_diner_ids, (len(batch_users), 1)
                )

                liked_items_by_batch_users = np.vstack(
                    val_liked_cold_users[lambda x: x["reviewer_id"].isin(batch_users)][
                        "diner_idx"
                    ].values
                )

                self.calculate_metric_at_current_batch(
                    metric_at_k=metric_at_k,
                    top_k_id=most_popular_reco_items,
                    liked_items=liked_items_by_batch_users,
                    top_k_values=top_k_values,
                )

                start += self.recommend_batch_size

        return metric_at_k

    def calculate_metric_at_current_epoch(
        self, metric_at_k: Dict, top_k_values: List[int]
    ) -> None:
        """
        Calculate metric at current epoch using validation data.

        Args:
            metric_at_k (Dict): Dictionary to store metric result.
            top_k_values (List[int]): List of top k values to evaluate.
        """
        for k in top_k_values:
            # save map
            metric_at_k[k][Metric.MAP] = safe_divide(
                numerator=metric_at_k[k][Metric.MAP],
                denominator=metric_at_k[k][Metric.COUNT],
            )
            self.metric_at_k_total_epochs[k][Metric.MAP].append(
                metric_at_k[k][Metric.MAP]
            )

            # save ndcg
            metric_at_k[k][Metric.NDCG] = safe_divide(
                numerator=metric_at_k[k][Metric.NDCG],
                denominator=metric_at_k[k][Metric.COUNT],
            )
            self.metric_at_k_total_epochs[k][Metric.NDCG].append(
                metric_at_k[k][Metric.NDCG]
            )

            # save recall
            metric_at_k[k][Metric.RECALL] = safe_divide(
                numerator=metric_at_k[k][Metric.RECALL],
                denominator=metric_at_k[k][Metric.COUNT],
            )
            self.metric_at_k_total_epochs[k][Metric.RECALL].append(
                metric_at_k[k][Metric.RECALL]
            )

            # save count
            self.metric_at_k_total_epochs[k][Metric.COUNT] = metric_at_k[k][
                Metric.COUNT
            ]

    def calculate_metric_at_current_batch(
        self,
        metric_at_k: Dict,
        top_k_id: NDArray,
        liked_items: NDArray,
        top_k_values: List[int],
    ) -> Dict:
        """
        For each batch when generating recommendations in warm / cold start users, calculate metric using vectorized function.
        Note that dimension of top_k_id (num_users_in_batch, 500) and dimension of liked_items is (num_users_in_batch, num_liked_items).
        Because upper loop is run with users who have identical number of liked items, this vectorized calculation is possible.

        Args:
            metric_at_k (Dict): Dictionary to store metric result.
             top_k_id (NDArray): Diner_id whose score is under max_k ranked score. (two dimensional array)
             liked_items (NDArray): Item ids liked by users. (two dimensional array)
             top_k_values (List[int]): A list of k values.

        Returns (Dict):
            Calculated metric for each user.
        """

        batch_num_users = liked_items.shape[0]

        for k in top_k_values:
            pred_liked_item_id = top_k_id[:, :k]
            metric = fully_vectorized_ranking_metrics_at_k(
                liked_items, pred_liked_item_id
            )
            metric_at_k[k][Metric.MAP] += metric[Metric.AP].sum()
            metric_at_k[k][Metric.NDCG] += metric[Metric.NDCG].sum()
            metric_at_k[k][Metric.RECALL] += metric[Metric.RECALL].sum()
            metric_at_k[k][Metric.COUNT] += batch_num_users

        return metric_at_k

    def _recommend(
        self,
        user_id: Tensor,
        already_liked_item_id: List[int],
        top_k: int = 10,
    ) -> Tuple[NDArray, NDArray]:
        """
        For qualitative evaluation, calculate score for `one` user.

        Args:
             user_id (Tensor): target user_id.
             already_liked_item_id (List[int]): diner_ids that are already liked by user_id.
             top_k (int): number of diners to recommend to user_id.
             # TODO
             latitude: user's current latitude
             longitude: user's current longitude

        Returns (Tuple[NDArray, NDArray]):
            top_k diner_ids and associated scores.
        """
        user_embed = self.get_embedding(user_id)
        diner_embeds = self.get_embedding(self.diner_ids)
        score = torch.mm(user_embed, diner_embeds.t()).squeeze(0)
        for diner_idx in already_liked_item_id:
            score[diner_idx] = -float("inf")
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
