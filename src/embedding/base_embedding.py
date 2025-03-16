from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

from constant.device.device import DEVICE
from constant.evaluation.recommend import RECOMMEND_BATCH_SIZE
from constant.metric.metric import Metric, NearCandidateMetric
from evaluation.metric import ranked_precision, ranking_metrics_at_k
from tools.generate_walks import generate_walks
from tools.utils import convert_tensor, safe_divide


class BaseEmbedding(nn.Module):
    def __init__(
        self,
        user_ids: Tensor,
        diner_ids: Tensor,
        top_k_values: List[int],
        graph: nx.Graph,
        embedding_dim: int,
        walks_per_node: int,
        num_negative_samples: int,
        num_nodes: int,
        model_name: str,
    ):
        super().__init__()
        self.user_ids = user_ids
        self.diner_ids = diner_ids
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.num_nodes = num_nodes
        self.model_name = model_name
        self.EPS = 1e-15
        self.num_users = len(self.user_ids)
        self.num_diners = len(self.diner_ids)
        self.tr_loss = []

        # store metric value at each epoch
        self.metric_at_k_total_epochs = {
            k: {
                Metric.MAP.value: [],
                Metric.NDCG.value: [],
                Metric.RECALL.value: [],
                Metric.COUNT.value: 0,
                NearCandidateMetric.RANKED_PREC.value: [],
                NearCandidateMetric.RANKED_PREC_COUNT.value: 0,
                NearCandidateMetric.NEAR_RECALL.value: [],
                NearCandidateMetric.RECALL_COUNT.value: 0,
            }
            for k in top_k_values
        }

        if self.model_name in ["node2vec", "metapath2vec"]:
            # trainable parameters
            self._embedding = Embedding(self.num_nodes, self.embedding_dim)
        else:
            # not trainable parameters, but result tensors from model forwarding
            self._embedding = torch.empty((self.num_nodes, self.embedding_dim))

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

    def recommend_all(
        self,
        X_train: Tensor,
        X_val: Tensor,
        top_k_values: List[int],
        nearby_candidates: Dict[int, list],
        filter_already_liked: bool = True,
    ) -> None:
        """
        Generate diner recommendations for all users.
        Suppose number of users is U and number of diners is D.
        The dimension of associated matrix between users and diners is U x D.
        However, to avoid out of memory error, batch recommendation is run.
        For every batch users, we calculate metric when there are no candidates,
        and there are near diner candidates.

            - when there are no candidates:
                map, ndcg, recall, ranked_prec are calculated at @3, @7, @10, @20
            - when there are near diner candidates:
                recall is calculated at @100, @300, @500

        Args:
             X_train (Tensor): number of reviews x (diner_id, reviewer_id) in train dataset.
             X_val (Tensor): number of reviews x (diner_id, reviewer_id) in val dataset.
             top_k_values (List[int]): a list of k values.
             nearby_candidates (Dict[int, List[int]]): near diners around ref diners with 1km.
             epoch (int): current epoch.
             filter_already_liked (bool): whether filtering pre-liked diner in train dataset or not.
        """
        # prepare for metric calculation
        # refresh at every epoch
        self.metric_at_k = {
            k: {
                Metric.MAP.value: 0,
                Metric.NDCG.value: 0,
                Metric.RECALL.value: 0,
                Metric.COUNT.value: 0,
                NearCandidateMetric.RANKED_PREC.value: 0,
                NearCandidateMetric.RANKED_PREC_COUNT.value: 0,
                NearCandidateMetric.NEAR_RECALL.value: 0,
                NearCandidateMetric.RECALL_COUNT.value: 0,
            }
            for k in top_k_values
        }
        max_k = max(top_k_values)
        start = 0
        diner_embeds = self.get_embedding(self.diner_ids)

        # store true diner id visited by user in validation dataset
        self.train_liked = convert_tensor(X_train, list)
        self.val_liked = convert_tensor(X_val, list)

        while start < self.num_users:
            batch_users = self.user_ids[start : start + RECOMMEND_BATCH_SIZE]
            user_embeds = self.get_embedding(batch_users)
            scores = torch.mm(user_embeds, diner_embeds.t())

            # TODO: change for loop to more efficient program
            # filter diner id already liked by user in train dataset
            if filter_already_liked:
                for i, user_id in enumerate(batch_users):
                    already_liked_ids = self.train_liked[user_id.item()]
                    for diner_id in already_liked_ids:
                        scores[i][diner_id] = -float("inf")

            max_k = min(scores.shape[1], max_k)  # to prevent index error in pytest
            top_k = torch.topk(scores, k=max_k)
            top_k_id = top_k.indices

            self.calculate_no_candidate_metric(
                user_ids=batch_users, top_k_id=top_k_id, top_k_values=top_k_values
            )

            self.calculate_near_candidate_metric(
                user_ids=batch_users,
                scores=scores,
                nearby_candidates=nearby_candidates,
                top_k_values=top_k_values,
            )

            start += RECOMMEND_BATCH_SIZE

        for k in top_k_values:
            # save map
            self.metric_at_k[k][Metric.MAP.value] = safe_divide(
                numerator=self.metric_at_k[k][Metric.MAP.value],
                denominator=self.metric_at_k[k][Metric.COUNT.value],
            )
            self.metric_at_k_total_epochs[k][Metric.MAP.value].append(
                self.metric_at_k[k][Metric.MAP.value]
            )

            # save ndcg
            self.metric_at_k[k][Metric.NDCG.value] = safe_divide(
                numerator=self.metric_at_k[k][Metric.NDCG.value],
                denominator=self.metric_at_k[k][Metric.COUNT.value],
            )
            self.metric_at_k_total_epochs[k][Metric.NDCG.value].append(
                self.metric_at_k[k][Metric.NDCG.value]
            )

            # save recall
            self.metric_at_k[k][Metric.RECALL.value] = safe_divide(
                numerator=self.metric_at_k[k][Metric.RECALL.value],
                denominator=self.metric_at_k[k][Metric.COUNT.value],
            )
            self.metric_at_k_total_epochs[k][Metric.RECALL.value].append(
                self.metric_at_k[k][Metric.RECALL.value]
            )

            # save ranked_prec
            self.metric_at_k[k][NearCandidateMetric.RANKED_PREC.value] = safe_divide(
                numerator=self.metric_at_k[k][NearCandidateMetric.RANKED_PREC.value],
                denominator=self.metric_at_k[k][
                    NearCandidateMetric.RANKED_PREC_COUNT.value
                ],
            )
            self.metric_at_k_total_epochs[k][
                NearCandidateMetric.RANKED_PREC.value
            ].append(self.metric_at_k[k][NearCandidateMetric.RANKED_PREC.value])

            # save near recall
            self.metric_at_k[k][NearCandidateMetric.NEAR_RECALL.value] = safe_divide(
                numerator=self.metric_at_k[k][NearCandidateMetric.NEAR_RECALL.value],
                denominator=self.metric_at_k[k][NearCandidateMetric.RECALL_COUNT.value],
            )
            self.metric_at_k_total_epochs[k][
                NearCandidateMetric.NEAR_RECALL.value
            ].append(self.metric_at_k[k][NearCandidateMetric.NEAR_RECALL.value])

            # save count
            self.metric_at_k_total_epochs[k][Metric.COUNT.value] = self.metric_at_k[k][
                Metric.COUNT.value
            ]
            self.metric_at_k_total_epochs[k][
                NearCandidateMetric.RANKED_PREC_COUNT.value
            ] = self.metric_at_k[k][NearCandidateMetric.RANKED_PREC_COUNT.value]
            self.metric_at_k_total_epochs[k][NearCandidateMetric.RECALL_COUNT.value] = (
                self.metric_at_k[k][NearCandidateMetric.RECALL_COUNT.value]
            )

    def calculate_no_candidate_metric(
        self,
        user_ids: Tensor,
        top_k_id: Tensor,
        top_k_values: List[int],
    ) -> None:
        """
        After calculating scores in `recommend_all` function, calculate metric without any candidates.
        Metrics calculated in this function are NDCG, mAP and recall.
        Note that this function does not consider locality, which means recommendations
        could be given regardless of user's location and diner's location

        Args:
             user_ids (Tensor): batch of user ids.
             top_k_id (Tensor): diner_id whose score is under max_k ranked score.
             top_k_values (List[int]): a list of k values.
        """

        # TODO: change for loop to more efficient program
        # calculate metric
        for i, user_id in enumerate(user_ids):
            user_id = user_id.item()
            val_liked_item_id = np.array(self.val_liked[user_id])

            for k in top_k_values:
                pred_liked_item_id = top_k_id[i][:k].detach().cpu().numpy()
                metric = ranking_metrics_at_k(val_liked_item_id, pred_liked_item_id)
                self.metric_at_k[k][Metric.MAP.value] += metric[Metric.AP.value]
                self.metric_at_k[k][Metric.NDCG.value] += metric[Metric.NDCG.value]
                self.metric_at_k[k][Metric.RECALL.value] += metric[Metric.RECALL.value]
                self.metric_at_k[k][Metric.COUNT.value] += 1

    def calculate_near_candidate_metric(
        self,
        user_ids: Tensor,
        scores: Tensor,
        nearby_candidates: Dict[int, list],
        top_k_values: List[int],
    ) -> None:
        """
        After calculating scores in `recommend_all` function, calculate metric with near candidates.
        Metrics calculated in this function are ranked_prec and recall.
        Note that this function does consider locality, which means recommendations
        could be given based on user's location and diner's location.
        Each row in validation dataset contains latitude ad longitude of user's rating's diner.
        We suppose that location of each user in each row in val dataset is location of each diner.

        Args:
             user_ids (Tensor): batch of user ids.
             scores (Tensor): calculated scores with all users and diners.
             nearby_candidates (Dict[int, List[int]]): near diners around ref diners with 1km
             top_k_values (List[int]): a list of k values.
        """
        # TODO: change for loop to more efficient program
        # calculate metric
        for i, user_id in enumerate(user_ids):
            user_id = user_id.item()
            for k in top_k_values:
                # diner_ids visited by user in validation dataset
                locations = self.val_liked[user_id]
                for location in locations:
                    # filter only near diner
                    near_diner_ids = torch.tensor(nearby_candidates[location]).to(
                        DEVICE
                    )
                    near_diner_scores = scores[i][near_diner_ids]

                    # sort indices using predicted score
                    sorted_indices = torch.argsort(near_diner_scores, descending=True)
                    near_diner_ids_sorted = near_diner_ids[sorted_indices].to(DEVICE)

                    # top k filtering
                    near_diner_ids_sorted = near_diner_ids_sorted[:k]

                    # calculate metric
                    self.metric_at_k[k][NearCandidateMetric.RANKED_PREC.value] += (
                        ranked_precision(
                            liked_item=location,
                            reco_items=near_diner_ids_sorted.detach().cpu().numpy(),
                        )
                    )
                    self.metric_at_k[k][
                        NearCandidateMetric.RANKED_PREC_COUNT.value
                    ] += 1

                    if near_diner_ids.shape[0] > k:
                        recall = 1 if location in near_diner_ids_sorted else 0
                        self.metric_at_k[k][NearCandidateMetric.NEAR_RECALL.value] += (
                            recall
                        )
                        self.metric_at_k[k][NearCandidateMetric.RECALL_COUNT.value] += 1

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

    def get_embedding(self, batch_tensor: Tensor):
        if self.model_name in ["node2vec", "metapath2vec"]:
            return self._embedding(batch_tensor)
        else:
            return self._embedding[batch_tensor]
