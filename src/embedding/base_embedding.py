from abc import abstractmethod
from typing import List, Union, Tuple, Dict, Any
import torch
import numpy as np
from numpy.typing import NDArray

from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from constant.device.device import DEVICE
from constant.metric.metric import Metric, NearCandidateMetric
from tools.utils import convert_tensor
from evaluation.metric import ranking_metrics_at_k, ranked_precision


class BaseEmbedding(nn.Module):
    def __init__(
            self,
            user_ids: Tensor,
            diner_ids: Tensor
        ):
        super().__init__()
        self.user_ids = user_ids
        self.diner_ids = diner_ids
        self.num_users = len(self.user_ids)
        self.num_diners = len(self.diner_ids)

    @abstractmethod
    def forward(self, batch: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def loader(self, **kwargs) -> DataLoader:
        raise NotImplementedError

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

    def recommend_all(
            self,
            X_train: Tensor,
            X_val: Tensor,
            max_k: int,
            filter_already_liked: bool = True,
        ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes score between all users and all diners.
        Suppose number of users is U and number of diners is D.
        The dimension of associated matrix between users and diners is U x D.
        This function also precalculates top-k indices and their scores.

        Args:
             X_train (Tensor): number of reviews x (diner_id, reviewer_id) in train dataset.
             X_val (Tensor): number of reviews x (diner_id, reviewer_id) in val dataset.
             max_k (int): maximum k among top_ks [3, 7, 10, 20, etc].
             filter_already_liked (bool): whether filtering pre-liked diner in train dataset or not.

        Returns (Tuple[Tensor, Tensor, Tensor]):
            top_k_id (number_of_users x max_k): diner_id whose score is under max_k ranked score.
            top_k_score (number_of_users x max_k): associated score with top_k_id.
            scores (number_of_users x number_of_diners): calculated scores with all users and diners.
        """
        user_embeds = self.embedding(self.user_ids)
        diner_embeds = self.embedding(self.diner_ids)
        scores = torch.mm(user_embeds, diner_embeds.t())

        # TODO: change for loop to more efficient program
        # filter diner id already liked by user in train dataset
        if filter_already_liked:
            for diner_id, user_id in X_train:
                diner_id = diner_id.item()
                user_id = user_id.item()
                # not recommend already chosen item_id by setting prediction value as -inf
                scores[user_id - self.num_diners][diner_id] = -float("inf")
        # store true diner id visited by user in validation dataset
        self.val_liked = convert_tensor(X_val, list)

        top_k = torch.topk(scores, k=max_k)
        top_k_id = top_k.indices
        top_k_score = top_k.values

        return top_k_id, top_k_score, scores

    def calculate_no_candidate_metric(
            self,
            top_k_id: Tensor,
            top_k_values: List[int],
        ) -> None:
        """
        After calculating scores in `recommend_all` function, calculate metric without any candidates.
        Metrics calculated in this function are NDCG, mAP.
        Note that this function does not consider locality, which means recommendations
        could be given regardless of user's location and diner's location

        Args:
             top_k_id (Tensor): diner_id whose score is under max_k ranked score.
             top_k_values (List[int]): a list of k values.
        """
        # prepare for metric calculation
        self.metric_at_k = {
            k: {
                Metric.MAP.value: 0,
                Metric.NDCG.value: 0,
                Metric.RECALL.value: 0,
                Metric.COUNT.value: 0,
                NearCandidateMetric.RANKED_PREC.value: 0,
                NearCandidateMetric.RANKED_PREC_COUNT.value: 0,
                NearCandidateMetric.RECALL.value: 0,
                NearCandidateMetric.RECALL_COUNT.value: 0,
            }
            for k in top_k_values
        }

        # TODO: change for loop to more efficient program
        # calculate metric
        for user_id in self.user_ids:
            user_id = user_id.item()
            val_liked_item_id = np.array(self.val_liked[user_id])

            for k in top_k_values:
                pred_liked_item_id = top_k_id[user_id - self.num_diners][:k].detach().cpu().numpy()
                if len(val_liked_item_id) >= k:
                    metric = ranking_metrics_at_k(val_liked_item_id, pred_liked_item_id)
                    self.metric_at_k[k][Metric.MAP.value] += metric[Metric.AP.value]
                    self.metric_at_k[k][Metric.NDCG.value] += metric[Metric.NDCG.value]
                    self.metric_at_k[k][Metric.RECALL.value] += metric[Metric.RECALL.value]
                    self.metric_at_k[k][Metric.COUNT.value] += 1

        for k in top_k_values:
            self.metric_at_k[k][Metric.MAP.value] /= self.metric_at_k[k][Metric.COUNT.value]
            self.metric_at_k[k][Metric.NDCG.value] /= self.metric_at_k[k][Metric.COUNT.value]

    def calculate_near_candidate_metric(
            self,
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
             scores (Tensor): calculated scores with all users and diners.
             nearby_candidates (Dict[int, List[int]]): near diners around ref diners with 1km
             top_k_values (List[int]): a list of k values.
        """
        # TODO: change for loop to more efficient program
        # calculate metric
        for user_id in self.user_ids:
            user_id = user_id.item()
            for k in top_k_values:
                # diner_ids visited by user in validation dataset
                locations = self.val_liked[user_id]
                for location in locations:
                    # filter only near diner
                    near_diner_ids = torch.tensor(nearby_candidates[location]).to(DEVICE)
                    near_diner_scores = scores[user_id - self.num_diners][near_diner_ids]

                    # sort indices using predicted score
                    sorted_indices = torch.argsort(near_diner_scores, descending=True)
                    near_diner_ids_sorted = near_diner_ids[sorted_indices].to(DEVICE)

                    # top k filtering
                    near_diner_ids_sorted = near_diner_ids_sorted[:k]

                    # calculate metric
                    self.metric_at_k[k][NearCandidateMetric.RANKED_PREC.value] += ranked_precision(
                        liked_item=location,
                        reco_items=near_diner_ids_sorted.detach().cpu().numpy(),
                    )
                    self.metric_at_k[k][NearCandidateMetric.RANKED_PREC_COUNT.value] += 1

                    if near_diner_ids.shape[0] > k:
                        # ranked_prec value higher than 0 indicates hitting of true y
                        self.metric_at_k[k][NearCandidateMetric.RECALL.value] += (self.metric_at_k[k][NearCandidateMetric.RANKED_PREC.value] > 0.)
                        self.metric_at_k[k][NearCandidateMetric.RECALL_COUNT.value] += 1

        for k in top_k_values:
            self.metric_at_k[k][NearCandidateMetric.RANKED_PREC.value] /= self.metric_at_k[k][NearCandidateMetric.RANKED_PREC_COUNT.value]
            self.metric_at_k[k][NearCandidateMetric.RECALL.value] /= self.metric_at_k[k][NearCandidateMetric.RECALL_COUNT.value]

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
        user_embed = self.embedding(user_id)
        diner_embeds = self.embedding(self.diner_ids)
        score = torch.mm(user_embed, diner_embeds.t()).squeeze(0)
        for diner_idx in already_liked_item_id:
            score[diner_idx] = -float('inf')
        top_k = torch.topk(score, k=top_k)
        pred_liked_item_id = top_k.indices.detach().cpu().numpy()
        pred_liked_item_score = top_k.values.detach().cpu().numpy()
        return pred_liked_item_id, pred_liked_item_score