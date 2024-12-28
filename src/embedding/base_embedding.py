from abc import abstractmethod
from typing import List, Union, Tuple, Dict, Any
import torch
import numpy as np
from numpy.typing import NDArray

from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from constant.device.device import DEVICE
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
            filter_already_liked=True,
        ) -> Tuple[Tensor, Tensor, Tensor]:
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
                scores[user_id - self.num_diners][diner_id] = -float('inf')
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
        ):
        # prepare for metric calculation
        self.metric_at_k = {
            k: {
                "map": 0,
                "ndcg": 0,
                "no_candidate_count": 0,
                "ranked_prec": 0,
                "near_candidate_recall": 0,
                "near_candidate_prec_count": 0,
                "near_candidate_recall_count": 0,
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
                    self.metric_at_k[k]["map"] += metric["ap"]
                    self.metric_at_k[k]["ndcg"] += metric["ndcg"]
                    self.metric_at_k[k]["no_candidate_count"] += 1

        for k in top_k_values:
            self.metric_at_k[k]["map"] /= self.metric_at_k[k]["no_candidate_count"]
            self.metric_at_k[k]["ndcg"] /= self.metric_at_k[k]["no_candidate_count"]

    def calculate_near_candidate_metric(
            self,
            scores: Tensor,
            nearby_candidates: Dict[int, list],
            top_k_values: List[int],
        ):
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

                    # calculate metric
                    self.metric_at_k[k]["ranked_prec"] += ranked_precision(
                        liked_item=location,
                        reco_items=near_diner_ids_sorted.detach().cpu().numpy(),
                    )
                    self.metric_at_k[k]["near_candidate_prec_count"] += 1

                    if len(locations) > k:
                        # ranked_prec value higher than 0 indicates hitting of true y
                        self.metric_at_k[k]["near_candidate_recall"] += (self.metric_at_k[k]["ranked_prec"] > 0.)
                        self.metric_at_k[k]["near_candidate_recall_count"] += 1

        for k in top_k_values:
            self.metric_at_k[k]["ranked_prec"] /= self.metric_at_k[k]["near_candidate_prec_count"]
            self.metric_at_k[k]["near_candidate_recall"] /= self.metric_at_k[k]["near_candidate_recall_count"]


    def recommend(
            self,
            X_train: Tensor,
            X_val: Tensor,
            nearby_candidates: Dict[int, list],
            top_K = [3, 5, 7, 10, 20, 100, 200, 300, 400, 500],
            filter_already_liked=True
        ) -> Dict[int, Any]:
        user_embeds = self.embedding(self.user_ids)
        diner_embeds = self.embedding(self.diner_ids)
        scores = torch.mm(user_embeds, diner_embeds.t())

        self.map = 0.
        self.ndcg = 0.

        train_liked = convert_tensor(X_train, list)
        val_liked = convert_tensor(X_val, list)
        res = {}
        metric_at_K = {k: {"map": 0, "ndcg": 0, "count": 0, "ranked_prec": 0, "near_candidate_recall": 0} for k in top_K}

        # filter item_id in train dataset
        if filter_already_liked:
            for user_id in self.user_ids:
                user_id = user_id.item()
                user_liked_items = train_liked[user_id]
                for already_liked_item_id in user_liked_items:
                    scores[user_id - self.num_diners][already_liked_item_id] = -float('inf')  # not recommend already chosen item_id

        # calculate metric
        for user_id in self.user_ids:
            user_id = user_id.item()
            val_liked_item_id = np.array(val_liked[user_id])

            # diner_ids visited by user in validation dataset
            locations = val_liked[user_id]

            for K in top_K:
                score = scores[user_id - self.num_diners]
                pred_liked_item_id = torch.topk(score, k=K).indices.detach().cpu().numpy()
                if len(val_liked_item_id) >= K:
                    metric = ranking_metrics_at_k(val_liked_item_id, pred_liked_item_id)
                    metric_at_K[K]["map"] += metric["ap"]
                    metric_at_K[K]["ndcg"] += metric["ndcg"]
                    metric_at_K[K]["count"] += 1

                for location in locations:
                    # filter only near diner
                    near_diner = np.array(nearby_candidates[location])
                    near_diner_score = np.array([score[i].item() for i in near_diner])

                    # sort indices using predicted score
                    indices = np.argsort(near_diner_score)[::-1]
                    pred_near_liked_item_id = near_diner[indices][:K]
                    metric_at_K[K]["ranked_prec"] += ranked_precision(location, pred_near_liked_item_id)
                    # ranked_prec value higher than 0 indicates hitting of true y
                    metric_at_K[K]["near_candidate_recall"] += (metric_at_K[K]["ranked_prec"] != 0.)

                # store recommendation result when K=20
                if K == 20:
                    res[user_id] = pred_liked_item_id
        for K in top_K:
            metric_at_K[K]["map"] /= metric_at_K[K]["count"]
            metric_at_K[K]["ndcg"] /= metric_at_K[K]["count"]
            metric_at_K[K]["ranked_prec"] /= X_val.shape[0]
            metric_at_K[K]["near_candidate_recall"] /= X_val.shape[0]
        self.metric_at_K = metric_at_K
        return res

    def _recommend(
            self,
            user_id: Tensor,
            already_liked_item_id: List[int],
            latitude: float = None,
            longitude: float = None,
            top_k: int = 10,
    ) -> Tuple[NDArray, NDArray]:
        user_embed = self.embedding(user_id)
        diner_embeds = self.embedding(self.diner_ids)
        score = torch.mm(user_embed, diner_embeds.t()).squeeze(0)
        for diner_idx in already_liked_item_id:
            score[diner_idx] = -float('inf')
        top_k = torch.topk(score, k=top_k)
        pred_liked_item_id = top_k.indices.detach().cpu().numpy()
        pred_liked_item_score = top_k.values.detach().cpu().numpy()
        return pred_liked_item_id, pred_liked_item_score