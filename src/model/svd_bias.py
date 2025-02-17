from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor

from evaluation.metric import ranked_precision, ranking_metrics_at_k
from tools.utils import convert_tensor, safe_divide


class Model(nn.Module):
    def __init__(self, num_users: int, num_items: int, num_factors: int, **kwargs):
        """
        Args:
            num_users (int): number of unique users across train / validation dataset.
            num_items (int): number of unique items (diners) across train / validation dataset.
            num_factors (int): dimension size of embedding vector.
        """
        super(Model, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.embed_user = nn.Embedding(num_users, num_factors)
        self.embed_item = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.mu = kwargs["mu"]

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)
        nn.init.xavier_normal_(self.user_bias.weight)
        nn.init.xavier_normal_(self.item_bias.weight)

    def forward(self, user_idx: Tensor, item_idx: Tensor) -> Tensor:
        """
        Forward pass for SVD Bias model.
        Predicts user's rating related with an item id.
        This forward pass decomposes rating value into product of user embedding and item embedding
        with each of bias included.

        Args:
            user_idx (Tensor): User id.
            item_idx (Tensor): Item id.

        Returns (Tensor):
            Predicted scores of each user related with item ids.
        """
        embed_user = self.embed_user(user_idx)  # batch_size * num_factors
        embed_item = self.embed_item(item_idx)  # batch_size * num_factors
        user_bias = self.user_bias(user_idx)  # batch_size * 1
        item_bias = self.item_bias(item_idx)  # batch_size * 1
        output = (
            (embed_user * embed_item).sum(axis=1)
            + user_bias.squeeze()
            + item_bias.squeeze()
            + self.mu
        )  # batch_size * 1
        return output

    def recommend(
        self,
        X_train: Tensor,
        X_val: Tensor,
        nearby_candidates: Dict[int, List[int]],
        top_K: List[int] = [3, 5, 7, 10, 20],
        filter_already_liked: bool = True,
    ) -> Dict[int, NDArray]:
        """
        Recommend item to each user based on predicted scores.
        Recommendations on two ways are performed.
         - Recommend items to each user not considering user's locality.
           -> Calculates NDCG, mAP metric.
         - Recommend items to each user considering user's locality.
           -> Calculates ranked precision metric.
        Second method gets candidates from `NearCandidateGenerator`,
        which filters diners within x km distance given user's latitude and longitude.

        Args:
            X_train (Tensor): Dataset used when training model.
                When recommendation, this is used when filtering items that already liked by user.
            X_val (Tensor): Dataset used when validation model.
            nearby_candidates (Dict[int, List[int]]): Each key is reference diner, and
                corresponding value is a list of diners within x km distance with reference diner.
            top_K (List[int]): A list of number of items to recommend to user.
            filter_already_liked (bool): Whether to filter items that already liked
                by user in train dataset.

        Returns (Dict[int, NDArray]):
            Defined metric will be stored in class attribute `metric_at_k`. This function returns
            recommendation item list at `20` of each user.
        """

        self.map = 0.0
        self.ndcg = 0.0

        train_liked = convert_tensor(X_train, dict)
        val_liked = convert_tensor(X_val, list)
        res = {}
        metric_at_K = {
            k: {"map": 0, "ndcg": 0, "count": 0, "ranked_prec": 0} for k in top_K
        }
        for user in range(self.num_users):
            item_idx = torch.arange(self.num_items)
            user_idx = torch.tensor([user]).repeat(self.num_items)

            # diner_ids visited by user in validation dataset
            locations = val_liked[user]

            # calculate one user's predicted scores for all item_ids
            with torch.no_grad():
                scores = self.forward(user_idx, item_idx)

            # filter item_id in train dataset
            if filter_already_liked:
                user_liked_items = train_liked[user]
                for already_liked_item_id in user_liked_items.keys():
                    scores[already_liked_item_id] = -float(
                        "inf"
                    )  # not recommend already chosen item_id

            # calculate metric
            val_liked_item_id = np.array(val_liked[user])
            for K in top_K:
                if len(val_liked_item_id) < K:
                    continue

                # recommendations for all item pools
                pred_liked_item_id = (
                    torch.topk(scores, k=K).indices.detach().cpu().numpy()
                )
                metric = ranking_metrics_at_k(val_liked_item_id, pred_liked_item_id)
                metric_at_K[K]["map"] += metric["ap"]
                metric_at_K[K]["ndcg"] += metric["ndcg"]
                metric_at_K[K]["count"] += 1

                for location in locations:
                    # filter only near diner
                    near_diner = np.array(nearby_candidates[location])
                    near_diner_score = np.array([scores[i].item() for i in near_diner])

                    # sort indices using predicted score
                    indices = np.argsort(near_diner_score)[::-1]
                    pred_near_liked_item_id = near_diner[indices][:K]
                    metric_at_K[K]["ranked_prec"] += ranked_precision(
                        location, pred_near_liked_item_id
                    )

                # store recommendation result when K=20
                if K == 20:
                    res[user] = pred_liked_item_id
        for K in top_K:
            metric_at_K[K]["map"] = safe_divide(
                numerator=metric_at_K[K]["map"], denominator=metric_at_K[K]["count"]
            )
            metric_at_K[K]["ndcg"] = safe_divide(
                numerator=metric_at_K[K]["ndcg"], denominator=metric_at_K[K]["count"]
            )
            metric_at_K[K]["ranked_prec"] = safe_divide(
                numerator=metric_at_K[K]["ranked_prec"], denominator=X_val.shape[0]
            )
        self.metric_at_K = metric_at_K
        return res
