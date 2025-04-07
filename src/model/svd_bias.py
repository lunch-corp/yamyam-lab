from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor

from constant.metric.metric import Metric, NearCandidateMetric
from evaluation.metric import (
    fully_vectorized_ranking_metrics_at_k,
)
from tools.utils import safe_divide


class Model(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        top_k_values: List[int],
        **kwargs,
    ):
        """
        Args:
            num_users (int): number of unique users across train / validation dataset.
            num_items (int): number of unique items (diners) across train / validation dataset.
            embedding_dim (int): dimension size of embedding vector.
        """
        super(Model, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.embed_user = nn.Embedding(num_users, embedding_dim)
        self.embed_item = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.mu = kwargs["mu"]

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)
        nn.init.xavier_normal_(self.user_bias.weight)
        nn.init.xavier_normal_(self.item_bias.weight)

        self.tr_loss = []

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
        embed_user = self.embed_user(user_idx)  # batch_size * embedding_dim
        embed_item = self.embed_item(item_idx)  # batch_size * embedding_dim
        user_bias = self.user_bias(user_idx)  # batch_size * 1
        item_bias = self.item_bias(item_idx)  # batch_size * 1
        output = (
            (embed_user * embed_item).sum(axis=1)
            + user_bias.squeeze()
            + item_bias.squeeze()
            + self.mu
        )  # batch_size * 1
        return output

    def recommend_all(
        self,
        X_train: Tensor,
        X_val: Tensor,
        recommend_batch_size: int,
        top_k_values: List[int],
        filter_already_liked: bool = True,
    ) -> None:
        """
        Generate diner recommendations for all users.
        Suppose number of users is U and number of diners is D.
        The dimension of associated matrix between users and diners is U x D.
        However, to avoid out of memory error, batch recommendation is run.

        Args:
             X_train (Tensor): number of reviews x (diner_id, reviewer_id) in train dataset.
             X_val (Tensor): number of reviews x (diner_id, reviewer_id) in val dataset.
             top_k_values (List[int]): a list of k values.
             filter_already_liked (bool): whether filtering pre-liked diner in train dataset or not.
        """
        # prepare for metric calculation
        # refresh at every epoch
        self.metric_at_k = {
            k: {
                Metric.MAP: 0,
                Metric.NDCG: 0,
                Metric.RECALL: 0,
                Metric.COUNT: 0,
                NearCandidateMetric.RANKED_PREC: 0,
                NearCandidateMetric.RANKED_PREC_COUNT: 0,
                NearCandidateMetric.NEAR_RECALL: 0,
                NearCandidateMetric.RECALL_COUNT: 0,
            }
            for k in top_k_values
        }
        max_k = max(top_k_values)
        start = 0

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
            pd.DataFrame(X_val, columns=["diner_idx", "reviewer_id"])
            .groupby("reviewer_id")["diner_idx"]
            .apply(np.array)
        )
        self.val_liked = (
            self.val_liked_series.to_frame()
            .reset_index()
            .assign(num_liked_items=lambda df: df["diner_idx"].apply(len))
        )

        liked_items_count2user_ids = (
            self.val_liked.groupby("num_liked_items")["reviewer_id"]
            .apply(np.array)
            .to_dict()
        )

        for count, user_ids in liked_items_count2user_ids.items():
            num_users = len(user_ids)
            start = 0
            user_ids = torch.tensor(user_ids)

            while start < num_users:
                batch_users = user_ids[start : start + recommend_batch_size]
                user_embeds = self.embed_user(batch_users)
                scores = torch.mm(user_embeds, self.embed_item.weight.T)
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

                self.calculate_no_candidate_metric(
                    top_k_id=top_k_id.detach().cpu().numpy(),
                    liked_items=liked_items_by_batch_users,
                    top_k_values=top_k_values,
                )

                start += recommend_batch_size

        for k in top_k_values:
            # save map
            self.metric_at_k[k][Metric.MAP] = safe_divide(
                numerator=self.metric_at_k[k][Metric.MAP],
                denominator=self.metric_at_k[k][Metric.COUNT],
            )
            self.metric_at_k_total_epochs[k][Metric.MAP].append(
                self.metric_at_k[k][Metric.MAP]
            )

            # save ndcg
            self.metric_at_k[k][Metric.NDCG] = safe_divide(
                numerator=self.metric_at_k[k][Metric.NDCG],
                denominator=self.metric_at_k[k][Metric.COUNT],
            )
            self.metric_at_k_total_epochs[k][Metric.NDCG].append(
                self.metric_at_k[k][Metric.NDCG]
            )

            # save recall
            self.metric_at_k[k][Metric.RECALL] = safe_divide(
                numerator=self.metric_at_k[k][Metric.RECALL],
                denominator=self.metric_at_k[k][Metric.COUNT],
            )
            self.metric_at_k_total_epochs[k][Metric.RECALL].append(
                self.metric_at_k[k][Metric.RECALL]
            )

            # save count
            self.metric_at_k_total_epochs[k][Metric.COUNT] = self.metric_at_k[k][
                Metric.COUNT
            ]

    def calculate_no_candidate_metric(
        self,
        top_k_id: NDArray,
        liked_items: NDArray,
        top_k_values: List[int],
    ) -> None:
        """
        After calculating scores in `recommend_all` function, calculate metric without any candidates.
        Metrics calculated in this function are NDCG, mAP and recall.
        Note that this function does not consider locality, which means recommendations
        could be given regardless of user's location and diner's location

        Args:
             top_k_id (NDArray): Diner_id whose score is under max_k ranked score. (two dimensional array)
             liked_items (NDArray): Item ids liked by users. (two dimensional array)
             top_k_values (List[int]): A list of k values.
        """

        batch_num_users = liked_items.shape[0]

        for k in top_k_values:
            pred_liked_item_id = top_k_id[:, :k]
            metric = fully_vectorized_ranking_metrics_at_k(
                liked_items, pred_liked_item_id
            )
            self.metric_at_k[k][Metric.MAP] += metric[Metric.AP].sum()
            self.metric_at_k[k][Metric.NDCG] += metric[Metric.NDCG].sum()
            self.metric_at_k[k][Metric.RECALL] += metric[Metric.RECALL].sum()
            self.metric_at_k[k][Metric.COUNT] += batch_num_users
