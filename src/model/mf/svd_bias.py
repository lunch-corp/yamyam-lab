from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

from constant.metric.metric import Metric
from model.graph.base_embedding import BaseEmbedding


class Model(BaseEmbedding):
    def __init__(
        self,
        user_ids: torch.Tensor,
        diner_ids: torch.Tensor,
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
        super().__init__(
            user_ids=user_ids,
            diner_ids=diner_ids,
            top_k_values=top_k_values,
            embedding_dim=embedding_dim,
        )
        num_users = user_ids.size(0)
        num_items = diner_ids.size(0)

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

    def _generate_recommendations_and_calculate_metric_for_warm_start_users(
        self,
        X_train: Tensor,
        X_val_warm_users: Tensor,
        top_k_values: List[int],
        metric_at_k: Dict,
        filter_already_liked: bool = True,
    ) -> Dict:
        """
        Overwrite `_generate_recommendations_and_calculate_metric_for_warm_start_users` method in BaseEmbedding.

        Args:
             X_train (Tensor): number of reviews x (diner_id, reviewer_id) in train dataset.
             X_val (Tensor): number of reviews x (diner_id, reviewer_id) in val dataset.
             top_k_values (List[int]): a list of k values.
             filter_already_liked (bool): whether filtering pre-liked diner in train dataset or not.
        """
        max_k = max(top_k_values)
        start = 0
        diner_embeds = self.embed_item(self.diner_ids)

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
                user_embeds = self.embed_user(batch_users)
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
