from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from yamyam_lab.constant.metric.metric import Metric


class Model(nn.Module):
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
        super().__init__()
        self.user_ids = user_ids
        self.diner_ids = diner_ids
        self.top_k_values = top_k_values
        self.embedding_dim = embedding_dim

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
