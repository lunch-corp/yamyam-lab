from abc import abstractmethod
from typing import List, Union, Tuple, Dict, Any
import torch
import numpy as np

from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from tools.utils import convert_tensor, get_user_locations
from evaluation.metric import ranking_metrics_at_k, ranked_precision

# set cpu or cuda for default option
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device.type)


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

    def recommend(
            self,
            X_train: Tensor,
            X_val: Tensor,
            nearby_candidates: Dict[int, list],
            top_K = [3, 5, 7, 10, 20],
            filter_already_liked=True
        ) -> Dict[int, Any]:
        user_embeds = self.embedding(self.user_ids)
        diner_embeds = self.embedding(self.diner_ids)
        scores = torch.mm(user_embeds, diner_embeds.t())

        self.map = 0.
        self.ndcg = 0.

        train_liked = convert_tensor(X_train, list)
        val_liked = convert_tensor(X_val, list)
        user_locations = get_user_locations(X_val)
        res = {}
        metric_at_K = {k: {"map": 0, "ndcg": 0, "count": 0, "ranked_prec": 0} for k in top_K}

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
            locations = user_locations[user_id]

            for K in top_K:
                if len(val_liked_item_id) < K:
                    continue
                score = scores[user_id - self.num_diners]
                pred_liked_item_id = torch.topk(score, k=K).indices.detach().cpu().numpy()
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

                # store recommendation result when K=20
                if K == 20:
                    res[user_id] = pred_liked_item_id
        for K in top_K:
            metric_at_K[K]["map"] /= metric_at_K[K]["count"]
            metric_at_K[K]["ndcg"] /= metric_at_K[K]["count"]
            metric_at_K[K]["ranked_prec"] /= X_val.shape[0]
        self.metric_at_K = metric_at_K
        return res
