import logging
from typing import Any, List

import torch
from numpy.typing import NDArray
from torch.nn import Embedding

from yamyam_lab.evaluation.metric_calculator import BaseMetricCalculator
from yamyam_lab.model.mf.svd_bias import Model


class SVDBiasMetricCalculator(BaseMetricCalculator):
    def __init__(
        self,
        top_k_values: List[int],
        diner_ids: NDArray,
        model: Model,
        embed_user: Embedding,
        embed_item: Embedding,
        user_bias: Embedding,
        item_bias: Embedding,
        filter_already_liked: bool = True,
        recommend_batch_size: int = 2000,
        device: str = "cpu",
        logger: logging.Logger = None,
    ) -> None:
        """
        MetricCalculator class for svd_bias model.

        Args:
            top_k_values (List[int]): List of top k values to calculate metrics (ndcg@k, map@k, recall@k)
            diner_ids (NDArray): Numpy array containing all diner_ids.
            model (Any): Trained ALS object.
            filter_already_liked (bool): Whether filter already liked items in train data when generating recommendations.
            logger (logging.Logger): Logger to report metrics.
        """
        super().__init__(
            top_k_values=top_k_values,
            diner_ids=diner_ids,
            model=model,
            filter_already_liked=filter_already_liked,
            recommend_batch_size=recommend_batch_size,
            device=device,
            logger=logger,
        )
        self.embed_user = embed_user
        self.embed_item = embed_item
        self.user_bias = user_bias
        self.item_bias = item_bias

    def generate_recommendations(
        self,
        user_ids: NDArray,
        **kwargs: Any,
    ) -> NDArray:
        """
        Generate recommendations to users.

        Args:
            user_ids (NDArray): Batch user_ids.

        Returns (NDArray):
            Numpy array consisting of recommendation item_ids whose dimension is (len(user_ids), K)
        """
        user_ids = torch.tensor(user_ids, device=self.device)
        diner_ids = torch.tensor(sorted(self.diner_ids), device=self.device)
        num_diners = diner_ids.size(0)

        user_embedding = self.embed_user(user_ids)
        item_embedding = self.embed_item(diner_ids)
        user_bias = self.user_bias(user_ids)
        item_bias = self.item_bias(diner_ids)

        # we do not use `forward` method in svd_bias model, because it requires too much memory, resulting in oom
        # matmul: recommend_batch_size x num_diners
        # user_bias: recommend_batch_size x 1
        # item_bias: num_diners x 1
        # to broadcast item_bias rowwise, transpose it with (1 x num_diners) dimension
        scores = (
            torch.matmul(user_embedding, item_embedding.T) + user_bias + item_bias.T
        )

        # should be reshaped because it was broadcast above
        scores = scores.reshape(-1, num_diners)
        max_k = min(
            num_diners, max(self.top_k_values)
        )  # to prevent index error in pytest
        top_k = torch.topk(scores, k=max_k)
        top_k_id = top_k.indices
        return top_k_id.detach().cpu().numpy()
