import logging
from typing import Any, List

import torch
from numpy.typing import NDArray
from torch import Tensor

from evaluation.metric_calculator.base_metric_calculator import BaseMetricCalculator


class EmbeddingMetricCalculator(BaseMetricCalculator):
    def __init__(
        self,
        top_k_values: List[int],
        diner_ids: NDArray,
        all_embeds: Tensor = None,
        filter_already_liked: bool = True,
        recommend_batch_size: int = 2000,
        device: str = "cpu",
        logger: logging.Logger = None,
    ) -> None:
        """
        MetricCalculator class for embedding based model.

        Args:
            top_k_values (List[int]): List of top k values to calculate metrics (ndcg@k, map@k, recall@k)
            diner_ids (NDArray): Numpy array containing all diner_ids.
            all_embeds (Tensor): Trained embedding matrix.
            filter_already_liked (bool): Whether filter already liked items in train data when generating recommendations.
            recommend_batch_size (int): Batch size when generating recommendations.
            logger (logging.Logger): Logger to report metrics.
        """
        super().__init__(
            top_k_values=top_k_values,
            diner_ids=diner_ids,
            all_embeds=all_embeds,
            filter_already_liked=filter_already_liked,
            recommend_batch_size=recommend_batch_size,
            device=device,
            logger=logger,
        )

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
        user_embeds = self.all_embeds[user_ids]
        diner_embeds = self.all_embeds[self.diner_ids]
        scores = torch.mm(user_embeds, diner_embeds.t())

        # TODO: change for loop to more efficient program
        # filter diner id already liked by user in train dataset
        if self.filter_already_liked:
            for i, user_id in enumerate(user_ids):
                already_liked_ids = self.train_liked_series[user_id.item()]
                scores[i][already_liked_ids] = -float("inf")

        max_k = min(
            scores.shape[1], max(self.top_k_values)
        )  # to prevent index error in pytest
        top_k = torch.topk(scores, k=max_k)
        top_k_id = top_k.indices

        return top_k_id.detach().cpu().numpy()
