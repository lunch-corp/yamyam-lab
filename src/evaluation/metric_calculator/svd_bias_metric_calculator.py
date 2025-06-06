import logging
from typing import Any, List

import torch
from numpy.typing import NDArray

from evaluation.metric_calculator import BaseMetricCalculator
from model.mf.svd_bias import Model


class SVDBiasMetricCalculator(BaseMetricCalculator):
    def __init__(
        self,
        top_k_values: List[int],
        diner_ids: NDArray,
        model: Model,
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
        num_batch_users = user_ids.size(0)

        # repeat user_ids and diner_ids to compute scores btw users and all of diners
        user_ids = user_ids.repeat_interleave(num_diners)
        diner_ids = diner_ids.tile(num_batch_users)

        # inference mode
        with torch.no_grad():
            scores = self.model(
                user_idx=user_ids,
                item_idx=diner_ids,
            )

        # should be reshaped because it was broadcast above
        scores = scores.reshape(-1, num_diners)
        max_k = min(
            num_diners, max(self.top_k_values)
        )  # to prevent index error in pytest
        top_k = torch.topk(scores, k=max_k)
        top_k_id = top_k.indices
        return top_k_id.detach().cpu().numpy()
