import logging
from typing import Any, List

from numpy.typing import NDArray

from evaluation.metric_calculator.base_metric_calculator import BaseMetricCalculator
from model.mf.als import ALS


class ALSMetricCalculator(BaseMetricCalculator):
    def __init__(
        self,
        top_k_values: List[int],
        diner_ids: NDArray,
        model: ALS,
        filter_already_liked: bool = True,
        recommend_batch_size: int = 2000,
        device: str = "cpu",
        logger: logging.Logger = None,
    ) -> None:
        """
        MetricCalculator class for ALS model.

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
        train_csr = kwargs.get("train_csr")
        if train_csr is None:
            raise ValueError("Training csr matrix should be given.")
        num_diners = len(self.diner_ids)
        max_k = min(
            num_diners, max(self.top_k_values)
        )  # to prevent index error in pytest
        top_k_ids, top_k_values = self.model.recommend(
            user_ids=user_ids,
            train_csr=train_csr[user_ids],
            filter_already_liked=self.filter_already_liked,
            topk=max_k,
        )
        return top_k_ids  # shape: (len(user_ids), N)
