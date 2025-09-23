import logging
from typing import Any, List

from numpy.typing import NDArray

from yamyam_lab.evaluation.metric_calculator.base_metric_calculator import (
    BaseMetricCalculator,
)


class MostPopularMetricCalculator(BaseMetricCalculator):
    def __init__(
        self,
        top_k_values: List[int],
        filter_already_liked: bool = True,
        recommend_batch_size: int = 2000,
        logger: logging.Logger = None,
    ) -> None:
        """
        MetricCalculator class for MostPopular model.
        Note that this class will be initialized just to use methods in `BaseMetricCalculator`.
        Therefore, `generate_recommendations` method will do nothing.
        Most popular item recommendations to warm users will be determined with `most_popular_rec_to_warm_users` argument in
        `generate_recommendations_and_calculate_metric` method in `BaseMetricCalculator` class.

        Args:
            top_k_values (List[int]): List of top k values to calculate metrics (ndcg@k, map@k, recall@k)
            filter_already_liked (bool): Whether filter already liked items in train data when generating recommendations.
            logger (logging.Logger): Logger to report metrics.
        """
        super().__init__(
            top_k_values=top_k_values,
            filter_already_liked=filter_already_liked,
            recommend_batch_size=recommend_batch_size,
            logger=logger,
        )

    def generate_recommendations(
        self,
        user_ids: NDArray,
        **kwargs: Any,
    ) -> NDArray:
        """
        This method will do nothing.
        """
        pass
