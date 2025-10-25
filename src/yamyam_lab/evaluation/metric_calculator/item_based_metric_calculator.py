import logging
from typing import Any, List

from numpy.typing import NDArray

from evaluation.metric_calculator.base_metric_calculator import BaseMetricCalculator
from model.classic_cf.item_based import ItemBasedCollaborativeFiltering


class ItemBasedMetricCalculator(BaseMetricCalculator):
    def __init__(
        self,
        top_k_values: List[int],
        diner_ids: NDArray,
        model: ItemBasedCollaborativeFiltering,
        filter_already_liked: bool = True,
        recommend_batch_size: int = 2000,
        method: str = "cosine_matrix",
        use_hybrid: bool = False,
        cf_weight: float = 1.0,
        content_weight: float = 0.0,
        embedding_weight: float = 0.0,
        device: str = "cpu",
        logger: logging.Logger = None,
    ) -> None:
        """
        MetricCalculator class for Item-Based Collaborative Filtering model.

        Args:
            top_k_values (List[int]): List of top k values to calculate metrics (ndcg@k, map@k, recall@k)
            diner_ids (NDArray): Numpy array containing all diner_ids.
            model (ItemBasedCollaborativeFiltering): Trained Item-Based CF object.
            filter_already_liked (bool): Whether filter already liked items in train data when generating recommendations.
            recommend_batch_size (int): Batch size for recommendation generation.
            method (str): Similarity method ('cosine_matrix' or 'jaccard').
            use_hybrid (bool): Whether to use hybrid similarity (CF + Content + Embedding).
            cf_weight (float): Weight for CF similarity in hybrid mode.
            content_weight (float): Weight for content similarity in hybrid mode.
            embedding_weight (float): Weight for embedding similarity in hybrid mode.
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
        self.method = method
        self.use_hybrid = use_hybrid
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.embedding_weight = embedding_weight

    def generate_recommendations(
        self,
        user_ids: NDArray,
        **kwargs: Any,
    ) -> NDArray:
        """
        Generate recommendations to users using Item-Based CF (with optional hybrid mode).

        Args:
            user_ids (NDArray): Batch user_ids.

        Returns (NDArray):
            Numpy array consisting of recommendation item_ids whose dimension is (len(user_ids), K)
        """
        import numpy as np

        num_diners = len(self.diner_ids)
        max_k = min(
            num_diners, max(self.top_k_values)
        )  # to prevent index error in pytest

        # Generate recommendations for each user in the batch
        recommendations = []
        for user_id in user_ids:
            if self.use_hybrid:
                # Use hybrid recommendation
                recs = self.model.recommend_for_user_hybrid(
                    user_id=int(user_id),
                    top_k=max_k,
                    cf_weight=self.cf_weight,
                    content_weight=self.content_weight,
                    embedding_weight=self.embedding_weight,
                    method=self.method,
                )
            else:
                # Use standard CF recommendation
                recs = self.model.recommend_for_user(
                    user_id=int(user_id),
                    top_k=max_k,
                    method=self.method,
                )

            # Extract item_ids from recommendations
            if len(recs) > 0:
                rec_items = [rec["item_id"] for rec in recs]
            else:
                # If no recommendations, return empty array (will be filled later)
                rec_items = []

            # Pad with -1 if necessary to ensure all have same length
            while len(rec_items) < max_k:
                rec_items.append(-1)

            recommendations.append(rec_items[:max_k])

        return np.array(recommendations)  # shape: (len(user_ids), max_k)
