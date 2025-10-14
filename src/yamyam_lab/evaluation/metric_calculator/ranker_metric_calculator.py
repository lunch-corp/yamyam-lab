import logging
from typing import Any, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from yamyam_lab.evaluation.metric_calculator import BaseMetricCalculator


class RankerMetricCalculator(BaseMetricCalculator):
    def __init__(
        self,
        top_k_values: List[int],
        model: Any,  # LightGBMTrainer 등 ranker 모델
        features: List[str],
        recommend_batch_size: int = 2000,
        filter_already_liked: bool = True,
        logger: logging.Logger = None,
    ) -> None:
        super().__init__(
            top_k_values=top_k_values,
            model=model,
            filter_already_liked=filter_already_liked,
            recommend_batch_size=recommend_batch_size,
            logger=logger,
        )
        self.features = features

    def generate_recommendations(
        self,
        user_ids: NDArray,
        candidates: pd.DataFrame,
        train_liked_series: pd.Series = None,
        **kwargs: Any,
    ) -> NDArray:
        """
        Args:
            user_ids: recommended user_ids
            candidates: candidates to recommend
            train_liked_series: already liked items of train data
            kwargs: additional arguments

        Returns:
            recommended items
        """
        max_k = max(self.top_k_values)
        user_group = candidates.groupby("reviewer_id")

        reco_items = []
        for user_id in user_ids:
            if user_id not in user_group.groups:
                # 사용자에 대한 후보가 없는 경우 -1로 패딩
                reco_items.append(np.full(max_k, -1))
                continue

            user_candidates = user_group.get_group(user_id)["diner_idx"].values

            # 이미 좋아한 아이템 필터링
            if (
                self.filter_already_liked
                and train_liked_series is not None
                and user_id in train_liked_series
            ):
                already_liked = train_liked_series[user_id]
                user_candidates = user_candidates[
                    ~np.isin(user_candidates, already_liked)
                ]

            # 후보가 max_k보다 적은 경우 -1로 패딩
            if len(user_candidates) < max_k:
                padded_candidates = np.full(max_k, -1)
                padded_candidates[: len(user_candidates)] = user_candidates
                reco_items.append(padded_candidates)
            else:
                # top-K만 추출
                reco_items.append(user_candidates[:max_k])

        return np.vstack(reco_items)
