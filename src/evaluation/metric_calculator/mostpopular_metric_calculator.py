import numpy as np
from numpy.typing import NDArray

from evaluation.metric_calculator.base_metric_calculator import BaseMetricCalculator


class MostPopularMetricCalculator(BaseMetricCalculator):
    def __init__(self, top_k_values, most_popular_diner_ids, **kwargs):
        super().__init__(top_k_values=top_k_values, **kwargs)
        self.most_popular_diner_ids = np.asarray(most_popular_diner_ids, dtype=np.int64)

    def generate_recommendations(
        self,
        user_ids: NDArray,
        **kwargs,
    ) -> NDArray:
        max_k = max(self.top_k_values)
        base_list = self.most_popular_diner_ids

        do_filter = bool(getattr(self, "filter_already_liked", False)) and hasattr(
            self, "train_liked_series"
        )

        rows = []
        for uid in user_ids:
            arr = base_list
            if do_filter and uid in self.train_liked_series.index:
                liked = self.train_liked_series[uid]
                mask = ~np.isin(arr, liked)
                arr = arr[mask]

            if arr.size >= max_k:
                row = arr[:max_k]
            else:
                pad = np.full(max_k - arr.size, -1, dtype=np.int64)  # 더미 id
                row = np.concatenate([arr, pad], axis=0)

            rows.append(row)

        return np.vstack(rows)