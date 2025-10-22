from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from evaluation.metric_calculator.base_metric_calculator import BaseMetricCalculator


class ItemBasedMetricCalculator(BaseMetricCalculator):
    """
    Metric calculator for Item-based Collaborative Filtering models.
    DataFrame 기반, warm/cold 제거, 단일 train/test 구조 사용
    """

    def __init__(
        self,
        model: Any,
        test_data: Dict[str, Any],
        top_k_values: list[int],
        recommend_batch_size: int = 2000,
        logger=None,
    ) -> None:
        super().__init__(
            top_k_values=top_k_values,
            model=model,
            recommend_batch_size=recommend_batch_size,
            logger=logger,
        )

        self.X_train = test_data["train"]  # DataFrame
        self.X_test = test_data["test"]  # DataFrame

        # warm/cold 제거
        self.X_val_warm_users = self.X_test
        self.X_val_cold_users = self.X_test.iloc[0:0]  # 빈 DataFrame

    # ---------------- 추천 생성 ----------------
    def generate_recommendations(
        self, user_ids: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        top_k = max(self.top_k_values)
        top_k_recs = []

        for user_id in tqdm(user_ids, leave=False):
            rec_items = self.model.recommend_for_user(user_id=user_id, top_k=top_k)
            top_k_recs.append(rec_items)

        return np.array(top_k_recs)

    # ---------------- 평가 ----------------
    def evaluate(self) -> Dict[str, float]:
        """
        단일 test set만 사용하여 metric 계산
        반환값: {metric_name: score}
        """
        self.logger.info(
            "Generating recommendations and calculating metrics for item-based CF model..."
        )

        most_popular_diner_ids = (
            self.X_train["diner_idx"]
            .value_counts()
            .index[: max(self.top_k_values)]
            .to_list()
        )

        metric_dict = self.generate_recommendations_and_calculate_metric(
            X_train=self.X_train,
            X_val_warm_users=self.X_test,
            X_val_cold_users=self.X_test.iloc[0:0],
            most_popular_diner_ids=most_popular_diner_ids,
        )

        mean_metrics = {}
        # top_k 존재 확인 후 평균 metric 계산
        for k in self.top_k_values:
            if k in metric_dict["all"]:
                for metric_name, score_list in metric_dict["all"][k].items():
                    mean_metrics[metric_name] = np.mean(score_list)
            else:
                self.logger.warning(
                    f"top_k={k} not found in metric_dict['all']. Skipping."
                )

        return mean_metrics
