import logging
from abc import abstractmethod
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch import Tensor

from constant.metric.metric import Metric
from evaluation.metric import fully_vectorized_ranking_metrics_at_k
from tools.utils import safe_divide


class BaseMetricCalculator:
    def __init__(
        self,
        top_k_values: List[int],
        diner_ids: NDArray = None,
        model: Any = None,
        all_embeds: Tensor = None,
        filter_already_liked: bool = True,
        recommend_batch_size: int = 2000,
        device: str = "cpu",
        logger: logging.Logger = None,
    ) -> None:
        """
        Base parent class when initializing MetricCalculator class for each model.
        MetricCalculator class is used when calculating metric for validation data or test data.
        For unified pipeline when calculating metric, it is recommended that this base class is inherited
        when initializing custom MetricCalculator class for each model.
        Note that the init params are required depending on the model.

        Args:
            top_k_values (List[int]): List of top k values to calculate metrics (ndcg@k, map@k, recall@k)
            diner_ids (NDArray): Numpy array containing all diner_ids.
            model (Any): Trained model object, which is used in ALSMetricCalculator.
            all_embeds (Tensor): Trained embeddings, which is used in graph based model.
            filter_already_liked (bool): Whether filter already liked items in train data when generating recommendations.
            device (str): Device type, either of cpu or gpu.
            logger (logging.Logger): Logger to report metrics.
        """
        self.top_k_values = top_k_values
        self.diner_ids = diner_ids
        self.model = model
        self.all_embeds = all_embeds
        self.filter_already_liked = filter_already_liked
        self.recommend_batch_size = recommend_batch_size
        self.device = device
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    @abstractmethod
    def generate_recommendations(self, user_ids: NDArray, **kwargs: Any) -> NDArray:
        """
        Abstract method for each recommendation model.
        This method is used when generating recommendations to each user.
        For example, when embedding based model, dot product is implemented in this method, or
        when ranker model, `model.predict` is implemented.

        Args:
            user_ids (NDArray): List of user_ids to generate recommendations, which is usually greater than one.

        Returns (NDArray):
            2 dimensional numpy array whose dimension is num_users * max_k.
        """
        raise NotImplementedError("Individual predict method should be implemented.")

    def generate_recommendations_and_calculate_metric(
        self,
        X_train: pd.DataFrame,
        X_val_warm_users: pd.DataFrame,
        X_val_cold_users: pd.DataFrame,
        most_popular_diner_ids: List[int],
        **kwargs: Any,
    ) -> Dict[str, Dict]:
        """
        Generate recommendations for warm / cold start users separately and calculate metric.
        For warm start users, `generate_recommendations` method will be used to generate recommendations.
        For cold start users, most popular items from train data will be used.

        Args:
            X_train (pd.DataFrame): Train dataset in pandas dataframe.
            X_val_warm_users (pd.DataFrame): Validation or test dataset in pandas dataframe which consists only with warm start users.
            X_val_cold_users (pd.DataFrame): Validation or test dataset in pandas dataframe which consists only with cold start users.
            most_popular_diner_ids (List[int]): List of most popular diner_ids from train dataset.

        Returns (Dict[str, Dict]):
            Dictionary including calculated metric for warm start users, cold start users and all of users.
        """
        # calculate metric for warm/cold users separately
        metric_at_k_warm_users = (
            self._generate_recommendations_and_calculate_metric_for_warm_start_users(
                X_train=X_train,
                X_val_warm_users=X_val_warm_users,
                **kwargs,
            )
        )
        metric_at_k_cold_users = (
            self._generate_recommendations_and_calculate_metric_for_cold_start_users(
                X_val_cold_users=X_val_cold_users,
                most_popular_diner_ids=most_popular_diner_ids,
            )
        )

        # aggregate metric results from warm/cold users
        metric_at_k_total = {
            k: {
                Metric.MAP: 0,
                Metric.NDCG: 0,
                Metric.RECALL: 0,
                Metric.COUNT: 0,
            }
            for k in self.top_k_values
        }
        for k in self.top_k_values:
            for metric in Metric:
                # metric is stored in map not ap, so pass it
                if metric.value == "ap":
                    continue
                metric_at_k_total[k][metric.value] = (
                    metric_at_k_warm_users[k][metric.value]
                    + metric_at_k_cold_users[k][metric.value]
                )

        # check whether all users from val are included
        num_val_users = len(X_val_warm_users["reviewer_id"].unique()) + len(
            X_val_cold_users["reviewer_id"].unique()
        )
        for k in self.top_k_values:
            metric_res = metric_at_k_total[k]
            if metric_res[Metric.COUNT] != num_val_users:
                raise ValueError(
                    "Number users whose metrics are calculated should be equal to total number users in val"
                )

        return {
            "warm": metric_at_k_warm_users,
            "cold": metric_at_k_cold_users,
            "all": metric_at_k_total,
        }

    def _generate_recommendations_and_calculate_metric_for_warm_start_users(
        self,
        X_train: pd.DataFrame,
        X_val_warm_users: pd.DataFrame,
        **kwargs: Any,
    ) -> Dict:
        """
        Generate recommendations for warm start users separately and calculate metric.
        For warm start users, `generate_recommendations` method will be used to generate recommendations.

        Args:
            X_train (pd.DataFrame): Train dataset in pandas dataframe.
            X_val_warm_users (pd.DataFrame): Validation or test dataset in pandas dataframe which consists only with warm start users.

        Returns (Dict):
            Dictionary with calculated metric for warm start users.
        """
        metric_at_k = {
            k: {
                Metric.MAP: 0,
                Metric.NDCG: 0,
                Metric.RECALL: 0,
                Metric.COUNT: 0,
            }
            for k in self.top_k_values
        }

        start = 0

        self.train_liked_series = X_train.groupby("reviewer_id")["diner_idx"].apply(
            np.array
        )
        self.train_liked = (
            self.train_liked_series.to_frame()
            .reset_index()
            .assign(num_liked_items=lambda df: df["diner_idx"].apply(len))
        )
        self.val_liked_series = X_val_warm_users.groupby("reviewer_id")[
            "diner_idx"
        ].apply(np.array)
        self.val_liked = (
            self.val_liked_series.to_frame()
            .reset_index()
            .assign(num_liked_items=lambda df: df["diner_idx"].apply(len))
            .sort_values(by="reviewer_id")
        )

        liked_items_count2user_ids = (
            self.val_liked.groupby("num_liked_items")["reviewer_id"]
            .apply(np.array)
            .to_dict()
        )

        for count, user_ids in liked_items_count2user_ids.items():
            for start in range(0, len(user_ids), self.recommend_batch_size):
                batch_users = user_ids[start : start + self.recommend_batch_size]

                top_k_id = self.generate_recommendations(
                    user_ids=batch_users,
                    **kwargs,
                )

                # batch_users_np = batch_users.detach().cpu().numpy()
                liked_items_by_batch_users = np.vstack(
                    self.val_liked[lambda x: x["reviewer_id"].isin(batch_users)][
                        "diner_idx"
                    ].values
                )

                self.calculate_metric_at_current_batch(
                    metric_at_k=metric_at_k,
                    top_k_id=top_k_id,
                    liked_items=liked_items_by_batch_users,
                )

        num_warm_start_users = len(X_val_warm_users["reviewer_id"].unique())
        for k in self.top_k_values:
            metric_res = metric_at_k[k]
            if metric_res[Metric.COUNT] != num_warm_start_users:
                raise ValueError(
                    "Number users whose metrics are calculated should be equal to number of warm start users"
                )
        return metric_at_k

    def _generate_recommendations_and_calculate_metric_for_cold_start_users(
        self,
        X_val_cold_users: pd.DataFrame,
        most_popular_diner_ids: List[int],
    ) -> Dict:
        """
        Generate recommendations for cold start users separately and calculate metric.
        For cold start users, most popular items from train data will be used.

        Args:
            X_val_cold_users (pd.DataFrame): Validation or test dataset in pandas dataframe which consists only with cold start users.
            most_popular_diner_ids (List[int]): List of most popular diner_ids from train dataset.

        Returns (Dict):
            Dictionary with calculated metric for cold start users.
        """
        metric_at_k = {
            k: {
                Metric.MAP: 0,
                Metric.NDCG: 0,
                Metric.RECALL: 0,
                Metric.COUNT: 0,
            }
            for k in self.top_k_values
        }

        if X_val_cold_users.shape[0] == 0:
            return metric_at_k
        val_liked_cold_users = (
            X_val_cold_users.groupby("reviewer_id")["diner_idx"]
            .apply(np.array)
            .to_frame()
            .reset_index()
            .assign(num_liked_items=lambda df: df["diner_idx"].apply(len))
            .sort_values(by="reviewer_id")
        )
        liked_items_count2user_ids = (
            val_liked_cold_users.groupby("num_liked_items")["reviewer_id"]
            .apply(np.array)
            .to_dict()
        )

        for count, user_ids in liked_items_count2user_ids.items():
            for start in range(0, len(user_ids), self.recommend_batch_size):
                batch_users = user_ids[start : start + self.recommend_batch_size]
                most_popular_reco_items = np.tile(
                    most_popular_diner_ids, (len(batch_users), 1)
                )

                liked_items_by_batch_users = np.vstack(
                    val_liked_cold_users[lambda x: x["reviewer_id"].isin(batch_users)][
                        "diner_idx"
                    ].values
                )

                self.calculate_metric_at_current_batch(
                    metric_at_k=metric_at_k,
                    top_k_id=most_popular_reco_items,
                    liked_items=liked_items_by_batch_users,
                )

        num_cold_start_users = len(X_val_cold_users["reviewer_id"].unique())
        for k in self.top_k_values:
            metric_res = metric_at_k[k]
            if metric_res[Metric.COUNT] != num_cold_start_users:
                raise ValueError(
                    "Number users whose metrics are calculated should be equal to number of cold start users"
                )
        return metric_at_k

    def calculate_metric_at_current_batch(
        self,
        metric_at_k: Dict,
        top_k_id: NDArray,
        liked_items: NDArray,
    ) -> None:
        """
        For each batch when generating recommendations in warm / cold start users, calculate metric using vectorized function.
        Note that dimension of top_k_id (num_users_in_batch, K) and dimension of liked_items is (num_users_in_batch, num_liked_items).
        Because upper loop is run with users who have identical number of liked items, this vectorized calculation is possible.
        Note that this function updates metric in `metric_at_k` **in place**, therefore does not return anything.

        Args:
            metric_at_k (Dict): Dictionary to store metric result.
            top_k_id (NDArray): Diner_id whose score is under max_k ranked score. (two dimensional array)
            liked_items (NDArray): Item ids liked by users. (two dimensional array)
        """

        batch_num_users = liked_items.shape[0]

        for k in self.top_k_values:
            pred_liked_item_id = top_k_id[:, :k]
            metric = fully_vectorized_ranking_metrics_at_k(
                liked_items, pred_liked_item_id
            )
            metric_at_k[k][Metric.MAP] += metric[Metric.AP].sum()
            metric_at_k[k][Metric.NDCG] += metric[Metric.NDCG].sum()
            metric_at_k[k][Metric.RECALL] += metric[Metric.RECALL].sum()
            metric_at_k[k][Metric.COUNT] += batch_num_users

    def calculate_mean_metric(self, metric_at_k: Dict) -> None:
        """
        Averages metric over all of users and updates **in place**, therefore does not return anything.

        Args:
            metric_at_k (Dict): Calculated metric before averaging.
        """
        for k in self.top_k_values:
            # average of ap
            metric_at_k[k][Metric.MAP] = safe_divide(
                numerator=metric_at_k[k][Metric.MAP],
                denominator=metric_at_k[k][Metric.COUNT],
            )
            # average of ndcg
            metric_at_k[k][Metric.NDCG] = safe_divide(
                numerator=metric_at_k[k][Metric.NDCG],
                denominator=metric_at_k[k][Metric.COUNT],
            )
            # average of recall
            metric_at_k[k][Metric.RECALL] = safe_divide(
                numerator=metric_at_k[k][Metric.RECALL],
                denominator=metric_at_k[k][Metric.COUNT],
            )

    def save_metric_at_current_epoch(
        self,
        metric_at_k: Dict,
        metric_at_k_total_epochs: Dict,
    ) -> None:
        """
        Save metric at current epoch **in place**, therefore does not return anything.

        Args:
            metric_at_k (Dict): Dictionary containing metrics at current epoch
            metric_at_k_total_epochs (Dict): Dictionary to store metric result, containing metrics from all previous epochs.
        """
        for k in self.top_k_values:
            # save map
            metric_at_k_total_epochs[k][Metric.MAP].append(metric_at_k[k][Metric.MAP])
            # save ndcg
            metric_at_k_total_epochs[k][Metric.NDCG].append(metric_at_k[k][Metric.NDCG])
            # save recall
            metric_at_k_total_epochs[k][Metric.RECALL].append(
                metric_at_k[k][Metric.RECALL]
            )
            # save count
            metric_at_k_total_epochs[k][Metric.COUNT] = metric_at_k[k][Metric.COUNT]

    def report_metric_with_warm_cold_all_users(
        self, metric_dict: Dict, data_type: str = "val"
    ) -> None:
        """
        Report and log calculated metric for warm / cold / all of users.
        Metric could be calculated from validation or test dataset.
        For prediction task, map and ndcg are reported.
        For candidate generation task, recall is reported.

        Args:
            metric_dict (Dict): Dictionary containing metrics from warm / cold / all of users
            data_type (str): Data type which is validation or test dataset.
        """
        for user_type, metric in metric_dict.items():
            maps = []
            ndcgs = []
            recalls = []

            self.logger.info(
                f"[ {user_type} users metric report calculated from {data_type} data ]"
            )

            # map, ndcg metric
            for k in [i for i in self.top_k_values if i <= 20]:
                map = round(metric[k][Metric.MAP], 5)
                ndcg = round(metric[k][Metric.NDCG], 5)
                maps.append(str(map))
                ndcgs.append(str(ndcg))

            self.logger.info(
                "top k results for direct prediction @3, @7, @10, @20 in order"
            )
            self.logger.info(f"map result: {'|'.join(maps)}")
            self.logger.info(f"ndcg result: {'|'.join(ndcgs)}")

            # recall metric
            for k in [i for i in self.top_k_values if i >= 100]:
                recall = round(metric[k][Metric.RECALL], 5)
                recalls.append(str(recall))

            if len(recalls) == 0:
                continue

            self.logger.info(
                "top k results for candidate generation @100, @300, @500, @1000, @2000"
            )
            self.logger.info(f"recall: {'|'.join(recalls)}")
