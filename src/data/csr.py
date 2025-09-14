from typing import Any, Dict, Self

import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from data.base import BaseDatasetLoader
from data.config import DataConfig


class CsrDatasetLoader(BaseDatasetLoader):
    def __init__(self: Self, data_config: DataConfig):
        super().__init__(data_config)

    def prepare_csr_dataset(
        self: Self,
        filter_config: Dict[str, Any] = None,
        is_rank: bool = True,
        is_csr: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Load and process training data for ranking models.

        Args:
            filter_config (Dict[str, Any]): Filter config used when filtering reviews.
            is_rank (bool): Indicator if it is ranking model or not. Defaults to True.
            is_csr (bool): Indicator if csr format or not for als model.
            **kwargs: Additional keyword arguments

        Returns (Dict[str, Any]):
            A dictionary containing the training and validation sets.
        """
        prepared_data = self.prepare_train_val_dataset(
            filter_config=filter_config,
            is_rank=is_rank,
            is_csr=is_csr,
            **kwargs,
        )

        train = prepared_data["train"]
        val = prepared_data["val"]
        test = prepared_data["test"]
        mapped_res = prepared_data["mapped_res"]

        val_warm_start_user_ids, val_cold_start_user_ids = (
            self.get_warm_cold_start_user_ids(
                train_review=train,
                test_review=val,
            )
        )
        test_warm_start_user_ids, test_cold_start_user_ids = (
            self.get_warm_cold_start_user_ids(
                train_review=train,
                test_review=test,
            )
        )
        train_user_ids = train["reviewer_id"].unique()
        val_user_ids = val["reviewer_id"].unique()
        test_user_ids = test["reviewer_id"].unique()
        train_diner_ids = train["diner_idx"].unique()
        val_diner_ids = val["diner_idx"].unique()
        test_diner_ids = test["diner_idx"].unique()
        val_warm_users = val[lambda x: x["reviewer_id"].isin(val_warm_start_user_ids)]
        val_cold_users = val[lambda x: x["reviewer_id"].isin(val_cold_start_user_ids)]
        test_warm_users = test[
            lambda x: x["reviewer_id"].isin(test_warm_start_user_ids)
        ]
        test_cold_users = test[
            lambda x: x["reviewer_id"].isin(test_cold_start_user_ids)
        ]

        return self.create_csr_dataset(
            train=train,
            val=val,
            test=test,
            train_user_ids=train_user_ids,
            val_user_ids=val_user_ids,
            test_user_ids=test_user_ids,
            train_diner_ids=train_diner_ids,
            val_diner_ids=val_diner_ids,
            test_diner_ids=test_diner_ids,
            val_warm_start_user_ids=val_warm_start_user_ids,
            val_cold_start_user_ids=val_cold_start_user_ids,
            test_warm_start_user_ids=test_warm_start_user_ids,
            test_cold_start_user_ids=test_cold_start_user_ids,
            val_warm_users=val_warm_users,
            val_cold_users=val_cold_users,
            test_warm_users=test_warm_users,
            test_cold_users=test_cold_users,
            mapped_res=mapped_res,
        )

    def create_csr_matrix_from_review_data(
        self: Self,
        df: pd.DataFrame,
        num_total_users: int,
        num_total_diners: int,
    ) -> csr_matrix:
        """
        Create csr matrix from review data for als model.
        """
        Cui_csr = csr_matrix(
            (df["reviewer_review_score"], (df["reviewer_id"], df["diner_idx"])),
            shape=(num_total_users, num_total_diners),
        )
        return Cui_csr

    def create_csr_dataset(
        self: Self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        train_user_ids: NDArray,
        val_user_ids: NDArray,
        test_user_ids: NDArray,
        train_diner_ids: NDArray,
        val_diner_ids: NDArray,
        test_diner_ids: NDArray,
        val_warm_start_user_ids: NDArray,
        val_cold_start_user_ids: NDArray,
        test_warm_start_user_ids: NDArray,
        test_cold_start_user_ids: NDArray,
        val_warm_users: pd.DataFrame,
        val_cold_users: pd.DataFrame,
        test_warm_users: pd.DataFrame,
        test_cold_users: pd.DataFrame,
        mapped_res: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create train / val csr matrix and return other data used in evaluation.
        """
        num_total_users = len(mapped_res.get("user_mapping"))
        num_total_diners = len(mapped_res.get("diner_mapping"))
        train_Cui_csr = self.create_csr_matrix_from_review_data(
            df=train,
            num_total_users=num_total_users,
            num_total_diners=num_total_diners,
        )
        val_Cui_csr = self.create_csr_matrix_from_review_data(
            df=val,
            num_total_users=num_total_users,
            num_total_diners=num_total_diners,
        )
        test_Cui_csr = self.create_csr_matrix_from_review_data(
            df=test,
            num_total_users=num_total_users,
            num_total_diners=num_total_diners,
        )
        return {
            "X_train": train_Cui_csr,
            "X_val": val_Cui_csr,
            "X_test": test_Cui_csr,
            "X_train_df": train,
            "X_val_warm_users": val_warm_users[self.X_columns],
            "y_val_warm_users": val_warm_users[self.y_columns],
            "X_val_cold_users": val_cold_users[self.X_columns],
            "y_val_cold_users": val_cold_users[self.y_columns],
            "X_test_warm_users": test_warm_users[self.X_columns],
            "y_test_warm_users": test_warm_users[self.y_columns],
            "X_test_cold_users": test_cold_users[self.X_columns],
            "y_test_cold_users": test_cold_users[self.y_columns],
            "most_popular_diner_ids": self.get_most_popular_diner_ids(
                train_review=train
            ),
            "val_warm_start_user_ids": val_warm_start_user_ids,
            "val_cold_start_user_ids": val_cold_start_user_ids,
            "test_warm_start_user_ids": test_warm_start_user_ids,
            "test_cold_start_user_ids": test_cold_start_user_ids,
            "train_user_ids": train_user_ids,
            "val_user_ids": val_user_ids,
            "test_user_ids": test_user_ids,
            "train_diner_ids": train_diner_ids,
            "val_diner_ids": val_diner_ids,
            "test_diner_ids": test_diner_ids,
            **mapped_res,
        }
