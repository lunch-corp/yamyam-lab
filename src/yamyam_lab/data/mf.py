from typing import Any, Dict, Self

import pandas as pd
import torch
from numpy.typing import NDArray

from yamyam_lab.data.base import BaseDatasetLoader
from yamyam_lab.data.config import DataConfig
from yamyam_lab.preprocess.preprocess import meta_mapping


class MFDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader specifically for matrix factorization models (MF, BPR, etc.)
    """

    def __init__(self: Self, data_config: DataConfig):
        super().__init__(data_config)

    def prepare_mf_dataset(
        self: Self,
        filter_config: Dict[str, Any] = None,
        is_networkx_graph: bool = False,
        is_tensor: bool = False,
        use_metadata: bool = False,
        weighted_edge: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Load and process training data for graph models.

        Args:
            filter_config (Dict[str, Any]): Filter config used when filtering reviews.
            is_networkx_graph (bool): Indicator if using networkx graph object or not.
            is_tensor (bool): Indicator if using tensor or not.
            use_metadata (bool): Indicator if using metadata or not in graph model, especially for metapath2vec
            weighted_edge (bool): Indicator if using weighted edge or not.
            **kwargs: Additional keyword arguments

        Returns (Dict[str, Any]):
            A dictionary containing the training and validation sets.
        """
        prepared_data = self.prepare_train_val_dataset(
            filter_config=filter_config,
            is_networkx_graph=is_networkx_graph,
            is_tensor=is_tensor,
            use_metadata=use_metadata,
            weighted_edge=weighted_edge,
        )
        train = prepared_data["train"]
        val = prepared_data["val"]
        test = prepared_data["test"]
        diner_meta_feature = prepared_data["diner_meta_feature"]
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

        if use_metadata:
            meta_mapping_info = meta_mapping(
                diner=diner_meta_feature,
                num_users=mapped_res["num_users"],
                num_diners=mapped_res["num_diners"],
            )
            mapped_res.update(meta_mapping_info)

        return self.create_torch_dataset(
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
            diner_meta_feature=diner_meta_feature,
            mapped_res=mapped_res,
        )

    def create_torch_dataset(
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
        diner_meta_feature: pd.DataFrame,
        mapped_res: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create torch dataloader and return other data used in evaluation.
        """
        from yamyam_lab.preprocess.preprocess import prepare_torch_dataloader

        X_train = torch.tensor(train[self.X_columns].values)
        y_train = torch.tensor(train[self.y_columns].values, dtype=torch.float32)
        X_val = torch.tensor(val[self.X_columns].values)
        y_val = torch.tensor(val[self.y_columns].values, dtype=torch.float32)

        train_dataloader, val_dataloader = prepare_torch_dataloader(
            X_train, y_train, X_val, y_val
        )

        return {
            "X_train": train[self.X_columns],
            "y_train": train[self.y_columns],
            "X_val": val[self.X_columns],
            "y_val": val[self.y_columns],
            "X_test": test[self.X_columns],
            "y_test": test[self.y_columns],
            "X_val_warm_users": val_warm_users[self.X_columns],
            "y_val_warm_users": val_warm_users[self.y_columns],
            "X_val_cold_users": val_cold_users[self.X_columns],
            "y_val_cold_users": val_cold_users[self.y_columns],
            "X_test_warm_users": test_warm_users[self.X_columns],
            "y_test_warm_users": test_warm_users[self.y_columns],
            "X_test_cold_users": test_cold_users[self.X_columns],
            "y_test_cold_users": test_cold_users[self.y_columns],
            "diner": diner_meta_feature,
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
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            **mapped_res,
        }
