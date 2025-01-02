import os
from abc import ABC, abstractmethod

import pandas as pd
import torch
from numpy.typing import NDArray
from prettytable import PrettyTable
from torch import Tensor

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data")


class BaseQualitativeEvaluation(ABC):
    def __init__(
        self,
        user_mapping: dict[int, int],
        diner_mapping: dict[int, int],
    ):
        """
        Base class for qualitative evaluation.
        Be cautious of id mapping. When preprocessing before model training,
        all user_ids and diner_ids are 1-1 mapped for efficient searching.
        For example, user_id 1742183 is mapped to 0.
        When qualitative evaluation, raw user_id (e.g., 1742183) is given.
        Therefore, re-mapping logic is required for proper evaluation

        Args:
             user_mapping (Dict[int, int]): user mapping dictionary in preprocessing step.
             diner_mapping (Dict[int, int]): diner mapping dictionary in preprocessing step.
        """
        self.diners = pd.read_csv(os.path.join(DATA_PATH, "diner/diner_df_20241219_yamyam.csv"))
        self.user_mapping = user_mapping
        # reverse mapping to original diner_id
        self.diner_mapping = {v: k for k, v in diner_mapping.items()}

    @abstractmethod
    def _recommend(
        self,
        user_id: Tensor,
        tr_liked_diners: list[int],
        top_k: int = 10,
    ) -> tuple[NDArray, NDArray]:
        """
        Abstract method for individual recommendation
        """
        raise NotImplementedError

    def recommend(
        self,
        user_id: int,
        tr_liked_diners: list[int],
        val_liked_diners: list[int],
        top_k: int = 10,
    ) -> PrettyTable:
        """
        Recommend top_k ranked diners to user_id.
        Exclude diners that are already liked by user_id in training dataset.
        Include various information, such as diner name or whether it is actually hitted.

        Args:
             user_id (int): target user_id to recommend before mapping.
             tr_liked_diners (List[int]): list of diners liked by user_id in training dataset.
             val_liked_diners (List[int]): list of diners liked by user_id validation dataset.

        Returns (PrettyTable):
            PrettyTable object.
        """
        user_id_mapping = self.user_mapping.get(user_id, None)
        if user_id_mapping is None:
            raise ValueError(f"No mapping for user {user_id}")
        pred_diner_id, pred_diner_score = self._recommend(
            user_id=torch.tensor([user_id_mapping]),
            tr_liked_diners=tr_liked_diners,
            top_k=top_k,
        )
        pred_diner_id_mapping = [self.diner_mapping[diner] for diner in pred_diner_id]

        tb = PrettyTable(field_names=["diner_name", "diner_category_small", "url", "score", "hitted"])
        for diner_idx, score in zip(pred_diner_id_mapping, pred_diner_score):
            info = self.diners[lambda x: x["diner_idx"] == diner_idx].iloc[0]
            tb.add_row(
                [
                    info["diner_name"],
                    info["diner_category_small"],
                    info["diner_url"],
                    score,
                    1 if diner_idx in val_liked_diners else 0,
                ]
            )
        return tb
