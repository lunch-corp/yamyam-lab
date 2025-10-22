import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import pandas as pd
import torch
from numpy.typing import NDArray
from preprocess.diner_transform import CategoryProcessor
from prettytable import PrettyTable
from torch import Tensor

from candidate.near import NearCandidateGenerator

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data")


class BaseQualitativeEvaluation(ABC):
    def __init__(
        self,
        user_mapping: Dict[int, int],
        diner_mapping: Dict[int, int],
        latitude: float = None,
        longitude: float = None,
        near_dist: float = 0.5,
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
        self.diners = pd.read_csv(os.path.join(DATA_PATH, "diner.csv"))
        # merge category
        diner_category = pd.read_csv(os.path.join(DATA_PATH, "diner_category_raw.csv"))
        processor = CategoryProcessor(diner_category)
        processor.process_all()
        self.diners = pd.merge(
            self.diners,
            processor.category_preprocessed_diners,
            how="left",
            on="diner_idx",
        )
        self.user_mapping = user_mapping
        # reverse mapping to original diner_id
        self.reverse_diner_mapping = {v: k for k, v in diner_mapping.items()}

        if latitude is not None and longitude is not None:
            near = NearCandidateGenerator()
            near_diner_ids = near.get_near_candidate(
                latitude=latitude,
                longitude=longitude,
                max_distance_km=near_dist,
                is_radians=False,
            )
            self.near_diner_ids = [
                diner_mapping[id_]
                for id_ in near_diner_ids
                if diner_mapping.get(id_) is not None
            ]
        else:
            self.near_diner_ids = None

    @abstractmethod
    def _recommend(
        self,
        user_id: Tensor,
        tr_liked_diners: List[int],
        top_k: int = 10,
        near_diner_ids: List[int] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        Abstract method for individual recommendation
        """
        raise NotImplementedError

    def recommend(
        self,
        user_id_mapping: int,
        tr_liked_diners: List[int],
        test_liked_diners: List[int] = None,
        top_k: int = 10,
    ) -> PrettyTable:
        """
        Recommend top_k ranked diners to user_id.
        Exclude diners that are already liked by user_id in training dataset.
        Include various information, such as diner name or whether it is actually hit.

        Args:
             user_id_mapping (int): target user_id to recommend after mapping.
             tr_liked_diners (List[int]): list of diners liked by user_id in training dataset.
             test_liked_diners (List[int]): list of diners liked by user_id test dataset.

        Returns (PrettyTable):
            PrettyTable object.
        """
        if test_liked_diners is None:
            test_liked_diners = []
        pred_diner_id, pred_diner_score = self._recommend(
            user_id=torch.tensor([user_id_mapping]),
            tr_liked_diners=tr_liked_diners,
            top_k=top_k,
            near_diner_ids=self.near_diner_ids,
        )
        pred_diner_id_mapping = [
            self.reverse_diner_mapping[diner] for diner in pred_diner_id
        ]

        tb = PrettyTable(
            field_names=[
                "diner_name",
                "diner_category_large",
                "diner_category_middle",
                "score",
                "hit",
            ],
        )
        tb._set_markdown_style()
        for diner_idx, score in zip(pred_diner_id_mapping, pred_diner_score):
            info = self.diners[lambda x: x["diner_idx"] == diner_idx].iloc[0]
            tb.add_row(
                [
                    info["diner_name"],
                    info["diner_category_large"],
                    info["diner_category_middle"],
                    score,
                    1 if diner_idx in test_liked_diners else 0,
                ]
            )

        return tb
