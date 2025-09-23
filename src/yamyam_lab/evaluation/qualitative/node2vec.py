import os
import pickle
import traceback
from datetime import datetime
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from evaluation.qualitative.base_qualitative_evaluation import BaseQualitativeEvaluation
from numpy.typing import NDArray
from torch import Tensor

from yamyam_lab.model.graph.node2vec import Model
from yamyam_lab.tools.logger import setup_logger
from yamyam_lab.tools.parse_args import parse_args_eval


class Node2VecQualitativeEvaluation(BaseQualitativeEvaluation):
    def __init__(
        self,
        model_path: str,
        user_ids: Tensor,
        diner_ids: Tensor,
        num_nodes: int,
        embedding_dim: int,
        user_mapping: Dict[int, int],
        diner_mapping: Dict[int, int],
        latitude: float = None,
        longitude: float = None,
        near_dist: float = 0.5,
    ):
        """
        Evaluation class for trained node2vec model.
        This class loads pre-trained weights of node2vec model, and does qualitative evaluation.

        Args:
             model_path (str): path to pre-trained node2vec model.
             user_ids (Tensor): list of user_ids used in node2vec model.
             diner_ids (Tensor): list of diner_ids used in node2vec model.
             graph (nx.Graph): dummy value for class initialization.
             num_nodes (int): dummy value for class initialization.
             embedding_dim (int): dimension of node embeddings.
             user_mapping (Dict[int, int]): mapping of user ids used in preprocessing.
             diner_mapping (Dict[int, int]): mapping of diner ids used in preprocessing.
        """
        super().__init__(
            user_mapping=user_mapping,
            diner_mapping=diner_mapping,
            latitude=latitude,
            longitude=longitude,
            near_dist=near_dist,
        )
        self.model = Model(
            user_ids=user_ids,
            diner_ids=diner_ids,
            embedding_dim=embedding_dim,
            inference=True,
            top_k_values=[1],
            graph=nx.Graph(),
            walks_per_node=1,
            num_negative_samples=1,
            num_nodes=num_nodes,
            model_name="node2vec",
            device="cpu",
            recommend_batch_size=2000,
            num_workers=4,
            # parameters for node2vec
            walk_length=1,
        )

        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def _recommend(
        self,
        user_id: Tensor,
        tr_liked_diners: List[int],
        top_k: int = 10,
        near_diner_ids: List[int] = None,
    ) -> Tuple[NDArray, NDArray]:
        return self.model._recommend(
            user_id=user_id,
            already_liked_item_id=tr_liked_diners,
            top_k=top_k,
            near_diner_ids=near_diner_ids,
        )


if __name__ == "__main__":
    BASE_PATH = os.path.join(os.path.dirname(__file__), "../../..")
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    LOG_PATH = f"{BASE_PATH}/result/qualitative_eval/node2vec/{dt}"
    os.makedirs(LOG_PATH, exist_ok=True)
    args = parse_args_eval()
    logger = setup_logger(os.path.join(LOG_PATH, "log.log"))

    try:
        data = pickle.load(open(args.data_obj_path, "rb"))
        num_nodes = data["num_users"] + data["num_diners"]

        train_liked_series = (
            pd.DataFrame(data["X_train"], columns=["diner_idx", "reviewer_id"])
            .groupby("reviewer_id")["diner_idx"]
            .apply(np.array)
        )
        test_liked_series = (
            pd.DataFrame(
                data["X_test_warm_users"], columns=["diner_idx", "reviewer_id"]
            )
            .groupby("reviewer_id")["diner_idx"]
            .apply(np.array)
        )

        # qualitative evaluation
        qualitative_eval = Node2VecQualitativeEvaluation(
            model_path=args.model_path,
            user_ids=torch.tensor(list(data["user_mapping"].values())),
            diner_ids=torch.tensor(list(data["diner_mapping"].values())),
            num_nodes=num_nodes,
            embedding_dim=args.embedding_dim,
            user_mapping=data["user_mapping"],
            diner_mapping=data["diner_mapping"],
            latitude=args.latitude,
            longitude=args.longitude,
            near_dist=args.near_dist,
        )

        for reviewer_id in args.user_ids:
            reviewer_id_mapping = data["user_mapping"].get(reviewer_id)
            if reviewer_id_mapping is None:
                logger.info(
                    f"reviewer_id {reviewer_id} does not exist in user mapping dictionary"
                    "Please check it again if you enter wrong reviewer_id."
                )
                continue
            if reviewer_id_mapping not in train_liked_series:
                if reviewer_id_mapping in test_liked_series:
                    logger.info(
                        f"reviewer_id {reviewer_id} is cold start user, so most popular items will be recommended"
                    )
                    continue
                else:
                    logger.info(
                        f"reviewer_id {reviewer_id} is cold start user and does not exist in test data"
                    )
                    continue
            else:
                if reviewer_id_mapping not in test_liked_series:
                    logger.info(
                        f"reviewer_id {reviewer_id} exists in train data, but does not exist in test data"
                    )
                else:
                    logger.info(
                        f"reviewer_id {reviewer_id} exists in both of train data and test data"
                    )

            if args.latitude is not None and args.longitude is not None:
                logger.info(
                    f"Diners within {args.near_dist}km based on coordinate {(args.latitude, args.longitude)} will be recommended"
                )

            tb = qualitative_eval.recommend(
                user_id_mapping=reviewer_id_mapping,
                tr_liked_diners=train_liked_series.get(reviewer_id_mapping),
                test_liked_diners=test_liked_series.get(reviewer_id_mapping),
                top_k=args.top_k,
            )
            logger.info(f"Recommendations for user {reviewer_id}")
            logger.info(tb)
    except:
        logger.error(traceback.format_exc())
        raise
