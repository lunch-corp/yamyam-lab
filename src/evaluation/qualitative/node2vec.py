import pickle

import networkx as nx
import torch
from numpy.typing import NDArray
from torch import Tensor

from constant.evaluation.qualitative import QualitativeReviewerId
from embedding.node2vec import Model
from evaluation.qualitative.base_qualitative_evaluation import BaseQualitativeEvaluation
from tools.parse_args import parse_args_eval
from tools.utils import convert_tensor


class Node2VecQualitativeEvaluation(BaseQualitativeEvaluation):
    def __init__(
        self,
        model_path: str,
        user_ids: Tensor,
        diner_ids: Tensor,
        graph: nx.Graph,
        num_nodes: int,
        embedding_dim: int,
        user_mapping: dict[int, int],
        diner_mapping: dict[int, int],
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
        )
        self.model = Model(
            user_ids=user_ids,
            diner_ids=diner_ids,
            graph=graph,
            embedding_dim=embedding_dim,  # trained model embedding dim
            walk_length=20,  # dummy value
            num_nodes=num_nodes,
            inference=True,
            top_k_values=[1],  # dummy value
        )

        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def _recommend(
        self,
        user_id: Tensor,
        tr_liked_diners: list[int],
        top_k: int = 10,
    ) -> tuple[NDArray, NDArray]:
        return self.model._recommend(
            user_id=user_id,
            already_liked_item_id=tr_liked_diners,
            top_k=top_k,
        )


if __name__ == "__main__":

    import traceback

    from tools.logger import setup_logger

    args = parse_args_eval()
    logger = setup_logger(args.log_path)

    try:
        data = pickle.load(open(args.data_obj_path, "rb"))
        num_nodes = data["num_users"] + data["num_diners"]

        train_liked = convert_tensor(data["X_train"], list)
        val_liked = convert_tensor(data["X_val"], list)

        # qualitative evaluation
        qualitative_eval = Node2VecQualitativeEvaluation(
            model_path=args.model_path,
            user_ids=torch.tensor(list(data["user_mapping"].values())),
            diner_ids=torch.tensor(list(data["diner_mapping"].values())),
            graph=nx.Graph(),  # dummy graph
            num_nodes=num_nodes,
            embedding_dim=args.embedding_dim,
            user_mapping=data["user_mapping"],
            diner_mapping=data["diner_mapping"],
        )

        for enum in QualitativeReviewerId:
            reviewer_id = enum.value
            reviewer_name = enum.name
            reviewer_id_mapping = data["user_mapping"].get(reviewer_id)
            if reviewer_id_mapping is None:
                logger.info(
                    f"reviewer {reviewer_name} not existing in training dataset"
                )
                continue
            tb = qualitative_eval.recommend(
                user_id=reviewer_id,
                tr_liked_diners=train_liked[reviewer_id_mapping],
                val_liked_diners=val_liked[reviewer_id_mapping],
                top_k=10,
            )
            logger.info(f"Recommendations for user {reviewer_name}")
            logger.info(tb)
    except:
        logger.error(traceback.format_exc())
        raise
