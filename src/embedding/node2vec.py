import torch
from torch_geometric.nn import Node2Vec as Node2VecPG

from embedding.base import BaseEmbedding


class Node2Vec(BaseEmbedding):
    def __init__(self):
        super().__init__()

    def initialize(self, edge_index, **kwargs):
        model = Node2VecPG(
            edge_index=edge_index,
            embedding_dim=kwargs["embedding_dim"],
            walk_length=kwargs["walk_length"],
            context_size=kwargs["context_size"],
            walks_per_node=kwargs["walks_per_node"],
            num_negative_samples=kwargs["num_negative_samples"],
            p=kwargs["p"],
            q=kwargs["q"],
            sparse=kwargs["sparse"],
        )
        optimizer = torch.optim.Adam(list(model.parameters()), lr=kwargs["lr"])
        return model, optimizer


if __name__ == "__main__":
    import traceback
    from tools.parse_args import parse_args
    from tools.logger import setup_logger
    from preprocess.preprocess import train_test_split_stratify, prepare_torch_geometric_data

    args = parse_args()
    logger = setup_logger(args.log_path)

    try:
        logger.info(f"batch size: {args.batch_size}")
        logger.info(f"learning rate: {args.lr}")
        logger.info(f"regularization: {args.regularization}")
        logger.info(f"epochs: {args.epochs}")
        logger.info(f"test ratio: {args.test_ratio}")
        logger.info(f"embedding dimension: {args.embedding_dim}")
        logger.info(f"walk length: {args.walk_length}")
        logger.info(f"context size: {args.context_size}")
        logger.info(f"walks per node: {args.walks_per_node}")
        logger.info(f"num neg samples: {args.num_negative_samples}")
        logger.info(f"p: {args.p}")
        logger.info(f"q: {args.q}")
        logger.info(f"sparse: {args.sparse}")

        data = train_test_split_stratify(test_size=args.test_ratio,
                                         min_reviews=3,
                                         X_columns=["diner_idx", "reviewer_id"],
                                         y_columns=["reviewer_review_score"])
        train, val = prepare_torch_geometric_data(
            X_train=data["X_train"],
            X_val=data["X_val"],
            num_diners=data["num_diners"],
            num_reviewers=data["num_users"],
        )
        node2vec = Node2Vec()
        model, optimizer = node2vec.initialize(
            edge_index=train.edge_index,
            **vars(args)
        )
        for epoch in range(args.epochs):
            model, train_loss = node2vec.train(model, optimizer, batch_size=args.batch_size)
            logger.info(f"epoch {epoch}: train loss {train_loss:.4f}")
    except:
        logger.error(traceback.format_exc())
        raise