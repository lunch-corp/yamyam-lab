import torch
from torch_geometric.nn import Node2Vec as Node2VecPG

from embedding.base import BaseEmbedding
from tools.candidate import get_diner_nearby_candidates

# set cpu or cuda for default option
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device.type)


class Node2Vec(BaseEmbedding):
    def __init__(self, user_ids, diner_ids):
        super().__init__(
            user_ids=user_ids,
            diner_ids=diner_ids
        )

    def initialize(self, edge_index, **kwargs):
        self.model = Node2VecPG(
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
        self.optimizer = torch.optim.Adam(list(self.model.parameters()), lr=kwargs["lr"])


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
                                         y_columns=["reviewer_review_score"],
                                         pg_model=True)
        train, val = prepare_torch_geometric_data(
            X_train=data["X_train"],
            X_val=data["X_val"],
            num_diners=data["num_diners"],
            num_reviewers=data["num_users"],
        )
        node2vec = Node2Vec(
            user_ids=torch.tensor(list(data["user_mapping"].values())),
            diner_ids=torch.tensor(list(data["diner_mapping"].values())),
        )
        node2vec.initialize(
            edge_index=train.edge_index,
            **vars(args)
        )

        # get near 1km diner_ids
        nearby_candidates = get_diner_nearby_candidates(max_distance_km=1)
        # convert diner_ids
        diner_mapping = data["diner_mapping"]
        nearby_candidates_mapping = {}
        for ref_id, nearby_id in nearby_candidates.items():
            nearby_id_mapping = [diner_mapping[diner_id] for diner_id in nearby_id]
            nearby_candidates_mapping[diner_mapping[ref_id]] = nearby_id_mapping

        for epoch in range(args.epochs):
            train_loss = node2vec.train(
                batch_size=args.batch_size,
                random_state=args.random_state,
            )
            logger.info(f"epoch {epoch}: train loss {train_loss:.4f}")

            recommendations = node2vec.recommend(
                X_train=data["X_train"],
                X_val=data["X_val"],
                nearby_candidates=nearby_candidates_mapping,
                filter_already_liked=True
            )

            maps = []
            ndcgs = []
            ranked_precs = []
            for K in node2vec.metric_at_K.keys():
                map = round(node2vec.metric_at_K[K]["map"], 5)
                ndcg = round(node2vec.metric_at_K[K]["ndcg"], 5)
                ranked_prec = round(node2vec.metric_at_K[K]["ranked_prec"], 5)
                count = node2vec.metric_at_K[K]["count"]
                logger.info(f"maP@{K}: {map} with {count} users out of all {node2vec.num_users} users")
                logger.info(f"ndcg@{K}: {ndcg} with {count} users out of all {node2vec.num_users} users")
                logger.info(f"ranked_prec@{K}: {ranked_prec}")

                maps.append(str(map))
                ndcgs.append(str(ndcg))
                ranked_precs.append(str(ranked_prec))

            logger.info(f"map result: {'|'.join(maps)}")
            logger.info(f"ndcg result: {'|'.join(ndcgs)}")
            logger.info(f"ranked_prec result: {'|'.join(ranked_precs)}")

        torch.save(node2vec.model.state_dict(), "node2vec.pt")

        logger.info("successfully saved node2vec torch model")
    except:
        logger.error(traceback.format_exc())
        raise