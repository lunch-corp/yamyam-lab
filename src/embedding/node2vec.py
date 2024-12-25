from typing import List, Tuple, Union
import networkx as nx

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

# set cpu or cuda for default option
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device.type)

from candidate.near import NearCandidateGenerator
from embedding.base_embedding import BaseEmbedding
from tools.generate_walks import generate_walks, precompute_probabilities
from constant.preprocess import MIN_REVIEWS
from constant.candidate import MAX_DISTANCE_KM


class Node2Vec(BaseEmbedding):
    r"""
    This is a customized version of pytorch geometric implementation of node2vec.
    It differs from pg implementation in 2 aspects.
        - class initialization: Does not use any pyg-lib or torch-cluster.
            Make random walks using explicit function.
        - data structure: Uses networkx.Graph.

    Original doc string starts here.
    ----------------------------------
    The Node2Vec model from the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
    length :obj:`walk_length` are sampled in a given graph, and node embeddings
    are learned via negative sampling optimization.

    .. note::

        For an example of using Node2Vec, see `examples/node2vec.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        node2vec.py>`_.

    Args:
        graph (nx.Graph): Graph object.
        embedding_dim (int): The size of each embedding vector.
        walk_length (int): The walk length.
        context_size (int): The actual context size which is considered for
            positive samples. This parameter increases the effective sampling
            rate by reusing samples across different source nodes.
        walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
        num_negative_samples (int, optional): The number of negative samples to
            use for each positive sample. (default: :obj:`1`)
    """
    def __init__(
        self,
        user_ids: Tensor,
        diner_ids: Tensor,
        graph: nx.Graph,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        num_nodes: int,
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
    ):
        super().__init__(
            user_ids=user_ids,
            diner_ids=diner_ids,
        )
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.EPS = 1e-15
        self.num_nodes = num_nodes

        self.embedding = Embedding(self.num_nodes, embedding_dim)

        self.d_graph = precompute_probabilities(
            graph=graph,
            p=p,
            q=q,
        )

    def forward(self, batch: Tensor) -> Tensor:
        """
        Dummy forward pass which actually does not do anything.

        Args:
            batch (Tensor): A batch of node ids.

        Returns (Tensor):
            A batch of node embeddings.
        """
        emb = self.embedding.weight
        return emb if batch is None else emb[batch]

    def loader(self, **kwargs) -> DataLoader:
        """
        Node id generator in pytorch dataloader type.

        Returns (DataLoader):
            DataLoader used when training model.
        """
        return DataLoader(torch.tensor([node for node in self.graph.nodes()]), collate_fn=self.sample,
                          **kwargs)

    @torch.jit.export
    def pos_sample(self, batch: Tensor) -> Tensor:
        """
        For each of node id, generate biased random walk using `generate_walks` function.
        Based on transition probabilities information (`d_graph`), perform biased random walks.

        Args:
            batch (Tensor): A batch of node ids which are starting points in each biased random walk.

        Returns (Tensor):
            Generated biased random walks. Number of random walks are based on walks_per_node,
            walk_length, and context size. Note that random walks are concatenated row-wise.
        """
        batch = batch.repeat(self.walks_per_node)
        rw = generate_walks(
            node_ids=batch.detach().cpu().numpy(),
            d_graph=self.d_graph,
            walk_length=self.walk_length,
            num_walks=self.walks_per_node,
        )
        walks = []
        num_walks_per_rw = self.walk_length - (self.context_size - 1)
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        """
        Sample negative with uniform sampling.
        In word2vec objective function, to reduce computation burden, negative sampling
        is performed and approximate denominator of probability.

        Args:
            batch (Tensor): A batch of node ids.

        Returns (Tensor):
            Negative samples for each of node ids.
        """
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(self.num_nodes, (batch.size(0), self.walk_length),
                           dtype=batch.dtype, device=batch.device)
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = self.walk_length - (self.context_size - 1)
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Wrapper function for positive, negative sampling.
        This function is used as `collate_fn` in pytorch dataloader.

        Args:
            batch (Union[List[int], Tensor]): A batch of node ids.

        Returns (Tuple[Tensor, Tensor]):
            Positive, negative samples.
        """
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    @torch.jit.export
    def loss(
            self,
            pos_rw: Tensor,
            neg_rw: Tensor
        ) -> Tensor:
        """
        Computes word2vec skip-gram based loss.

        Args:
             pos_rw (Tensor): Node ids of positive samples
             neg_rw (Tensor): Node ids of negative samples

        Returns (Tensor):
            Calculated loss.
        """
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss


if __name__ == "__main__":
    import traceback
    from tools.parse_args import parse_args
    from tools.logger import setup_logger
    from preprocess.preprocess import train_test_split_stratify, prepare_networkx_data

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

        data = train_test_split_stratify(
            test_size=args.test_ratio,
            min_reviews=MIN_REVIEWS,
            X_columns=["diner_idx", "reviewer_id"],
            y_columns=["reviewer_review_score"],
            pg_model=False
        )
        train_graph, val_graph = prepare_networkx_data(
            X_train=data["X_train"],
            X_val=data["X_val"],
        )
        num_nodes = data["num_users"] + data["num_diners"]
        model = Node2Vec(
            user_ids=torch.tensor(list(data["user_mapping"].values())),
            diner_ids=torch.tensor(list(data["diner_mapping"].values())),
            graph=train_graph,
            embedding_dim=args.embedding_dim,
            walk_length=args.walk_length,
            walks_per_node=args.walks_per_node,
            num_nodes=num_nodes,
            context_size=args.context_size,
            q=args.q,
            p=args.p,
        )
        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)

        # get near 1km diner_ids
        candidate_generator = NearCandidateGenerator()
        near_diners = candidate_generator.get_near_candidates_for_all_diners(max_distance_km=MAX_DISTANCE_KM)

        # convert diner_ids
        diner_mapping = data["diner_mapping"]
        nearby_candidates_mapping = {}
        for ref_id, nearby_id in near_diners.items():
            # only get diner appeared in train/val dataset
            if diner_mapping.get(ref_id) is None:
                continue
            nearby_id_mapping = [diner_mapping.get(diner_id) for diner_id in nearby_id if diner_mapping.get(diner_id) != None]
            nearby_candidates_mapping[diner_mapping[ref_id]] = nearby_id_mapping

        seed = torch.Generator(device=device.type).manual_seed(args.random_state)
        loader = model.loader(batch_size=args.batch_size, shuffle=True, generator=seed)
        for epoch in range(args.epochs):
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            total_loss /= len(loader)

            logger.info(f"epoch {epoch}: train loss {total_loss:.4f}")

            recommendations = model.recommend(
                X_train=data["X_train"],
                X_val=data["X_val"],
                nearby_candidates=nearby_candidates_mapping,
                filter_already_liked=True
            )

            maps = []
            ndcgs = []
            ranked_precs = []
            recalls = []
            for K in model.metric_at_K.keys():
                map = round(model.metric_at_K[K]["map"], 5)
                ndcg = round(model.metric_at_K[K]["ndcg"], 5)
                ranked_prec = round(model.metric_at_K[K]["ranked_prec"], 5)
                near_candidate_recall = round(model.metric_at_K[K]["near_candidate_recall"], 5)
                count = model.metric_at_K[K]["count"]
                if K <= 20:
                    logger.info(f"maP@{K}: {map} with {count} users out of all {model.num_users} users")
                    logger.info(f"ndcg@{K}: {ndcg} with {count} users out of all {model.num_users} users")
                    logger.info(f"ranked_prec@{K}: {ranked_prec}")
                else:
                    logger.info(f"near_candidate_recall@{K}: {near_candidate_recall}")

                maps.append(str(map))
                ndcgs.append(str(ndcg))
                ranked_precs.append(str(ranked_prec))
                recalls.append(str(near_candidate_recall))

            logger.info(f"map result: {'|'.join(maps)}")
            logger.info(f"ndcg result: {'|'.join(ndcgs)}")
            logger.info(f"ranked_prec result: {'|'.join(ranked_precs)}")
            logger.info(f"near_candidate_recall result: {'|'.join(recalls)}")
        torch.save(model.state_dict(), "node2vec.pt")

        logger.info("successfully saved node2vec torch model")
    except:
        logger.error(traceback.format_exc())
        raise