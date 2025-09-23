from typing import List, Tuple, Union

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.sampling import np_edge_dropout
from torch import Tensor

from yamyam_lab.model.graph.base_embedding import BaseEmbedding


class Model(BaseEmbedding):
    """
    LightGCN implementation compatible with train_graph.py format.
    """

    def __init__(
        self,
        # parameters for base_embedding
        user_ids: Tensor,
        diner_ids: Tensor,
        top_k_values: List[int],
        graph: nx.Graph,
        embedding_dim: int,
        walks_per_node: int,
        num_negative_samples: int,
        num_nodes: int,
        model_name: str,
        device: str,
        recommend_batch_size: int,
        num_workers: int,
        # parameters for lightgcn
        num_layers: int = 3,
        drop_ratio: float = 0.0,
        inference: bool = False,
        **kwargs,
    ):
        """
        LightGCN model compatible with train_graph.py format.

        Args:
            user_ids (Tensor): User ids in data.
            diner_ids (Tensor): Diner ids in data.
            top_k_values (List[int]): Top k values used when calculating metric for prediction and candidate generation.
            graph (nx.Graph): Networkx graph object generated from train data.
            embedding_dim (int): Dimension of user / diner embedding vector.
            walks_per_node (int): Number of generated walks for each node (not used in LightGCN).
            num_negative_samples (int): Number of negative samples for each node.
            num_nodes (int): Total number of nodes.
            model_name (str): Model name.
            device (str): Device on which train is run. (cpu or cuda)
            recommend_batch_size (int): Batch size when calculating validation metric.
            num_layers (int): Number of LightGCN layers.
            drop_ratio (float): Edge dropout ratio.
            inference (bool): Indicator whether inference mode or not.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            user_ids=user_ids,
            diner_ids=diner_ids,
            top_k_values=top_k_values,
            graph=graph,
            embedding_dim=embedding_dim,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            num_nodes=num_nodes,
            model_name=model_name,
            device=device,
            recommend_batch_size=recommend_batch_size,
            num_workers=num_workers,
        )

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio

        # Create node ID mappings for LightGCN
        self._create_node_mappings()

        # Convert networkx graph to interaction matrix
        self.interaction = self._graph_to_interaction_matrix()

        # Initialize embeddings
        self._init_embeddings()

        # Pre-build graph without dropout for evaluation
        self.graph_static = self._build_graph(drop_ratio=0.0).to(self.device)

    def _create_node_mappings(self):
        """
        Create mappings between original node IDs and sequential indices.
        """
        # Get all nodes from the graph
        all_nodes = sorted(list(self.graph.nodes()))

        # Create mappings
        self.node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}

        # Update user and diner counts based on actual graph
        user_nodes = [node for node in all_nodes if node < self.num_users]
        diner_nodes = [node for node in all_nodes if node >= self.num_users]

        self.actual_num_users = len(user_nodes)
        self.actual_num_diners = len(diner_nodes)

    def _graph_to_interaction_matrix(self) -> sp.csr_matrix:
        """
        Convert networkx graph to interaction matrix.
        """
        # Create user-item interaction matrix
        user_item_edges = []
        for edge in self.graph.edges(data=True):
            u, v, data = edge
            # Assuming users come first in node ordering
            if u < self.num_users and v >= self.num_users:
                # u is user, v is diner
                user_idx = u
                diner_idx = v - self.num_users
                weight = data.get("weight", 1.0)
                user_item_edges.append((user_idx, diner_idx, weight))
            elif v < self.num_users and u >= self.num_users:
                # v is user, u is diner
                user_idx = v
                diner_idx = u - self.num_users
                weight = data.get("weight", 1.0)
                user_item_edges.append((user_idx, diner_idx, weight))

        if not user_item_edges:
            # Create empty matrix if no edges
            return sp.csr_matrix((self.num_users, self.num_diners))

        users, diners, weights = zip(*user_item_edges)
        return sp.coo_matrix(
            (weights, (users, diners)), shape=(self.num_users, self.num_diners)
        ).tocsr()

    def _init_embeddings(self) -> None:
        """
        Initialize user and diner embeddings.
        """
        self.user_emb = nn.Parameter(torch.empty(self.num_users, self.embedding_dim))
        self.diner_emb = nn.Parameter(torch.empty(self.num_diners, self.embedding_dim))
        nn.init.xavier_uniform_(self.user_emb)
        nn.init.xavier_uniform_(self.diner_emb)

    def _build_graph(self, drop_ratio: float = 0.0) -> torch.sparse_coo_tensor:
        """
        Build Laplacian-normalised adjacency with optional edge dropout.
        """
        user_item = self.interaction  # (U × I)

        # Bipartite adjacency
        tl = sp.csr_matrix((user_item.shape[0], user_item.shape[0]))
        br = sp.csr_matrix((user_item.shape[1], user_item.shape[1]))
        adj = sp.bmat([[tl, user_item], [user_item.T, br]]).tocsr()

        if drop_ratio:
            adj = adj.tocoo()
            new_val = np_edge_dropout(adj.data, drop_ratio)
            adj = sp.coo_matrix((new_val, (adj.row, adj.col)), shape=adj.shape).tocsr()

        return self._to_sparse_tensor(self._normalize_adj(adj))

    def forward(self, batch: Tensor) -> Tensor:
        """
        Forward pass for LightGCN.
        """
        # For LightGCN, we need to propagate through the graph
        user_emb, diner_emb = self.propagate()

        # Update _embedding for compatibility
        self._embedding[: self.num_users] = user_emb
        self._embedding[self.num_users :] = diner_emb

        return self._embedding if batch is None else self._embedding[batch]

    def propagate(self, test: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Propagate embeddings through LightGCN layers.
        Args:
            test (bool): Indicator whether test mode or not.
        Returns:
            Tuple[Tensor, Tensor]: User and diner embeddings.
        """
        if test:
            graph = self.graph_static
        else:
            graph = self._build_graph(drop_ratio=self.drop_ratio).to(self.device)

        # Initial embeddings
        user_emb = self.user_emb
        diner_emb = self.diner_emb

        # Stack all embeddings
        all_emb = torch.cat([user_emb, diner_emb], dim=0)

        # LightGCN propagation
        emb_list = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            emb_list.append(all_emb)

        # Final embeddings (average of all layers)
        final_emb = torch.stack(emb_list, dim=1).mean(dim=1)

        # Split back to user and diner embeddings
        user_final_emb = final_emb[: self.num_users]
        diner_final_emb = final_emb[self.num_users :]

        return user_final_emb, diner_final_emb

    def pos_sample(self, batch: Tensor) -> Tensor:
        """
        Generate positive samples for training.
        For LightGCN, we use BPR loss, so this returns user-item pairs.
        """
        pos_items = []

        for user in batch:
            user_id = min(int(user), self.num_users - 1)

            # Default: random item
            pos_item = np.random.randint(self.num_diners)

            if self.graph.has_node(user_id):
                neighbors = list(self.graph.neighbors(user_id))
                diner_neighbors = [n for n in neighbors if n >= self.num_users]

                if diner_neighbors:
                    # Pick a real positive
                    pos_item = np.random.choice(diner_neighbors) - self.num_users
                    pos_item = min(pos_item, self.num_diners - 1)  # ensure valid index

            pos_items.append(pos_item)

        pos_items = torch.tensor(pos_items, dtype=batch.dtype, device=batch.device)
        return torch.stack([batch, pos_items], dim=1)

    def neg_sample(self, batch: Tensor) -> Tensor:
        """
        Generate negative samples for training.
        Args:
            batch (Tensor): Batch of user ids.
        Returns:
            Tensor: Negative samples.
        """
        neg_items = []

        for user_id in batch:
            user_neg_items = []
            # Ensure user_id is within bounds
            user_id = min(user_id.item(), self.num_users - 1)

            # Check if user_id exists in the graph
            if self.graph.has_node(user_id):
                # Find positive items for this user from the graph
                user_neighbors = list(self.graph.neighbors(user_id))
                # Filter to only diner nodes
                positive_diners = set(
                    [n - self.num_users for n in user_neighbors if n >= self.num_users]
                )
            else:
                # If user_id doesn't exist in graph, no positive items
                positive_diners = set()

            # Sample negative items that are not positive
            for _ in range(self.num_negative_samples):
                while True:
                    neg_item = np.random.randint(self.num_diners)
                    if neg_item not in positive_diners:
                        user_neg_items.append(neg_item)
                        break

            neg_items.append(user_neg_items)

        neg_items = torch.tensor(neg_items, dtype=batch.dtype, device=batch.device)
        return torch.cat([batch.unsqueeze(1), neg_items], dim=1)

    def _normalize_adj(self, adj: sp.spmatrix) -> sp.csr_matrix:
        """
        Symmetric Laplacian normalisation  D^{-1/2} A D^{-1/2}.
        Args:
            adj (sp.spmatrix): Adjacency matrix.
        Returns:
            sp.csr_matrix: Normalised adjacency matrix.
        """
        row_inv_sqrt = 1.0 / np.sqrt(adj.sum(axis=1).A.ravel() + 1e-8)
        col_inv_sqrt = 1.0 / np.sqrt(adj.sum(axis=0).A.ravel() + 1e-8)
        d_row = sp.diags(row_inv_sqrt)
        d_col = sp.diags(col_inv_sqrt)
        return d_row @ adj @ d_col

    def _to_sparse_tensor(self, mat: sp.spmatrix) -> torch.sparse_coo_tensor:
        """
        Convert a SciPy sparse matrix to a PyTorch sparse tensor.
        Args:
            mat (sp.spmatrix): Sparse matrix.
        Returns:
            torch.sparse_coo_tensor: Sparse tensor.
        """
        mat = mat.tocoo()
        indices = torch.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64))
        values = torch.from_numpy(mat.data.astype(np.float32))
        return torch.sparse_coo_tensor(
            indices, values, torch.Size(mat.shape), dtype=torch.float32
        )

    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Wrapper function for positive, negative sampling.
        """
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        """
        Compute BPR loss for LightGCN.
        Args:
            pos_rw (Tensor): Positive samples.
            neg_rw (Tensor): Negative samples.
        Returns:
            Tensor: BPR loss.
        """
        # pos_rw: (batch_size, 2) - [user_id, pos_item_id]
        # neg_rw: (batch_size, 1 + num_negative_samples) - [user_id, neg_item_ids...]

        users = pos_rw[:, 0]
        pos_items = pos_rw[:, 1]
        neg_items = neg_rw[:, 1:]  # Remove user_id column

        # Get embeddings
        user_emb, diner_emb = self.propagate()

        # Ensure user and item indices are within bounds
        users = torch.clamp(users, 0, self.num_users - 1)
        pos_items = torch.clamp(pos_items, 0, self.num_diners - 1)
        neg_items = torch.clamp(neg_items, 0, self.num_diners - 1)

        # Get user and item embeddings
        user_vec = user_emb[users]  # (batch_size, embedding_dim)
        pos_item_vec = diner_emb[pos_items]  # (batch_size, embedding_dim)
        neg_item_vec = diner_emb[neg_items]  # (batch_size, num_neg, embedding_dim)

        # Calculate scores
        pos_scores = (user_vec * pos_item_vec).sum(dim=-1)  # (batch_size,)
        neg_scores = (user_vec.unsqueeze(1) * neg_item_vec).sum(
            dim=-1
        )  # (batch_size, num_neg)

        # BPR loss: -log(sigmoid(pos_score - neg_score))
        loss = F.softplus(neg_scores - pos_scores.unsqueeze(1)).mean()

        return loss

    def get_embedding(self, batch_tensor: Tensor) -> Tensor:
        """
        Get embeddings for given nodes.
        """
        # Use stored embeddings if available, otherwise propagate
        if self._embedding.sum() == 0:
            user_emb, diner_emb = self.propagate(test=True)
            self._embedding[: self.num_users] = user_emb
            self._embedding[self.num_users :] = diner_emb

        # Ensure all requested nodes are within bounds
        batch_tensor = torch.clamp(batch_tensor, 0, self.num_nodes - 1)

        # Return embeddings for requested nodes
        return self._embedding[batch_tensor]

    def propagate_and_store_embedding(self, batch_nodes: Tensor):
        """
        Propagates LightGCN algorithm to get embeddings for batch_nodes.
        This method is called for compatibility with train_graph.py.

        Args:
            batch_nodes (Tensor): List of nodes to propagate
        """
        with torch.no_grad():
            user_emb, diner_emb = self.propagate(test=True)
            self._embedding[: self.num_users] = user_emb
            self._embedding[self.num_users :] = diner_emb
