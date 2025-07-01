from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from loss.custom import cal_bpr_loss
from tools.sampling import np_edge_dropout


# --------------------------------------------------------------------------- #
#                               Helper routines                               #
# --------------------------------------------------------------------------- #
def normalize_adj(adj: sp.spmatrix) -> sp.csr_matrix:
    """
    Symmetric Laplacian normalisation  D^{-1/2} A D^{-1/2}.
    """
    row_inv_sqrt = 1.0 / np.sqrt(adj.sum(axis=1).A.ravel() + 1e-8)
    col_inv_sqrt = 1.0 / np.sqrt(adj.sum(axis=0).A.ravel() + 1e-8)
    d_row = sp.diags(row_inv_sqrt)
    d_col = sp.diags(col_inv_sqrt)
    return d_row @ adj @ d_col


def to_sparse_tensor(mat: sp.spmatrix) -> torch.sparse.FloatTensor:
    """
    Convert a SciPy sparse matrix to a PyTorch sparse tensor.
    """
    mat = mat.tocoo()
    indices = torch.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64))
    values = torch.from_numpy(mat.data.astype(np.float32))
    return torch.sparse.FloatTensor(indices, values, torch.Size(mat.shape))


# --------------------------------------------------------------------------- #
#                                  Model                                      #
# --------------------------------------------------------------------------- #
class LightGCN(nn.Module):
    """
    Minimal LightGCN implementation with optional edge dropout.
    """

    print("this is called")

    def __init__(self, conf: dict, interaction: sp.csr_matrix) -> None:
        super().__init__()
        self.conf = conf
        self.device = torch.device(conf["device"])

        self.embedding_size: int = conf["embedding_size"]
        self.num_reviewer: int = conf["num_reviewer"]
        self.num_diner: int = conf["num_diner"]
        self.num_layers: int = conf["num_layers"]
        self.drop_ratio: float = conf.get("drop_ratio", 0.0)

        # Interaction matrix (U × I)
        self.interaction = interaction

        self._init_embeddings()

        # Pre-build graph without dropout for evaluation
        self.graph_static = self._build_graph(drop_ratio=0.0).to(self.device)

    # --------------------------------------------------------------------- #
    #                               Forward                                 #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        batch: Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute BPR loss for one mini-batch.

        Parameters
        ----------
        batch
            reviewers: (bs,)               – user indices
            diners:    (bs, 1 + neg_num)   – positive + negative item indices
            weights:   (bs, 1)             – sample weights
        """
        reviewers, diners, weights = batch

        if reviewers.dim() > 1:
            reviewers = reviewers.squeeze(-1)

        # Re-build graph with edge-dropout at every iteration (as in paper)
        self.graph = self._build_graph(drop_ratio=self.drop_ratio).to(self.device)

        reviewer_emb, diner_emb = self.reviewer_emb, self.diner_emb
        # Shape alignment
        rev_vec = reviewer_emb[reviewers].unsqueeze(1).expand(-1, diners.size(1), -1)
        din_vec = diner_emb[diners]

        preds = (rev_vec * din_vec).sum(dim=-1)
        return cal_bpr_loss(preds, weights)

    # --------------------------------------------------------------------- #
    #                              Inference                                #
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def evaluate(
        self,
        reviewers: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Return the full score matrix for reviewers × all diners.
        """
        reviewer_emb, diner_emb = self.reviewer_emb, self.diner_emb
        return reviewer_emb[reviewers] @ diner_emb.t()

    # --------------------------------------------------------------------- #
    #                         Initialisation utils                          #
    # --------------------------------------------------------------------- #
    def _init_embeddings(self) -> None:
        self.reviewer_emb = nn.Parameter(
            torch.empty(self.num_reviewer, self.embedding_size)
        )
        self.diner_emb = nn.Parameter(torch.empty(self.num_diner, self.embedding_size))
        nn.init.xavier_uniform_(self.reviewer_emb)
        nn.init.xavier_uniform_(self.diner_emb)

    # --------------------------------------------------------------------- #
    #                            Graph builder                              #
    # --------------------------------------------------------------------- #
    def _build_graph(self, drop_ratio: float = 0.0) -> torch.sparse.FloatTensor:
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

        return to_sparse_tensor(normalize_adj(adj))
