import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from loss.custom import cal_bpr_loss
from tools.sampling import np_edge_dropout


class LightGCN(nn.Module):
    def __init__(self, conf, trn_graph):
        super().__init__()
        self.conf = conf
        self.device = self.conf["device"]
        self.embedding_size = conf["embedding_size"]
        self.num_reviewer = conf["num_reviewer"]
        self.num_diner = conf["num_diner"]
        self.num_layers = self.conf["num_layers"]
        self.trn_graph = trn_graph

        self.init_emb()
        self.get_graph_ori()
        self.get_graph()

    def init_emb(self):
        # no usage
        self.reviewer_emb = nn.Parameter(
            torch.FloatTensor(self.num_reviewer, self.embedding_size)
        )
        nn.init.xavier_normal_(self.reviewer_emb)
        self.diner_emb = nn.Parameter(
            torch.FloatTensor(self.num_diner, self.embedding_size)
        )
        nn.init.xavier_normal_(self.diner_emb)

    def get_graph(self):
        graph = self.trn_graph
        device = self.device
        drop_ratio = self.conf["drop_ratio"]
        total_graph = sp.bmat(
            [
                [sp.csr_matrix((graph.shape[0], graph.shape[0])), graph],
                [graph.T, sp.csr_matrix((graph.shape[1], graph.shape[1]))],
            ]
        )
        if drop_ratio != 0:
            graph = total_graph.tocoo()
            values = np_edge_dropout(graph.data, drop_ratio)
            total_graph = sp.coo_matrix(
                (values, (graph.row, graph.col)), shape=graph.shape
            ).tocsr()
        self.graph = to_tensor(laplace_transform(total_graph)).to(device)

    def get_graph_ori(self):
        graph = self.trn_graph
        device = self.device
        total_graph = sp.bmat(
            [
                [sp.csr_matrix((graph.shape[0], graph.shape[0])), graph],
                [graph.T, sp.csr_matrix((graph.shape[1], graph.shape[1]))],
            ]
        )
        self.graph_ori = to_tensor(laplace_transform(total_graph)).to(device)

    def propagate(self, test=False):
        reviewer_vec = self.reviewer_emb
        diner_vec = self.diner_emb
        return reviewer_vec, diner_vec

    def cal_loss(self, reviewer_feature, diner_feature, weight):
        pred = torch.sum(reviewer_feature * diner_feature, 2)
        bpr_loss = cal_bpr_loss(pred, weight)
        return bpr_loss

    def forward(self, batch):
        self.get_graph()

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        # weights: [bs, 1]
        users, bundles, weights = batch
        # print(users.shape)
        # print(bundles.shape)
        # print(weights.shape)
        reviewer_feature, diner_feature = self.propagate()

        reviewer_embedding = reviewer_feature[users].expand(-1, bundles.shape[1], -1)
        diner_embedding = diner_feature[bundles]
        bpr_loss = self.cal_loss(reviewer_embedding, diner_embedding, weights)

        return bpr_loss

    def evaluate(self, propagate_result, reviewers):
        reviewer_feature, diner_feature = propagate_result
        scores = torch.mm(reviewer_feature[reviewers], diner_feature.t())
        return scores
    

def to_tensor(graph: sp.coo_matrix) -> torch.sparse.FloatTensor:
    """
    Convert scipy COO matrix to PyTorch sparse tensor.

    Args:
        graph (sp.coo_matrix): Input graph.

    Returns:
        torch.sparse.FloatTensor: PyTorch sparse tensor.
    """
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(
        torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape)
    )
    return graph


def laplace_transform(graph: sp.csr_matrix) -> sp.csr_matrix:
    """
    Apply symmetric Laplacian normalization to a sparse matrix.

    Args:
        graph (sp.csr_matrix): Adjacency matrix.

    Returns:
        sp.csr_matrix: Normalized graph.
    """
    rowsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph