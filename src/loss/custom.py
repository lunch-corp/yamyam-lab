import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-15


def svd_loss(
    pred: Tensor,
    true: Tensor,
    params: nn.Parameter,
    regularization: float,
    user_idx: Tensor,
    diner_idx: Tensor,
    num_users: int,
    num_diners: int,
) -> Tensor:
    """
    Calculates svd loss using bias together.

    Args:
        pred (Tensor): Predicted ratings using model.
        true (Tensor): True ratings from validation dataset.
        params (nn.Parameter): Model parameters in tensor generator.
        regularization (float): Regularization parameter.
        user_idx (Tensor): User ids used when extracting related embeddings.
        diner_idx (Tensor): Diner ids used when extracting related embeddings.
        num_users (int): Number of users.
        num_diners (int): Number of diners.

    Returns (Tensor):
        Calculated svd loss.
    """
    true = true.squeeze()
    mse = F.mse_loss(pred, true, reduction="mean")
    penalty = torch.tensor(0.0, requires_grad=True)
    for param in params:
        if param.shape[0] == num_users:
            param = param[user_idx]
        elif param.shape[0] == num_diners:
            param = param[diner_idx]
        else:
            continue
        penalty = penalty + param.data.norm(dim=1).pow(2).sum() * regularization
    return mse + penalty


def basic_contrastive_loss(pos_rw_emb: Tensor, neg_rw_emb: Tensor) -> Tensor:
    """
    Calculate contrastive loss used in word2vec.

    Loss_u = - \log ( \sigma( z^T_u z_v ) ) - \sum^k_{i=1} \log ( \sigma( -z^T_u z_{n_i} ) )

    Args:
        pos_rw_emb (Tensor): 3 dim tensor with (pos_start_node_size x pos_sample_size x embedding_dim)
        neg_rw_emb (Tensor): 3 dim tensor with (neg_start_node_size x neg_sample_size x embedding_dim)

    Returns (Tensor):
        Calculated contrastive loss combined with positive loss and negative loss.
    """
    assert pos_rw_emb.dim() == 3
    assert neg_rw_emb.dim() == 3
    assert pos_rw_emb.size(0) == neg_rw_emb.size(0)
    assert pos_rw_emb.size(2) == neg_rw_emb.size(2)

    # positive loss
    h_start = pos_rw_emb[:, 0:1, :]
    h_rest = pos_rw_emb[:, 1:, :]

    out = (h_start * h_rest).sum(dim=-1).view(-1)
    pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

    # negative loss
    h_start = neg_rw_emb[:, 0:1, :]
    h_rest = neg_rw_emb[:, 1:, :]

    out = (h_start * h_rest).sum(dim=-1).view(-1)
    neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

    return pos_loss + neg_loss


def cal_bpr_loss(pred, weight):
    """
    Calculates the Bayesian Personalized Ranking (BPR) loss.

    Args:
        pred (torch.Tensor): Predicted scores with shape [batch_size, 1 + neg_num].
                             The first column is the positive item, and the second is a negative item.
        weight (torch.Tensor): Sample-wise weights with shape [batch_size] or [batch_size, 1].

    Returns:
        torch.Tensor: Scalar tensor representing the BPR loss over the batch.
    """
    negs = pred[:, 1].unsqueeze(1)
    pos = pred[:, 0].unsqueeze(1)
    loss = -torch.mean(weight * torch.log(torch.sigmoid(pos - negs)))  # [bs]
    return loss