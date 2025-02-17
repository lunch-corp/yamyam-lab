import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def svd_loss(
    pred: Tensor,
    true: Tensor,
    params: nn.Parameter,
    regularization: float,
    user_idx: Tensor,
    diner_idx: Tensor,
    num_users: int,
    num_diners: int,
) -> float:
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

    Returns (float):
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
