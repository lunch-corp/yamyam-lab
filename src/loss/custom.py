import torch
import torch.nn.functional as F


def svd_loss(pred, true, params, regularization):
    true = true.squeeze()
    mse = F.mse_loss(pred, true, reduction='mean')
    penalty = torch.tensor(0., requires_grad=True)
    for param in params:
        penalty = penalty + param.data.norm(dim=1).pow(2).sum() * regularization
    return mse + penalty