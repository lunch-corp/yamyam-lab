import torch
from torch import Tensor


def uniform_sampling_without_replacement_from_pool(
        pool: Tensor,
        size: int,
) -> Tensor:
    indices = torch.randperm(len(pool))[:size]
    samples = pool[indices]
    return samples