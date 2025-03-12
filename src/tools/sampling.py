import torch
from torch import Tensor


def uniform_sampling_without_replacement_from_pool(
        pool: Tensor,
        size: int,
) -> Tensor:
    """
    Uniform sampling without replacement using `torch.randperm`.

    Args:
        pool (Tensor): Sampling pool.
        size (int): Number of samples.

    Returns (Tensor):
        Sampling result without replacement.
    """
    indices = torch.randperm(len(pool))[:size]
    samples = pool[indices]
    return samples