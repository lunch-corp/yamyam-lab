import random
from typing import List

import torch
from torch import Tensor


def uniform_sampling_without_replacement_from_large_size_pool(
    pool: Tensor,
    size: int,
) -> Tensor:
    """
    Uniform sampling without replacement using `torch.randperm`.
    When the size of pool is very large, it is more efficient to use permutation than random.sample,
    in case uniform sampling.

    Args:
        pool (Tensor): Sampling pool.
        size (int): Number of samples.

    Returns (Tensor):
        Sampling result without replacement.
    """
    indices = torch.randperm(len(pool))[:size]
    samples = pool[indices]
    return samples


def uniform_sampling_without_replacement_from_small_size_pool(
    pool: List[int],
    size: int,
) -> List[int]:
    """
    Uniform sampling without replacement using `random.sampling`.
    Note that when size of pool is very large, using permutation is recommended.

    Args:
        pool (List[int]): Sampling pool.
        size (int): Number of samples.

    Returns (List[int]):
        Sampling result without replacement.
    """
    return random.sample(pool, size)
