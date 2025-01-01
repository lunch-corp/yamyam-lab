import os
from collections import defaultdict
from typing import Union

from torch import Tensor


def convert_tensor(
    ts: Tensor, structure: Union[dict, list]
) -> dict[int, Union[list[int], dict[int, int]]]:
    """
    Convert 2 dimensional tensor to dict or list.
    Original tensor includes interaction between reviewer and diner.

    Args:
        ts (Tensor): n x 2 dimension tensors whose columns are matched with (diner_id, reviewer_id).
            Should be careful of column ordering.
        structure (Union[dict, list]): Data type of value corresponding key in return object.

    Returns (Dict[int, Union[List[int], Dict[int, int]]]):
        Key is reviewer id and values are diner_id interacted by reviewer id.
        Data types of values are dictionary or list.
        In case dictionary, res[reviewer_id][diner_id] is 1 if interacted else 0.
        In case list, res[reviewer_id] is a list of diner_id interacted by reviewer id.
    """
    assert ts.shape[1] == 2
    assert structure in [dict, list]
    res = defaultdict(structure)
    for diner_id, reviewer_id in ts:
        reviewer_id = reviewer_id.item()
        diner_id = diner_id.item()
        if isinstance(structure, dict):
            res[reviewer_id][diner_id] = 1
        else:
            res[reviewer_id].append(diner_id)
    return res


def get_num_workers() -> int:
    """
    Get number of workers for data loader in pytorch.

    Returns (int)
        Number of workers for data loader in pytorch. Note that even if there are
        lots of cpus, it may not be a good idea to use many of them because
        context switching overhead could interrupt training.
        It could be best to determine optimal num_workers with minimal experiments.
    """
    num_cores = os.cpu_count()
    return min(4, num_cores // 2)


def safe_divide(numerator, denominator):
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return 0
