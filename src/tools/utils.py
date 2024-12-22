from typing import Union, List, Dict
from collections import defaultdict

from torch import Tensor


def convert_tensor(
        ts: Tensor,
        structure: Union[dict, list]
) -> Dict[int, Union[List[int], Dict[int, int]]]:
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
    for (diner_id, reviewer_id) in ts:
        reviewer_id = reviewer_id.item()
        diner_id = diner_id.item()
        if structure == dict:
            res[reviewer_id][diner_id] = 1
        else:
            res[reviewer_id].append(diner_id)
    return res
