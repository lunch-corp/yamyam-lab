from collections import defaultdict


def convert_tensor(ts, structure):
    """Convert 2 dimensional tensor to dict
    ts: torch.tensor (n x 2)
        Columns should be matched with (diner_id, reviewer_id)
    structure: dict or list
    """
    assert ts.shape[1] == 2
    assert structure in [dict, list]
    res = defaultdict(structure)
    for diner_id, reviewer_id in ts:
        reviewer_id = reviewer_id.item()
        diner_id = diner_id.item()
        if structure == dict:
            res[reviewer_id][diner_id] = 1
        else:
            res[reviewer_id].append(diner_id)
    return res
