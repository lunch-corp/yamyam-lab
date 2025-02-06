import torch
from torch import Tensor


def unpad_by_mask(padded_tensor: Tensor, padding_value: int) -> Tensor:
    """
    Remove padding values from given tensor and stck them.

    Args:
        padded_tensor (Tensor): Padded tensor with `padding_value`. Note that to stck these tensors,
            each tensor should have identical length after unpadding.
        padding_value (int): Integer value used when padding tensor.

    Returns (Tensor):
        Unpadded tensor.
    """
    mask = padded_tensor != padding_value
    return torch.stack([seq[mask[i]] for i, seq in enumerate(padded_tensor)])
