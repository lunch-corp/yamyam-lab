"""Loss functions for embedding learning."""

from yamyam_lab.loss.infonce import infonce_loss_with_multiple_negatives
from yamyam_lab.loss.triplet import (
    batch_hard_triplet_loss,
    triplet_margin_loss,
    triplet_margin_loss_with_category,
    triplet_margin_loss_with_multiple_negatives,
)

__all__ = [
    "infonce_loss_with_multiple_negatives",
    "triplet_margin_loss",
    "triplet_margin_loss_with_category",
    "triplet_margin_loss_with_multiple_negatives",
    "batch_hard_triplet_loss",
]
