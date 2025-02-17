from enum import Enum


class TransitionKeyMetaPath(str, Enum):
    """
    Enum used in transition probabilities object.
    """

    NEIGHBORS = "neighbors"
    PROB = "prob"
