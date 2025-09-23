from enum import StrEnum


class TransitionKeyMetaPath(StrEnum):
    """
    Enum used in transition probabilities object.
    """

    NEIGHBORS = "neighbors"
    PROB = "prob"
