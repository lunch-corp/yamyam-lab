from enum import StrEnum


class TransitionKey(StrEnum):
    """
    Enum used in transition probabilities object.
    """

    FIRST_PROB = "first_prob"
    NEIGHBORS = "neighbors"
    NEXT_PROB = "next_prob"
