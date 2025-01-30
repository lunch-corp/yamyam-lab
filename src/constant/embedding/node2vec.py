from enum import Enum


class TransitionKey(str, Enum):
    """
    Enum used in transition probabilities object.
    """

    FIRST_PROB = "first_prob"
    NEIGHBORS = "neighbors"
    NEXT_PROB = "next_prob"
