from enum import StrEnum


class TransitionKey(StrEnum):
    FIRST_PROB = "first_prob"
    NEIGHBORS = "neighbors"
    NEXT_PROB = "next_prob"