from enum import StrEnum


class Metric(StrEnum):
    """
    Enum for metric when there are no candidates
    """

    AP = "ap"
    MAP = "map"
    NDCG = "ndcg"
    RECALL = "recall"
    COUNT = "count"


class NearCandidateMetric(StrEnum):
    """
    Enum for metric when after near candidates filtering
    """

    RANKED_PREC = "ranked_prec"
    NEAR_RECALL = "near_recall"
    RANKED_PREC_COUNT = "ranked_prec_count"
    RECALL_COUNT = "recall_count"
