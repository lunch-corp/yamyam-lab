from enum import Enum


class FileName(Enum):
    LOG = "log.log"
    WEIGHT = "weight.pt"
    TRAINING_LOSS = "training_loss.pkl"
    METRIC = "metric.pkl"
    DATA_OBJECT = "data_object.pkl"
