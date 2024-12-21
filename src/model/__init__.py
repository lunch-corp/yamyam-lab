from omegaconf import DictConfig

from .base import *
from .boosting import *


def build_model(cfg: DictConfig):
    model_type = {"lightgbm": LightGBMTrainer(cfg), "xgboost": XGBoostTrainer(cfg)}

    if trainer := model_type.get(cfg.models.name):
        return trainer

    else:
        raise NotImplementedError(f"Model '{cfg.models.name}' is not implemented.")
