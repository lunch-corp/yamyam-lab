from omegaconf import DictConfig

from .base import BaseModel
from .boosting import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer

__all__ = [
    "BaseModel",
    "CatBoostTrainer",
    "LightGBMTrainer",
    "XGBoostTrainer",
    "build_model",
]


def build_model(cfg: DictConfig):
    model_type = {
        "lightgbm": LightGBMTrainer(cfg),
        "xgboost": XGBoostTrainer(cfg),
        "catboost": CatBoostTrainer(cfg),
    }

    if trainer := model_type.get(cfg.models.name):
        return trainer

    else:
        raise NotImplementedError(f"Model '{cfg.models.name}' is not implemented.")
