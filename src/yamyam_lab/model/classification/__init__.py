"""Classification models for category prediction."""

from yamyam_lab.model.classification.base_classifier import BaseClassifier
from yamyam_lab.model.classification.catboost_classifier import (
    CatBoostCategoryClassifier,
)

__all__ = ["BaseClassifier", "CatBoostCategoryClassifier"]
