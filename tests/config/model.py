from dataclasses import dataclass
from typing import Any, Dict, List

from omegaconf import OmegaConf


@dataclass
class DataConfig:
    test_size: float
    min_reviews: int
    features: List[str]
    cat_features: List[str]


@dataclass
class ModelConfig:
    model_path: str
    results: str
    name: str
    num_boost_round: int
    verbose_eval: int
    early_stopping_rounds: int
    params: Dict[str, Any]
    seed: int = 42


@dataclass
class TestConfig:
    __test__ = False
    data: DataConfig
    models: ModelConfig


TEST_CONFIG = TestConfig(
    data=DataConfig(
        test_size=0.2,
        min_reviews=5,
        features=[
            "badge_level",
            "badge_grade",
            "date_scaled",
            "date_weight",
            "diner_review_cnt_category",
            "min_price",
            "max_price",
            "mean_price",
            "median_price",
            "menu_count",
            "taste",
            "kind",
            "mood",
            "chip",
            "parking",
        ],
        cat_features=["diner_review_cnt_category", "badge_level", "badge_grade"],
    ),
    models=ModelConfig(
        model_path="res/models/",
        results="ranker",
        name="lightgbm",
        params=OmegaConf.create(
            {
                "objective": "lambdarank",
                "boosting_type": "gbdt",
                "metric": "ndcg",
                "num_leaves": 16,
                "learning_rate": 0.1,
            }
        ),
        num_boost_round=1,
        verbose_eval=1,
        early_stopping_rounds=1,
        seed=42,
    ),
)
