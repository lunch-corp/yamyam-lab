import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest
from omegaconf import OmegaConf


@dataclass
class DataConfig:
    test_size: float
    min_reviews: int
    features: List[str]
    cat_features: List[str]
    diner_engineered_feature_names: List[str]


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


@pytest.fixture(scope="function")
def setup_config(request):
    model, use_metadata = request.param
    args = argparse.ArgumentParser()
    args.model = model
    args.batch_size = 128
    args.lr = 0.01
    args.regularization = 1e-4
    args.num_factors = 16
    args.patience = 5
    args.epochs = 1
    args.test_ratio = 0.3
    args.embedding_dim = 32
    args.walk_length = 20
    args.walks_per_node = 10
    args.num_negative_samples = 20
    args.p = 1
    args.q = 0.5
    args.result_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"../result/{args.model}"
    )
    args.weighted_edge = True
    args.use_metadata = use_metadata
    args.meta_path = [
        ["user", "diner", "user", "diner", "user", "diner"],
        ["user", "diner", "category", "diner", "user"],
    ]
    args.category_column_for_meta = "diner_category_large"
    args.test = True
    return args


@pytest.fixture(scope="function")
def setup_ranker_config(request) -> TestConfig:
    model, params, epoch = request.param

    test_config = TestConfig(
        data=DataConfig(
            test_size=0.2,
            min_reviews=5,
            features=[
                "badge_level",
                "badge_grade",
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
            diner_engineered_feature_names=[
                "all_review_cnt",
                "diner_review_tags",
                "diner_menu_price",
            ],
        ),
        models=ModelConfig(
            model_path=f"result/{model}/",
            results="ranker",
            name=model,
            params=OmegaConf.create(params),
            num_boost_round=epoch,
            verbose_eval=epoch,
            early_stopping_rounds=1,
            seed=42,
        ),
    )
    return test_config
