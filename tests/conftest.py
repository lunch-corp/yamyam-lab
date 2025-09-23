import argparse
import os

import pytest
from omegaconf import OmegaConf


@pytest.fixture(scope="function")
def setup_data_config(request):
    model, params, epoch = request.param
    config = {
        "data": {
            "test_size": 0.3,
            "min_reviews": 3,
            "num_neg_samples": 0,
            "category_column_for_meta": "diner_category_large",
            "user_engineered_feature_names": {
                "categorical_feature_count": {
                    "categorical_feature_names": ["diner_category_large"]
                },
            },
            "diner_engineered_feature_names": {
                "all_review_cnt": {},
                "diner_review_tags": {},
                "diner_menu_price": {},
            },
            "test": True,
            "random_state": 42,
            "stratify": "reviewer_id",
            "use_unique_mapping_id": False,
            "sampling_type": "random",
            "is_timeseries_by_users": False,
            "is_timeseries_by_time_point": True,
            "train_time_point": "2024-09-01",
            "val_time_point": "2024-12-01",
            "test_time_point": "2025-01-01",
            "end_time_point": "2025-02-01",
            "candidate_type": "node2vec",
        },
        "models": {
            "ranker": {
                "_target_": "src.yamyam_lab.model.rank.boosting.LightGBMTrainer",
                "model_path": f"result/{model}/",
                "results": "ranker",
                "features": [
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
                    "asian",
                    "japanese",
                    "chinese",
                    "korean",
                    "western",
                ],
                "cat_features": ["diner_review_cnt_category"],
                "params": OmegaConf.create(params),
                "num_boost_round": epoch,
                "verbose_eval": epoch,
                "early_stopping_rounds": 1,
                "seed": 42,
            },
        },
        "model_path": "res/models/",
        "results": "lightgbm_ranker",
        "user_name": 3830746302,
        "top_n": 20,
        "user_address": "강남역",
        "distance_threshold": 0.5,
        "diner_category_large": ["한식"],
        "preprocess": {
            "filter": {
                "martial_law_reviews": {
                    "target_months": ["2025-01", "2024-12"],
                    "min_common_word_count_with_abusive_words": 3,
                    "min_review_count_by_diner_id": 3,
                    "included_tags": ["NNG", "NNP"],
                    "abusive_words": [
                        "총",
                        "내란",
                        "공수처",
                        "시위",
                        "좌우",
                        "애국",
                        "정치",
                        "총살",
                        "테러",
                        "민주주의",
                        "윤석열",
                        "총기",
                        "좌파",
                        "우파",
                        "극우",
                        "집회",
                        "계엄",
                    ],
                    "pre_calculated_diner_ids": [
                        20557155,
                        561814157,
                        717255023,
                        1210281986,
                        1210386151,
                        1275807781,
                        1390211388,
                        1420824177,
                        1567102742,
                        1983344097,
                    ],
                }
            }
        },
    }
    return OmegaConf.create(config)


@pytest.fixture(scope="function")
def setup_config(request):
    model, use_metadata = request.param
    args = argparse.ArgumentParser()
    args.model = model
    args.device = "cpu"
    args.batch_size = 128
    args.lr = 0.01
    args.regularization = 1e-4
    args.patience = 5
    args.epochs = 1
    args.test_ratio = 0.3
    args.embedding_dim = 32
    args.walk_length = 20
    args.walks_per_node = 10
    args.num_negative_samples = 20
    args.p = 1
    args.q = 1
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
    args.num_sage_layers = 2
    args.aggregator_funcs = ["mean", "mean"]
    args.num_neighbor_samples = 3
    args.num_lightgcn_layers = 3
    args.drop_ratio = 0.1
    args.test = True
    args.save_candidate = False
    return args


@pytest.fixture(scope="function")
def setup_als_config():
    args = argparse.ArgumentParser()
    args.alpha = 1
    args.factors = 100
    args.regularization = 0.01
    args.iterations = 15
    args.use_gpu = False
    args.calculate_training_loss = True
    args.test = True
    args.save_candidate = False
    return args


@pytest.fixture(scope="function")
def setup_ranker_config(request):
    model, params, epoch = request.param

    config = {
        "data": {
            "test_size": 0.3,
            "min_reviews": 3,
            "num_neg_samples": 0,
            "category_column_for_meta": "diner_category_large",
            "user_engineered_feature_names": {
                "categorical_feature_count": {
                    "categorical_feature_names": ["diner_category_large"]
                },
            },
            "diner_engineered_feature_names": {
                "all_review_cnt": {},
                "diner_review_tags": {},
                "diner_menu_price": {},
            },
            "test": True,
            "random_state": 42,
            "stratify": "reviewer_id",
            "use_unique_mapping_id": False,
            "sampling_type": "random",
            "is_timeseries_by_users": False,
            "is_timeseries_by_time_point": True,
            "train_time_point": "2024-09-01",
            "val_time_point": "2024-12-01",
            "test_time_point": "2025-01-01",
            "end_time_point": "2025-02-01",
            "candidate_type": "node2vec",
        },
        "models": {
            "ranker": {
                "_target_": "src.yamyam_lab.model.rank.boosting.LightGBMTrainer",
                "model_path": f"result/{model}/",
                "results": "ranker",
                "features": [
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
                    "asian",
                    "japanese",
                    "chinese",
                    "korean",
                    "western",
                ],
                "cat_features": ["diner_review_cnt_category"],
                "params": OmegaConf.create(params),
                "num_boost_round": epoch,
                "verbose_eval": epoch,
                "early_stopping_rounds": 1,
                "seed": 42,
            },
        },
        "training": {
            "evaluation": {
                "recommend_batch_size": 1000000,
                "top_k_values_for_pred": [3, 7, 10, 20],
            },
        },
        "preprocess": {
            "filter": {
                "martial_law_reviews": {
                    "target_months": ["2025-01", "2024-12"],
                    "min_common_word_count_with_abusive_words": 3,
                    "min_review_count_by_diner_id": 3,
                    "included_tags": ["NNG", "NNP"],
                    "abusive_words": [
                        "총",
                        "내란",
                        "공수처",
                        "시위",
                        "좌우",
                        "애국",
                        "정치",
                        "총살",
                        "테러",
                        "민주주의",
                        "윤석열",
                        "총기",
                        "좌파",
                        "우파",
                        "극우",
                        "집회",
                        "계엄",
                    ],
                    "pre_calculated_diner_ids": [
                        20557155,
                        561814157,
                        717255023,
                        1210281986,
                        1210386151,
                        1275807781,
                        1390211388,
                        1420824177,
                        1567102742,
                        1983344097,
                    ],
                }
            }
        },
    }
    return OmegaConf.create(config)
