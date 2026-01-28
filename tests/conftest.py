import argparse

import pandas as pd
import pytest
from omegaconf import OmegaConf


@pytest.fixture
def mock_diner_with_raw_category():
    """Create mock diner category data."""
    return pd.DataFrame(
        {
            "diner_idx": [
                1,
                2,
                3,
                4,
                5,
                2000734287,
                1244228762,
                1534943798,
                1985593786,
                6,
                7,
                8,
                9,
            ],
            "diner_category_large": [
                "한식",
                "중식",
                "양식",
                "일식",
                "한식",
                "일식",
                "한식",
                "일식",
                "한식",
                "간식",
                "아시아음식",
                "패스트푸드",
                "디저트",
            ],
            "diner_category_middle": [
                "고기",
                "중화요리",
                "이탈리안",
                "초밥",
                "찌개",
                "라멘",
                "국밥",
                "라멘",
                "탕",
                "분식",
                "태국음식",
                "치킨",
                "베이커리",
            ],
            "diner_category_small": [
                "삼겹살",
                "짜장면",
                "파스타",
                "회",
                "김치찌개",
                "이에케라멘",
                "순대국",
                "시오라멘",
                "감자탕",
                "떡볶이",
                "팟타이",
                "치킨",
                "케이크",
            ],
        }
    )


@pytest.fixture
def mock_review():
    """Create mock review data with 20 reviews per month from 2024-09 to 2025-11."""
    import random
    from datetime import datetime

    # Define date range: 2024-09 to 2025-11 (15 months)
    start_date = datetime(2024, 9, 1)
    num_months = 15
    reviews_per_month = 20

    # Diner IDs to cycle through
    diner_ids = [
        1,
        2,
        3,
        4,
        5,
        2000734287,
        1244228762,
        1534943798,
        1985593786,
        6,
        7,
        8,
        9,
    ]

    diner_idx_list = []
    reviewer_id_list = []
    reviewer_review_score_list = []
    reviewer_review_date_list = []

    reviewer_id_counter = 101

    for month_offset in range(num_months):
        # Calculate the current month
        current_month = start_date.month + month_offset
        current_year = start_date.year + (current_month - 1) // 12
        current_month = ((current_month - 1) % 12) + 1

        for review_idx in range(reviews_per_month):
            # Cycle through diners
            diner_id = diner_ids[review_idx % len(diner_ids)]

            # Generate reviewer ID
            reviewer_id = reviewer_id_counter
            reviewer_id_counter += 1

            # Generate random score between 3.0 and 5.0
            score = round(random.uniform(3.0, 5.0), 1)

            # Generate random day and time within the month
            day = random.randint(1, 28)  # Keep it safe for all months
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)

            review_date = datetime(current_year, current_month, day, hour, minute)

            diner_idx_list.append(diner_id)
            reviewer_id_list.append(reviewer_id)
            reviewer_review_score_list.append(score)
            reviewer_review_date_list.append(review_date.strftime("%Y-%m-%d %H:%M:%S"))

    return pd.DataFrame(
        {
            "diner_idx": diner_idx_list,
            "reviewer_id": reviewer_id_list,
            "reviewer_review_score": reviewer_review_score_list,
            "reviewer_review_date": reviewer_review_date_list,
        }
    )


@pytest.fixture
def mock_diner():
    """Create mock diner data."""
    return pd.DataFrame(
        {
            "diner_idx": [
                1,
                2,
                3,
                4,
                5,
                2000734287,
                1244228762,
                1534943798,
                1985593786,
                6,
                7,
                8,
                9,
            ],
            "diner_name": [
                "삼겹살집",
                "중화요리",
                "이탈리안레스토랑",
                "초밥전문점",
                "김치찌개",
                "이에케라멘",
                "순대국밥",
                "시오라멘",
                "감자탕",
                "떡볶이집",
                "태국음식점",
                "치킨집",
                "베이커리카페",
            ],
            "diner_review_avg": [
                4.5,
                3.75,
                3.75,
                4.25,
                4.0,
                4.8,
                4.6,
                3.5,
                4.9,
                4.25,
                4.5,
                3.5,
                4.45,
            ],
            "diner_review_cnt": [4, 4, 4, 4, 4, 5, 4, 3, 3, 4, 4, 4, 4],
            "blog_review_cnt": [10, 8, 12, 15, 9, 20, 18, 25, 16, 7, 9, 11, 13],
            "bayesian_score": [
                2.8,
                2.3,
                2.4,
                2.7,
                2.5,
                2.9,
                2.8,
                2.2,
                2.85,
                2.6,
                2.7,
                2.1,
                2.65,
            ],
            "diner_lat": [
                37.5665,
                37.5172,
                37.4979,
                37.5145,
                37.5511,
                37.5233,
                37.5651,
                37.5789,
                37.5442,
                37.5321,
                37.5555,
                37.5678,
                37.5234,
            ],
            "diner_lon": [
                126.9780,
                127.0473,
                127.0276,
                127.0357,
                126.9882,
                127.0421,
                126.9996,
                127.0012,
                126.9707,
                126.9543,
                127.0234,
                127.0567,
                126.9876,
            ],
            "diner_review_tags": [
                ["맛있다", "친절하다"],
                ["특별하다", "양많다"],
                ["분위기좋다", "깔끔하다"],
                ["신선하다", "고급스럽다"],
                ["푸짐하다", "따뜻하다"],
                ["진하다", "맛집", "일본"],
                ["국물진하다", "고기많다", "저렴하다"],
                ["깔끔하다", "미슐랭"],
                ["감자탕", "뼈해장국", "진하다"],
                ["맛있다", "저렴하다"],
                ["이국적", "향긋하다"],
                ["바삭하다", "빠르다"],
                ["달콤하다", "예쁘다"],
            ],
            "diner_menu_name": [
                ["삼겹살", "목살", "항정살"],
                ["짜장면", "짬뽕", "탕수육"],
                ["까르보나라", "알리오올리오", "피자"],
                ["초밥세트", "회덮밥", "사시미"],
                ["김치찌개", "된장찌개", "순두부찌개"],
                ["이에케라멘", "차슈라멘", "교자"],
                ["순대국", "내장국", "수육"],
                ["시오라멘", "쇼유라멘"],
                ["감자탕", "뼈해장국", "수육"],
                ["떡볶이", "순대", "튀김"],
                ["팟타이", "똠얌꿍", "푸팟퐁커리"],
                ["후라이드", "양념치킨"],
                ["케이크", "마카롱", "쿠키"],
            ],
            "diner_menu_price": [
                [15000, 14000, 16000],
                [6000, 7000, 20000],
                [12000, 11000, 18000],
                [25000, 13000, 35000],
                [8000, 8000, 8000],
                [11000, 13000, 6000],
                [9000, 10000, 25000],
                [10000, 11000],
                [12000, 11000, 28000],
                [4000, 3500, 2000],
                [12000, 13000, 15000],
                [18000, 19000],
                [35000, 3000, 2500],
            ],
        }
    )


@pytest.fixture
def mock_load_dataset(mock_review, mock_diner, mock_diner_with_raw_category):
    """Fixture that returns the mock dataset tuple directly."""
    return mock_review.copy(), mock_diner.copy(), mock_diner_with_raw_category.copy()


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
                "_target_": "yamyam_lab.model.rank.boosting.LightGBMTrainer",
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
                    "score",
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
    args.result_path = None
    args.config_root_path = None
    args.num_workers = 1
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
    args.postfix = "pytest"
    return args


@pytest.fixture(scope="function")
def setup_als_config():
    args = argparse.ArgumentParser()
    args.model = "als"
    args.alpha = 1
    args.factors = 100
    args.regularization = 0.01
    args.iterations = 15
    args.use_gpu = False
    args.calculate_training_loss = True
    args.test = True
    args.save_candidate = False
    args.config_root_path = None
    args.postfix = "pytest"
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
        "log": {
            "experiment_name": "ranker_experiments",
            "run_name": "lightgbm_ranker_v1",
            "enable_mlflow": False,
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
                    "score",
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
