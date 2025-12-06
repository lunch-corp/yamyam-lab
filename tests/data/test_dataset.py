try:
    import os
    import sys

    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src")
    )

except ModuleNotFoundError:
    raise Exception("Module not found")

import os
import tempfile
import traceback
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from yamyam_lab.data.base import BaseDatasetLoader
from yamyam_lab.data.config import DataConfig
from yamyam_lab.data.ranker import load_test_dataset
from yamyam_lab.tools.utils import get_kakao_lat_lng


def _create_dummy_data():
    """더미 데이터 생성 함수"""
    np.random.seed(42)
    n_reviews = 1000
    n_diners = 100
    n_reviewers = 50

    # review 데이터 (ReviewSchema에 필요한 모든 컬럼 포함)
    review = pd.DataFrame(
        {
            "reviewer_id": np.random.randint(1, n_reviewers + 1, n_reviews),
            "diner_idx": np.random.randint(1, n_diners + 1, n_reviews),
            "review_id": range(1, n_reviews + 1),  # unique, nullable=False
            "reviewer_review": ["test review"] * n_reviews,
            "reviewer_review_date": pd.date_range(
                "2024-01-01", periods=n_reviews, freq="D"
            )[:n_reviews].astype(str),
            "reviewer_review_score": np.random.uniform(1.0, 5.0, n_reviews),
        }
    )

    # reviewer 데이터 (ReviewerSchema에 필요한 컬럼 포함)
    reviewer = pd.DataFrame(
        {
            "reviewer_id": range(1, n_reviewers + 1),
            "reviewer_user_name": [f"user_{i}" for i in range(1, n_reviewers + 1)],
            "reviewer_avg": np.random.uniform(1.0, 5.0, n_reviewers),
            "badge_grade": np.random.choice(["bronze", "silver", "gold"], n_reviewers),
            "badge_level": np.random.randint(1, 10, n_reviewers),
        }
    )

    # diner 데이터 (DinerSchema에 필요한 주요 컬럼 포함)
    diner = pd.DataFrame(
        {
            "diner_idx": range(1, n_diners + 1),
            "diner_name": [f"restaurant_{i}" for i in range(1, n_diners + 1)],
            "diner_tag": [["tag1", "tag2"]] * n_diners,
            "diner_menu_name": [["menu1", "menu2"]] * n_diners,
            "diner_menu_price": [[10000, 15000]] * n_diners,
            "diner_review_cnt": np.random.randint(10, 1000, n_diners),
            "diner_review_avg": np.random.uniform(1.0, 5.0, n_diners),
            "diner_blog_review_cnt": np.random.uniform(0, 100, n_diners),
            "diner_review_tags": [["taste", "kind"]] * n_diners,
            "diner_road_address": ["서울 용산구"] * n_diners,
            "diner_num_address": [f"address_{i}" for i in range(1, n_diners + 1)],
            "diner_phone": [f"02-1234-{i:04d}" for i in range(1, n_diners + 1)],
            "diner_lat": np.random.uniform(37.5, 37.6, n_diners),
            "diner_lon": np.random.uniform(126.9, 127.0, n_diners),
            "diner_open_time": ["09:00-22:00"] * n_diners,
            "diner_open_time_titles": [["월", "화", "수"]] * n_diners,
            "diner_open_time_hours": [["09:00-22:00"]] * n_diners,
            "diner_open_time_off_days_title": [[]] * n_diners,
            "diner_open_time_off_days_hours": [[]] * n_diners,
            "bayesian_score": np.random.uniform(1.0, 5.0, n_diners),
        }
    )

    # category 데이터 (CategorySchema에 필요한 컬럼 포함)
    categories = [
        "한식",
        "중식",
        "양식",
        "일식",
        "아시안",
        "패스트푸드",
        "치킨",
        "술집",
    ]
    category = pd.DataFrame(
        {
            "diner_idx": range(1, n_diners + 1),
            "industry_category": np.random.choice(["음식점", "카페"], n_diners),
            "diner_category_large": np.random.choice(categories, n_diners),
            "diner_category_middle": [f"middle_{i}" for i in range(1, n_diners + 1)],
            "diner_category_small": [f"small_{i}" for i in range(1, n_diners + 1)],
        }
    )

    return review, reviewer, diner, category


def _mock_read_csv(filepath_or_buffer, **kwargs):
    """pd.read_csv를 mock하는 함수"""
    review, reviewer, diner, category = _create_dummy_data()

    # 파일 경로에 따라 적절한 데이터 반환
    filepath = (
        str(filepath_or_buffer)
        if hasattr(filepath_or_buffer, "__str__")
        else filepath_or_buffer
    )

    if "review.csv" in filepath or (
        "review" in filepath.lower() and "reviewer" not in filepath.lower()
    ):
        return review
    elif "reviewer.csv" in filepath or "reviewer" in filepath.lower():
        return reviewer
    elif "diner.csv" in filepath or (
        "diner" in filepath.lower() and "category" not in filepath.lower()
    ):
        return diner
    elif "category.csv" in filepath or "category" in filepath.lower():
        return category
    else:
        # 기본값으로 review 반환
        return review


def _mock_check_data_and_return_paths():
    """check_data_and_return_paths를 mock하는 함수"""
    # 임시 디렉토리 생성 (실제 파일은 생성하지 않음, pd.read_csv가 mock됨)
    temp_dir = tempfile.mkdtemp()

    return {
        "review": os.path.join(temp_dir, "review.csv"),
        "reviewer": os.path.join(temp_dir, "reviewer.csv"),
        "diner": os.path.join(temp_dir, "diner.csv"),
        "category": os.path.join(temp_dir, "category.csv"),
    }


def _mock_build_feature(review, diner, all_user_ids, all_diner_ids, **kwargs):
    """build_feature를 mock하는 함수 - 필요한 컬럼들을 포함한 더미 feature 반환"""
    np.random.seed(42)

    # user_feature 생성 (필요한 컬럼들 포함)
    user_feature = pd.DataFrame({"reviewer_id": sorted(all_user_ids)})
    # categorical_feature_count로 생성되는 컬럼들
    for col in [
        "asian",
        "japanese",
        "chinese",
        "korean",
        "western",
        "snack",
        "fastfood",
        "dessert",
        "cafe",
    ]:
        user_feature[col] = np.random.randint(0, 10, len(user_feature))

    # diner_feature 생성 (필요한 컬럼들 포함)
    diner_feature = pd.DataFrame({"diner_idx": sorted(all_diner_ids)})
    # diner_review_tags로 생성되는 컬럼들
    for col in [
        "diner_review_cnt_category",
        "taste",
        "kind",
        "mood",
        "chip",
        "parking",
    ]:
        diner_feature[col] = np.random.uniform(0, 5, len(diner_feature))
    # diner_menu_price로 생성되는 컬럼들
    for col in ["min_price", "max_price", "mean_price", "median_price", "menu_count"]:
        diner_feature[col] = np.random.uniform(1000, 50000, len(diner_feature))

    # diner_meta_feature 생성
    diner_meta_feature = pd.DataFrame({"diner_idx": sorted(all_diner_ids)})

    return user_feature, diner_feature, diner_meta_feature


@patch("yamyam_lab.data.base.build_feature")
@patch("yamyam_lab.data.base.load_yaml")
@patch("yamyam_lab.tools.config.load_yaml")
@patch("pandas.read_csv")
@patch("yamyam_lab.tools.google_drive.check_data_and_return_paths")
@patch("yamyam_lab.data.base.check_data_and_return_paths")
def test_loader_dataset(
    mock_base_check,
    mock_drive_check,
    mock_read_csv,
    mock_config_load_yaml,
    mock_base_load_yaml,
    mock_build_feature,
):
    """더미 데이터를 사용하여 테스트 통과"""
    # mock 설정
    mock_drive_check.return_value = _mock_check_data_and_return_paths()
    mock_base_check.return_value = _mock_check_data_and_return_paths()
    mock_read_csv.side_effect = _mock_read_csv
    mock_config_load_yaml.return_value = {}  # additional_reviews.yaml을 빈 딕셔너리로 mock
    mock_base_load_yaml.return_value = {}  # additional_reviews.yaml을 빈 딕셔너리로 mock
    mock_build_feature.side_effect = _mock_build_feature  # build_feature를 mock

    try:
        data_loader = BaseDatasetLoader(
            data_config=DataConfig(
                X_columns=["diner_idx", "reviewer_id"],
                y_columns=["reviewer_review_score"],
                user_engineered_feature_names={
                    "categorical_feature_count": {
                        "categorical_feature_names": ["diner_category_large"]
                    }
                },
                diner_engineered_feature_names={
                    "all_review_cnt": {},
                    "diner_review_tags": {},
                    "diner_menu_price": {},
                },
                is_timeseries_by_time_point=True,
                train_time_point="2024-09-01",
                val_time_point="2024-12-01",
                test_time_point="2025-01-01",
                end_time_point="2025-02-01",
                random_state=42,
                stratify="reviewer_id",
                test=True,
                candidate_type="node2vec",
            ),
        )
        data = data_loader.prepare_train_val_dataset(is_tensor=True)

        assert data["train"].shape[0] > 0
        assert data["val"].shape[0] > 0
        assert data["test"].shape[0] > 0
        assert data["user_feature"].shape[0] > 0
        assert data["diner_feature"].shape[0] > 0
        assert data["diner_meta_feature"].shape[0] > 0
        assert data["mapped_res"] is not None

    except Exception as e:
        # 에러 정보 출력
        print(f"Test failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise


@pytest.mark.parametrize(
    "setup_data_config", [("lightgbm", {}, 1)], indirect=["setup_data_config"]
)
@patch("yamyam_lab.data.base.build_feature")
@patch("yamyam_lab.data.base.load_yaml")
@patch("yamyam_lab.tools.config.load_yaml")
@patch("pandas.read_csv")
@patch("yamyam_lab.tools.google_drive.check_data_and_return_paths")
@patch("yamyam_lab.data.base.check_data_and_return_paths")
def test_load_test_dataset(
    mock_base_check,
    mock_drive_check,
    mock_read_csv,
    mock_config_load_yaml,
    mock_base_load_yaml,
    mock_build_feature,
    setup_data_config,
):
    """더미 데이터를 사용하여 테스트 통과"""
    # mock 설정
    mock_drive_check.return_value = _mock_check_data_and_return_paths()
    mock_base_check.return_value = _mock_check_data_and_return_paths()
    mock_read_csv.side_effect = _mock_read_csv
    mock_config_load_yaml.return_value = {}  # additional_reviews.yaml을 빈 딕셔너리로 mock
    mock_base_load_yaml.return_value = {}  # additional_reviews.yaml을 빈 딕셔너리로 mock
    mock_build_feature.side_effect = _mock_build_feature  # build_feature를 mock

    try:
        test = load_test_dataset(setup_data_config)
        assert test is not None
        assert len(test) > 0
        assert isinstance(test, pd.DataFrame)
    except Exception as e:
        # 에러 정보 출력
        print(f"Test failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")


def test_get_kakao_lat_lng():
    location = get_kakao_lat_lng("서울 강남구 강남대로 324")
    assert location is not None
    assert location["lat"] is not None
    assert location["lng"] is not None
