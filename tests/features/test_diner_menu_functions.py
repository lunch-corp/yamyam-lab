from collections import Counter
from unittest.mock import Mock, patch

import pandas as pd
import pytest


class TestDinerMenuFunctions:
    """Test class for DinerFeatureStore menu-related functions."""

    @pytest.fixture
    def sample_config(self):
        """Create sample menu configuration for testing."""
        return {
            "menu_cleaning": {
                "stopwords": ["1인분", "세트", "가격표", "추가"],
                "units": ["g", "kg", "개"],
                "patterns": {
                    "remove_parentheses": "\\([^)]*\\)",
                    "keep_korean_and_space": "[^\\uAC00-\\uD7A3\\s]",
                    "remove_multiple_spaces": "\\s+",
                    "remove_units_at_end": "\\d+(?:g|kg|개|mm|ml|L|원)\\s*",
                },
            },
            "menu_filtering": {
                "min_menu_length": 2,
                "default_frequency_threshold": 2,
                "default_max_features_per_category": 50,
                "default_min_multi_hot_freq": 5,
            },
        }

    @pytest.fixture
    def sample_review_data(self):
        """Create sample review data for testing."""
        return pd.DataFrame(
            {
                "diner_idx": [1, 1, 2, 2, 3, 3],
                "reviewer_id": [101, 102, 103, 104, 105, 106],
                "reviewer_review_score": [4.5, 4.0, 3.5, 4.2, 4.8, 3.9],
            }
        )

    @pytest.fixture
    def sample_diner_data(self):
        """Create sample diner data for testing."""
        return pd.DataFrame(
            {
                "diner_idx": [1, 2, 3],
                "diner_name": ["한식당A", "중식당B", "한식당C"],
                "diner_category_large": ["한식", "중식", "한식"],
                "diner_menu_name": [
                    ["김치찌개(1인분)", "불고기 세트", "가격표", "된장찌개"],
                    ["짜장면123", "짬뽕", "탕수육"],
                    ["김치찌개(1인분)", "갈비탕", "된장찌개"],
                ],
                "diner_lat": [37.5665, 37.5651, 37.5672],
                "diner_lon": [126.9780, 126.9895, 126.9836],
            }
        )

    @pytest.fixture
    def diner_feature_store(self, sample_review_data, sample_diner_data):
        """Create DinerFeatureStore instance for testing."""
        # Mock h3 imports and config loading
        with patch.dict("sys.modules", {"tools": Mock(), "tools.h3": Mock()}):
            # Import after mocking
            from src.yamyam_lab.features.diner import DinerFeatureStore

            # Mock the config loading
            def mock_load_config(self):
                return {
                    "menu_cleaning": {
                        "stopwords": ["1인분", "세트", "가격표", "추가"],
                        "units": ["g", "kg", "개"],
                        "patterns": {
                            "remove_parentheses": "\\([^)]*\\)",
                            "keep_korean_and_space": "[^\\uAC00-\\uD7A3\\s]",
                            "remove_multiple_spaces": "\\s+",
                            "remove_units_at_end": "\\d+(?:g|kg|개|mm|ml|L|원)\\s*",
                        },
                    },
                    "menu_filtering": {
                        "min_menu_length": 1,
                        "default_frequency_threshold": 2,
                        "default_max_features_per_category": 50,
                        "default_min_multi_hot_freq": 5,
                    },
                }

            # Patch the config loading method
            with patch.object(DinerFeatureStore, "_load_menu_config", mock_load_config):
                feature_store = DinerFeatureStore(
                    review=sample_review_data,
                    diner=sample_diner_data,
                    all_diner_ids=[1, 2, 3],
                    feature_param_pair={},
                )
                return feature_store

    def test_clean_single_menu_name(self, diner_feature_store):
        """Test _clean_single_menu_name function."""
        # Test normal cases
        assert (
            diner_feature_store._clean_single_menu_name("김치찌개(1인분)") == "김치찌개"
        )
        assert diner_feature_store._clean_single_menu_name("불고기 세트") == "불고기"
        assert diner_feature_store._clean_single_menu_name("짜장면123") == "짜장면"

        # Test stopwords removal
        assert diner_feature_store._clean_single_menu_name("가격표") is None
        assert diner_feature_store._clean_single_menu_name("추가메뉴") == "메뉴"

        # Test edge cases
        assert diner_feature_store._clean_single_menu_name(None) is None
        assert diner_feature_store._clean_single_menu_name("") is None
        assert diner_feature_store._clean_single_menu_name("a") is None  # Too short

    def test_clean_menu_list(self, diner_feature_store):
        """Test _clean_menu_list function."""
        # Test list input
        menu_list = ["김치찌개(1인분)", "불고기 세트", "가격표"]
        result = diner_feature_store._clean_menu_list(menu_list)
        expected = ["김치찌개", "불고기"]
        assert result == expected

        # Test string input (should be parsed as list)
        menu_string = "['짜장면123', '짬뽕', '가격표']"
        result = diner_feature_store._clean_menu_list(menu_string)
        expected = ["짜장면", "짬뽕"]
        assert result == expected

        # Test edge cases
        assert diner_feature_store._clean_menu_list([]) == []
        assert diner_feature_store._clean_menu_list("invalid_string") == []
        assert diner_feature_store._clean_menu_list(None) == []

    def test_collect_menus_by_category(self, diner_feature_store):
        """Test _collect_menus_by_category function."""
        # Prepare test data
        test_diner = pd.DataFrame(
            {
                "diner_category_large": ["한식", "한식", "중식"],
                "cleaned_menu_list": [
                    ["김치찌개", "불고기"],
                    ["김치찌개", "된장찌개"],
                    ["짜장면", "짬뽕"],
                ],
            }
        )
        diner_feature_store.diner = test_diner

        result = diner_feature_store._collect_menus_by_category("diner_category_large")

        # Check results
        assert "한식" in result
        assert "중식" in result
        assert result["한식"]["김치찌개"] == 2
        assert result["한식"]["불고기"] == 1
        assert result["한식"]["된장찌개"] == 1
        assert result["중식"]["짜장면"] == 1
        assert result["중식"]["짬뽕"] == 1

    def test_select_frequent_menus_per_category(self, diner_feature_store):
        """Test _select_frequent_menus_per_category function."""
        # Prepare test data
        category_menu_counts = {
            "한식": Counter({"김치찌개": 5, "불고기": 3, "된장찌개": 1}),
            "중식": Counter({"짜장면": 2, "짬뽕": 1}),
        }

        # Test with min_frequency=2, no max_count
        result = diner_feature_store._select_frequent_menus_per_category(
            category_menu_counts, min_frequency=2
        )
        assert "김치찌개" in result["한식"]
        assert "불고기" in result["한식"]
        assert "된장찌개" not in result["한식"]
        assert "짜장면" in result["중식"]
        assert "짬뽕" not in result["중식"]

        # Test with max_count=1
        result = diner_feature_store._select_frequent_menus_per_category(
            category_menu_counts, min_frequency=1, max_count=1
        )
        assert len(result["한식"]) == 1
        assert "김치찌개" in result["한식"]  # Highest frequency
        assert len(result["중식"]) == 1
        assert "짜장면" in result["중식"]  # Highest frequency

        # Test return_as_set=True
        result = diner_feature_store._select_frequent_menus_per_category(
            category_menu_counts, min_frequency=2, return_as_set=True
        )
        assert isinstance(result["한식"], set)
        assert isinstance(result["중식"], set)

    def test_apply_valid_menu_filter(self, diner_feature_store):
        """Test _apply_valid_menu_filter function."""
        # Prepare test data
        test_row = pd.Series(
            {
                "diner_category_large": "한식",
                "valid_menu_names": ["김치찌개", "불고기", "피자", "된장찌개"],
            }
        )
        category_valid_menus = {
            "한식": {"김치찌개", "불고기"},
            "중식": {"짜장면", "짬뽕"},
        }

        result = diner_feature_store._apply_valid_menu_filter(
            test_row, "diner_category_large", category_valid_menus
        )

        # Should only return menus that are in the valid set for the category
        assert "김치찌개" in result
        assert "불고기" in result
        assert "피자" not in result
        assert "된장찌개" not in result

        # Test with unknown category
        test_row_unknown = pd.Series(
            {
                "diner_category_large": "일식",
                "valid_menu_names": ["초밥", "라멘"],
            }
        )
        result = diner_feature_store._apply_valid_menu_filter(
            test_row_unknown, "diner_category_large", category_valid_menus
        )
        assert result == []

    def test_create_multi_hot_menu_features(self, diner_feature_store):
        """Test _create_multi_hot_menu_features function."""
        # Prepare test data
        category_top_menus = {"한식": ["김치찌개", "불고기"], "중식": ["짜장면"]}

        test_diner = pd.DataFrame(
            {
                "diner_category_large": ["한식", "한식", "중식"],
                "valid_menu_names": [
                    ["김치찌개", "불고기"],
                    ["김치찌개"],
                    ["짜장면"],
                ],
            }
        )
        diner_feature_store.diner = test_diner

        result_columns = diner_feature_store._create_multi_hot_menu_features(
            category_top_menus, "diner_category_large"
        )

        # Check that columns were created
        expected_columns = [
            "menu_한식_김치찌개",
            "menu_한식_불고기",
            "menu_중식_짜장면",
        ]
        assert set(result_columns) == set(expected_columns)

        # Check that columns exist in dataframe
        for col in expected_columns:
            assert col in diner_feature_store.diner.columns

        # Check values
        assert diner_feature_store.diner.loc[0, "menu_한식_김치찌개"] == 1
        assert diner_feature_store.diner.loc[0, "menu_한식_불고기"] == 1
        assert diner_feature_store.diner.loc[0, "menu_중식_짜장면"] == 0

        assert diner_feature_store.diner.loc[1, "menu_한식_김치찌개"] == 1
        assert diner_feature_store.diner.loc[1, "menu_한식_불고기"] == 0

        assert diner_feature_store.diner.loc[2, "menu_중식_짜장면"] == 1
        assert diner_feature_store.diner.loc[2, "menu_한식_김치찌개"] == 0


if __name__ == "__main__":
    pytest.main([__file__])
