import ast
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Self, Union

import numpy as np
import pandas as pd

from tools.h3 import get_h3_index, get_hexagon_neighbors

from .base import BaseFeatureStore


class DinerFeatureStore(BaseFeatureStore):
    """
    Feature engineering class for diner (restaurant) data.

    This class provides various feature engineering methods for restaurant data,
    including review statistics, menu analysis, location-based features, and more.
    """

    def __init__(
        self: Self,
        review: pd.DataFrame,
        diner: pd.DataFrame,
        all_diner_ids: List[int],
        feature_param_pair: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Initialize DinerFeatureStore with data and feature configurations.

        Args:
            review: Review data which will be used as train dataset.
            diner: Diner (restaurant) data.
            all_diner_ids: Diner IDs from all review data (train, val, test).
            feature_param_pair: Dictionary mapping feature names to their parameters.
                Keys are feature names, values are parameter dictionaries.

        Raises:
            ValueError: If a feature name in feature_param_pair is not implemented.
        """
        super().__init__(
            review=review,
            diner=diner,
            feature_param_pair=feature_param_pair,
        )
        self.all_diner_ids = all_diner_ids

        self.feature_methods = {
            "all_review_cnt": self.calculate_all_review_cnt,
            "diner_review_tags": self.calculate_diner_score,
            "diner_menu_price": self.calculate_diner_price,
            "diner_mean_review_score": self.calculate_diner_mean_review_score,
            "one_hot_encoding_categorical_features": self.one_hot_encoding_categorical_features,
            "diner_category_meta_combined_with_h3": self.make_diner_category_meta_combined_with_h3,
            "valid_menu_names": self.generate_valid_menus_by_freq,
            # "bayesian_score": self.calculate_bayesian_score,
        }

        for feat, arg in feature_param_pair.items():
            if feat not in self.feature_methods.keys():
                raise ValueError(f"{feat} not matched with implemented method")

        self.engineered_feature_names = ["diner_idx"]
        self.engineered_meta_feature_names = ["diner_idx"]

    def make_features(self: Self) -> None:
        """
        Feature engineer using `self.features`.
        """
        for feat, params in self.feature_param_pair.items():
            self.feature_methods[feat](**params)

    def calculate_all_review_cnt(self: Self, **kwargs) -> None:
        """
        Calculate number of review counts for each diner.
        """
        diner_idx2review_cnt = self.review["diner_idx"].value_counts().to_dict()
        self.diner["all_review_cnt"] = (
            self.diner["diner_idx"].map(diner_idx2review_cnt).fillna(0)
        )
        self.engineered_feature_names.append("all_review_cnt")

    def calculate_diner_score(self: Self, **kwargs) -> None:
        """
        Add categorical and statistical features to the diner dataset.
        """
        bins = [-1, 0, 10, 50, 200, float("inf")]
        self.diner["diner_review_cnt_category"] = (
            pd.cut(self.diner["all_review_cnt"], bins=bins, labels=False)
            .fillna(0)
            .astype(int)
        )

        # Categories for extracting scores
        tag_categories = [
            ("맛", "taste"),
            ("친절", "kind"),
            ("분위기", "mood"),
            ("가성비", "chip"),
            ("주차", "parking"),
        ]

        scores = self._extract_scores_array(
            self.diner["diner_review_tags"].to_list(), tag_categories
        )

        # 결과를 DataFrame으로 변환 및 병합
        self.diner[["taste", "kind", "mood", "chip", "parking"]] = scores
        self.engineered_feature_names.extend(
            ["diner_review_cnt_category", "taste", "kind", "mood", "chip", "parking"]
        )

    def calculate_diner_price(self: Self, **kwargs) -> None:
        """
        Add statistical features to the diner dataset.
        """
        # 새 컬럼으로 추가 (최소값, 최대값, 평균, 중앙값, 항목 수)
        self.diner[
            ["min_price", "max_price", "mean_price", "median_price", "menu_count"]
        ] = self.diner["diner_menu_price"].apply(lambda x: self._extract_statistics(x))

        for col in [
            "min_price",
            "max_price",
            "mean_price",
            "median_price",
            "menu_count",
        ]:
            self.diner[col] = self.diner[col].fillna(self.diner[col].median())

        self.engineered_feature_names.extend(
            ["min_price", "max_price", "mean_price", "median_price", "menu_count"]
        )

    def calculate_diner_mean_review_score(self: Self, **kwargs) -> None:
        """
        Calculate mean review score for each diner from review data.

        Adds 'mean_review_score' column to the diner DataFrame.
        """
        diner_id2score = (
            self.review.groupby("diner_idx")["reviewer_review_score"].mean().to_dict()
        )
        # diners that do not have any reviews
        diner_id_not_exists = set(self.diner["diner_idx"].unique()) - set(
            self.review["diner_idx"].unique()
        )
        for diner_id in diner_id_not_exists:
            # processing null values as zero could trigger bias
            # because it treats diners to have lowest review scores
            diner_id2score[diner_id] = 0
        self.diner["mean_review_score"] = self.diner["diner_idx"].map(diner_id2score)

        self.engineered_feature_names.append("mean_review_score")

    def one_hot_encoding_categorical_features(
        self: Self,
        categorical_feature_name: List[str],
        drop_first: bool = False,
        **kwargs,
    ) -> None:
        """
        One hot encoding categorical features.
        This method converts a categorical feature with C categories into one-hot encoded C dimensional features.
        Depending on the type of algorithm, C-1 dimensional features could be required.

        Args:
            categorical_feature_name (List[str]): List of categorical feature names.
            drop_first (bool): Whether using C-1 columns or not. When set as False, uses C columns.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If a feature name is not found in the diner DataFrame.
        """
        for feature_name in categorical_feature_name:
            if feature_name not in self.diner.columns:
                raise ValueError(f"{feature_name} not in diner data")
            one_hot_encoding_feat = pd.get_dummies(
                self.diner[feature_name],
                prefix=feature_name,
                drop_first=drop_first,
            ).astype(int)
            self.diner = pd.concat([self.diner, one_hot_encoding_feat], axis=1)

            self.engineered_feature_names.extend(list(one_hot_encoding_feat.columns))

    def make_diner_category_meta_combined_with_h3(
        self: Self,
        category_column_for_meta: str,
        h3_resolution: int,
        **kwargs,
    ) -> None:
        """
        Generates node meta combining category column and h3 index.
        Here, h3 index indicates hexagon id where diner locates offered by uber.

        Example of this fe
        When set as
        - category_column_for_meta: diner_category_middle
        - h3_resolution: 9
        two features are generated.
        - metadata_id: `치킨_3ffafda3123`
        - metadata_id_neighbors: [`치킨_3ffazxv78`, `치킨_3ffaqcz511`, `치킨_3ffavnzx321`]

        For each diner, metas like `치킨_3ffafda3123` will be generated where `치킨` is diner_category_middle
        and `3ffafda3123` is h3 index for that diner.
        Also, this function generates metadata for neighboring hexagon.

        Args:
            category_column_for_meta (str): Categorical column name combined with h3 index.
            h3_resolution (int): Resolution value for h3 index. Large values creates smaller hexagon.
            **kwargs: Additional keyword arguments.
        """
        # get diner's h3_index
        self.diner["h3_index"] = self.diner.apply(
            lambda row: get_h3_index(row["diner_lat"], row["diner_lon"], h3_resolution),
            axis=1,
        )
        # get h3_index neighboring with diner's h3_index and concat with meta field
        self.diner["metadata_id_neighbors"] = self.diner.apply(
            lambda row: [
                row[category_column_for_meta] + "_" + h3_index
                for h3_index in get_hexagon_neighbors(row["h3_index"], k=1)
            ],
            axis=1,
        )
        # get current h3_index and concat with meta field
        self.diner["metadata_id"] = self.diner.apply(
            lambda row: row[category_column_for_meta] + "_" + row["h3_index"], axis=1
        )
        self.engineered_meta_feature_names.extend(
            ["metadata_id", "metadata_id_neighbors"]
        )

    # NaN 또는 빈 리스트를 처리할 수 있도록 정의
    def _extract_statistics(self: Self, prices: str) -> pd.Series:
        if not prices or any(pd.isna(prices)):  # 빈 리스트라면 NaN 반환
            return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])

        # 문자열을 리스트로 변환, 이 부분은 데이터 검증 과정에서 처리할 필요가 있어보입니다.
        # 추후에 데이터 검증 코드 완성되면 이 부분은 수정이 필요할 것 같습니다.
        # when prices do not include pure float, such as `변동가격`,
        # float(price) raises error
        # todo: preprocess null value

        prices = [float(price) for price in prices if price not in ["변동가격"]]

        if not prices:  # 변동가격만 존재하는 경우
            return pd.Series([np.nan, np.nan, np.nan, np.nan, 0])

        return pd.Series(
            [
                min(prices),
                max(prices),
                np.nanmean(prices),
                np.median(prices),
                len(prices),
            ]
        )

    # numpy 기반으로 점수 추출 최적화
    def _extract_scores_array(
        self: Self, reviews: list[str], categories: list[tuple[str, str]]
    ) -> np.ndarray:
        # 카테고리 인덱스 매핑
        category_map = {cat: idx for idx, (cat, _) in enumerate(categories)}

        # 결과 배열 초기화
        scores = np.zeros((len(reviews), len(categories)), dtype=int)

        # 리뷰 파싱 후 벡터화
        for i, review in enumerate(reviews):
            if any(pd.isna(review)):  # 결측치 예외 처리
                continue
            try:
                for info in review:
                    cat, score = ast.literal_eval(info)

                    if cat in category_map:
                        scores[i, category_map[cat]] = score

            except (SyntaxError, ValueError, TypeError):
                continue  # 파싱 에러 방지

        return scores

    def generate_valid_menus_by_freq(
        self: Self,
        menu_count_threshold: int = 2,
        category_column: str = "diner_category_large",
        create_multi_hot: bool = False,
        min_menu_freq: int = 5,
        max_features_per_category: int = 50,
        **kwargs,
    ) -> None:
        """
        Clean menu names, filter by frequency threshold per category, and optionally create multi-hot features.

        Args:
            menu_count_threshold: Minimum count threshold for menu items per category.
            category_column: Category column name to group by.
            create_multi_hot: Whether to create multi-hot encoding features.
            min_menu_freq: Minimum frequency for multi-hot features (only used if create_multi_hot=True).
            max_features_per_category: Maximum number of menu features per category (only used if create_multi_hot=True).
            **kwargs: Additional keyword arguments.
        """
        # Clean menu lists
        self.diner["cleaned_menu_list"] = self.diner["diner_menu_name"].apply(
            self._clean_menu_list
        )

        # Count menus by category
        category_menu_counts = self._collect_menus_by_category(
            category_column, "cleaned_menu_list"
        )

        # Create category-specific valid menu sets
        category_valid_menus = self._filter_menus_by_frequency(
            category_menu_counts, menu_count_threshold, return_as_set=True
        )

        # Filter menu lists based on category-specific thresholds
        self.diner["valid_menu_names"] = self.diner.apply(
            lambda row: self._filter_menus_by_category_threshold(
                row, category_column, category_valid_menus
            ),
            axis=1,
        )

        # Clean up temporary column
        self.diner.drop("cleaned_menu_list", axis=1, inplace=True)
        self.engineered_feature_names.append("valid_menu_names")

        # Optionally create multi-hot encoding features
        if create_multi_hot:
            # Collect menu frequencies from valid menu names
            valid_menu_counts = self._collect_menus_by_category(
                category_column, "valid_menu_names"
            )

            # Select top menus for multi-hot encoding
            category_top_menus = self._filter_menus_by_frequency(
                valid_menu_counts, min_menu_freq, max_features_per_category
            )

            # Create multi-hot encoding features
            menu_feature_columns = self._create_multi_hot_menu_features(
                category_top_menus, category_column
            )

            self.engineered_feature_names.extend(menu_feature_columns)

    def calculate_bayesian_score(self: Self, k: int = 5, **kwargs) -> None:
        """
        Calculate Bayesian average score for each diner.

        Args:
            k: Minimum guaranteed review count (smoothing parameter).
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If 'combined_score' column is not found in review data.
        """
        if "combined_score" not in self.review.columns:
            raise ValueError(
                "'combined_score' column not found in review data. "
                "Please ensure reviews are properly processed."
            )

        # Calculate review statistics per diner
        grouped = (
            self.review.groupby("diner_idx")
            .agg(
                review_count=("reviewer_id", "count"),
                mean_combined_score=("combined_score", "mean"),
            )
            .reset_index()
        )

        # Calculate global mean
        mu = grouped["mean_combined_score"].mean()

        # Apply Bayesian Average formula
        grouped["bayesian_score"] = (
            (grouped["mean_combined_score"] * grouped["review_count"]) + (mu * k)
        ) / (grouped["review_count"] + k)

        # Merge with diner dataframe
        self.diner = pd.merge(
            left=self.diner,
            right=grouped[["diner_idx", "bayesian_score"]],
            on="diner_idx",
            how="left",
        )

        # Handle diners with no reviews
        self.diner["bayesian_score"] = self.diner["bayesian_score"].fillna(0)
        self.engineered_feature_names.append("bayesian_score")

    # Private helper methods
    def _clean_single_menu_name(self: Self, menu: str) -> Optional[str]:
        """
        Clean a single menu name by removing stopwords and formatting.

        Args:
            menu: Raw menu name string.

        Returns:
            Cleaned menu name or None if invalid.
        """
        stopwords = [
            "1인분",
            "1인세트",
            "세트",
            "추가",
            "가격표",
            "가격",
            "참고",
            "오시는 길",
            "내부에 추천메뉴",
        ]
        units = ["g", "kg", "개", "mm"]

        if pd.isna(menu) or not isinstance(menu, str):
            return None

        # Remove parentheses
        menu = re.sub(r"\([^)]*\)", "", menu)

        # Remove stopwords and units
        for word in stopwords + units:
            menu = menu.replace(word, "")

        # Keep only Korean characters and spaces
        menu = re.sub(r"[^\uAC00-\uD7A3\s]", "", menu)
        menu = re.sub(r"\s+", " ", menu).strip()

        # Filter out single characters
        if len(menu) <= 1:
            return None

        return menu

    def _clean_menu_list(self: Self, menu_list: Union[str, List[str]]) -> List[str]:
        """
        Clean a list of menu names.

        Args:
            menu_list: Raw menu list (string or list).

        Returns:
            List of cleaned menu names.
        """
        if isinstance(menu_list, str):
            try:
                menu_list = ast.literal_eval(menu_list)
            except (ValueError, SyntaxError):
                return []
        elif not isinstance(menu_list, list):
            return []

        cleaned_menus = []
        for menu in menu_list:
            if isinstance(menu, str):
                cleaned = self._clean_single_menu_name(menu)
                if cleaned:
                    cleaned_menus.append(cleaned)

        return cleaned_menus

    def _collect_menus_by_category(
        self: Self, category_column: str, menu_list_column: str = "cleaned_menu_list"
    ) -> Dict[str, Counter]:
        """
        Collect menu frequencies grouped by category.

        Args:
            category_column: Category column name.
            menu_list_column: Column containing menu lists.

        Returns:
            Dictionary mapping categories to menu frequency counters.
        """
        category_menu_counts = defaultdict(Counter)

        for _, row in self.diner.iterrows():
            category = row.get(category_column)
            menu_list = row.get(menu_list_column, [])

            if pd.notna(category) and isinstance(menu_list, list):
                for menu in menu_list:
                    if menu:
                        category_menu_counts[category][menu] += 1

        return dict(category_menu_counts)

    def _filter_menus_by_frequency(
        self: Self,
        category_menu_counts: Dict[str, Counter],
        min_frequency: int,
        max_count: Optional[int] = None,
        return_as_set: bool = False,
    ) -> Union[Dict[str, set], Dict[str, List[str]]]:
        """
        Filter menus by frequency and optionally limit count per category.

        Args:
            category_menu_counts: Menu frequency counters per category.
            min_frequency: Minimum frequency threshold.
            max_count: Maximum number of menus per category (None for unlimited).
            return_as_set: Whether to return sets instead of lists.

        Returns:
            Dictionary mapping categories to filtered menus (sets or lists).
        """
        result = {}

        for category, menu_counter in category_menu_counts.items():
            # Filter by minimum frequency
            frequent_menus = {
                menu: count
                for menu, count in menu_counter.items()
                if count >= min_frequency
            }

            if max_count is None:
                # Return all frequent menus
                filtered_menus = list(frequent_menus.keys())
            else:
                # Get top menus sorted by frequency
                top_menus = sorted(
                    frequent_menus.items(), key=lambda x: x[1], reverse=True
                )[:max_count]
                filtered_menus = [menu for menu, _ in top_menus]

            if return_as_set:
                result[category] = set(filtered_menus)
            else:
                result[category] = filtered_menus

        return result

    def _filter_menus_by_category_threshold(
        self: Self,
        row: pd.Series,
        category_column: str,
        category_valid_menus: Dict[str, set],
    ) -> List[str]:
        """
        Filter menus for a single row based on category-specific valid menu sets.

        Args:
            row: DataFrame row.
            category_column: Category column name.
            category_valid_menus: Dictionary of valid menus per category.

        Returns:
            List of filtered menu names.
        """
        category = row.get(category_column)
        menus = row.get("cleaned_menu_list", [])

        if pd.isna(category) or category not in category_valid_menus:
            return []

        valid_menus = category_valid_menus[category]
        return [menu for menu in menus if menu in valid_menus]

    def _create_multi_hot_menu_features(
        self: Self, category_top_menus: Dict[str, List[str]], category_column: str
    ) -> List[str]:
        """
        Create multi-hot encoding features for menu names.

        Args:
            category_top_menus: Top menus per category.
            category_column: Category column name.

        Returns:
            List of created feature column names.
        """
        menu_feature_columns = []

        for category, top_menus in category_top_menus.items():
            for menu in top_menus:
                # Create safe column name
                safe_menu_name = re.sub(r"[^\w]", "_", menu)
                col_name = f"menu_{category}_{safe_menu_name}"
                menu_feature_columns.append(col_name)

                # Initialize column with zeros
                self.diner[col_name] = 0

                # Set 1 for diners that have this menu in the same category
                mask = (self.diner[category_column] == category) & (
                    self.diner["valid_menu_names"].apply(
                        lambda x: menu in x if isinstance(x, list) else False
                    )
                )
                self.diner.loc[mask, col_name] = 1

        return menu_feature_columns

    @property
    def engineered_features(self: Self) -> pd.DataFrame:
        """
        Get engineered features without original features.

        Returns:
            DataFrame containing only engineered features with primary key.
        """
        return self.diner[self.engineered_feature_names]

    @property
    def engineered_meta_features(self: Self) -> pd.DataFrame:
        """
        Get engineered meta features without original features.

        Returns:
            DataFrame containing only engineered meta features with primary key.
        """
        return self.diner[self.engineered_meta_feature_names]
