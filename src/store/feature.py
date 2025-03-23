from typing import Any, Dict, List, Self

import numpy as np
import pandas as pd

from store.base import BaseFeatureStore
from tools.h3 import get_h3_index, get_hexagon_neighbors


class DinerFeatureStore(BaseFeatureStore):
    def __init__(
        self: Self,
        review: pd.DataFrame,
        diner: pd.DataFrame,
        feature_param_pair: Dict[str, Dict[str, Any]],
    ):
        """
        Feature engineering on diner data.
        This class gets `feature_param_pair` indicating which features to make with corresponding parameters.
        Unimplemented feature name will raise error with `self.feature_methods`.

        Args:
            review (pd.DataFrame): Review data.
            diner (pd.DataFrame): Diner data.
            feature_param_pair (Dict[str, Dict[str, Any]]): Key is name of engineered feature and
                values are its corresponding parameters.
        """
        super().__init__(
            review=review,
            diner=diner,
            feature_param_pair=feature_param_pair,
        )

        self.feature_methods = {
            "all_review_cnt": self.calculate_all_review_cnt,
            "diner_review_tags": self.calculate_diner_score,
            "diner_menu_price": self.calculate_diner_price,
            "diner_mean_review_score": self.calculate_diner_mean_review_score,
            "one_hot_encoding_categorical_features": self.one_hot_encoding_categorical_features,
            "diner_category_meta_combined_with_h3": self.make_diner_category_meta_combined_with_h3,
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
            self.diner["diner_review_tags"], tag_categories
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
        Calculates mean review score from review data.
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
        self: Self, reviews: str, categories: list[tuple[str, str]]
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
                for cat, score in review:
                    if cat in category_map:
                        scores[i, category_map[cat]] = score

            except (SyntaxError, ValueError, TypeError):
                continue  # 파싱 에러 방지

        return scores

    @property
    def engineered_features(self) -> pd.DataFrame:
        """
        Get engineered features only without original features with primary key.

        Returns (pd.DataFrame):
            Engineered features dataframe.
        """
        return self.diner[self.engineered_feature_names]

    @property
    def engineered_meta_features(self) -> pd.DataFrame:
        """
        Get engineered `meta` features only without original features with primary key.

        Returns (pd.DataFrame):
            Engineered features dataframe.
        """
        return self.diner[self.engineered_meta_feature_names]


class UserFeatureStore(BaseFeatureStore):
    def __init__(
        self: Self,
        review: pd.DataFrame,
        diner: pd.DataFrame,
        feature_param_pair: Dict[str, Dict[str, Any]],
    ):
        """
        Feature engineering on user data.
        This class gets `feature_param_pair` indicating which features to make with corresponding parameters.
        Unimplemented feature name will raise error with `self.feature_methods`.

        Args:
            review (pd.DataFrame): Review data.
            diner (pd.DataFrame): Diner data.
            feature_param_pair (Dict[str, Dict[str, Any]]): Key is name of engineered feature and
                values are its corresponding parameters.
        """
        super().__init__(
            review=review,
            diner=diner,
            feature_param_pair=feature_param_pair,
        )

        self.review = pd.merge(
            left=review,
            right=diner[["diner_idx", "diner_category_large", "diner_category_middle"]],
            how="left",
            on="diner_idx",
        )
        self.feature_methods = {
            "categorical_feature_count": self.calculate_categorical_feature_count,
            "user_mean_review_score": self.calculate_user_mean_review_score,
        }
        for feat, arg in feature_param_pair.items():
            if feat not in self.feature_methods.keys():
                raise ValueError(f"{feat} not matched with implemented method")
        self.feature_param_pair = feature_param_pair
        # initial user feature dataframe to merge with other engineered features
        self.user = pd.DataFrame(
            {"reviewer_id": sorted(review["reviewer_id"].unique())}
        )

    def make_features(self: Self) -> None:
        """
        Feature engineer using `self.features`.
        """
        for feat, params in self.feature_param_pair.items():
            self.feature_methods[feat](**params)

    def calculate_categorical_feature_count(
        self: Self, categorical_feature_names: List[str], **kwargs
    ) -> None:
        """
        Count number of appearance of each category in a categorical feature.

        Args:
            categorical_feature_names (List[str]): List of categorical features.
            **kwargs: Additional keyword arguments.
        """
        for feature in categorical_feature_names:
            if feature not in self.review.columns:
                raise ValueError(f"{feature} not in review data columns")
            category_feat = (
                self.review.groupby(["reviewer_id", feature])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )
            self.user = pd.merge(
                left=self.user,
                right=category_feat,
                how="inner",
                on="reviewer_id",
            )

    def calculate_user_mean_review_score(self: Self, **kwargs) -> None:
        """
        Calculates mean review score of each user.

        Args:
            **kwargs: Additional keyword arguments.
        """
        score_feat = (
            self.review.groupby("reviewer_id")["reviewer_review_score"]
            .mean()
            .reset_index()
        )
        self.user = pd.merge(
            left=self.user,
            right=score_feat,
            how="inner",
            on="reviewer_id",
        )

    @property
    def engineered_features(self: Self) -> pd.DataFrame:
        """
        Get engineered features only without original features with primary key.

        Returns (pd.DataFrame):
            Engineered features dataframe.
        """
        return self.user
