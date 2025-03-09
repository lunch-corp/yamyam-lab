import ast
from typing import List, Self, Dict, Any

import numpy as np
import pandas as pd

from store.base import BaseFeatureStore


class DinerFeatureStore(BaseFeatureStore):
    def __init__(
        self: Self, review: pd.DataFrame, diner: pd.DataFrame, feature_param_pair: Dict[str, Dict[str, Any]]
    ):
        """
        Feature engineering on diner data.
        This class gets `features` indicating which features to make.
        Unimplemented feature name will raise error with `self.feature_methods`.

        Args:
            review (pd.DataFrame): Review data.
            diner (pd.DataFrame): Diner data.
            feature_param_pair (Dict[str, Dict[str, Any]]): Key is name of engineered feature and
                values are its corresponding parameters.
        """
        super().__init__(review, diner, features)

        self.feature_methods = {
            "all_review_cnt": self.calculate_all_review_cnt,
            "diner_review_tags": self.calculate_diner_score,
            "diner_menu_price": self.calculate_diner_price,
            "diner_mean_review_score": self.calculate_diner_mean_review_score,
            "one_hot_encoding_categorical_features": self.one_hot_encoding_categorical_features,
        }
        for feat, arg in feature_param_pair.items():
            if feat not in self.feature_methods.keys():
                raise ValueError(f"{feat} not matched with implemented method")
        self.feature_param_pair = feature_param_pair

        self.engineered_feature_names = []

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
        self.diner["all_review_cnt"] = self.diner["diner_idx"].map(diner_idx2review_cnt)
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
        # mean review score from review data
        diner_id2score = (self.review.groupby("diner_idx")["reviewer_review_score"]
                          .mean()
                          .to_dict())
        # diners that do not have any reviews
        diner_id_not_exists = set(self.diner["diner_idx"].unique()) - set(self.review["diner_idx"].unique())
        for diner_id in diner_id_not_exists:
            # processing null values as zero could trigger bias
            # because it treats diners to have lowest review scores
            diner_id2score[diner_id] = 0
        self.diner["mean_review_score"] = self.diner["diner_idx"].map(diner_id2score)

        self.engineered_feature_names.append("mean_review_score")

    def one_hot_encoding_categorical_features(self: Self, categorical_feature_name: List[str], **kwargs) -> None:
        for feature_name in categorical_feature_name:
            if feature_name not in self.diner.columns:
                raise ValueError(f"{feature_name} not in diner data")
            one_hot_encoding_feat = pd.get_dummies(self.diner[feature_name], prefix=feature_name).astype(int)
            self.diner = pd.concat([self.diner, one_hot_encoding_feat], axis=1)

            self.engineered_feature_names.extend(
                list(one_hot_encoding_feat.columns)
            )

    # NaN 또는 빈 리스트를 처리할 수 있도록 정의
    def _extract_statistics(self: Self, prices: str) -> pd.Series:
        if not prices or pd.isna(prices):  # 빈 리스트라면 NaN 반환
            return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])

        # 문자열을 리스트로 변환, 이 부분은 데이터 검증 과정에서 처리할 필요가 있어보입니다.
        # 추후에 데이터 검증 코드 완성되면 이 부분은 수정이 필요할 것 같습니다.
        prices = ast.literal_eval(prices)
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
            if pd.isna(review):  # 결측치 예외 처리
                continue

            try:
                parsed_review = ast.literal_eval(review)  # 안전한 문자열 평가
                for cat, score in parsed_review:
                    if cat in category_map:
                        scores[i, category_map[cat]] = score

            except (SyntaxError, ValueError, TypeError):
                continue  # 파싱 에러 방지

        return scores

    def _get_engineered_features(self) -> pd.DataFrame:
        return self.diner[self.engineered_feature_names]


class UserFeatureStore:
    def __init__(
        self: Self, review: pd.DataFrame, diner: pd.DataFrame, feature_param_pair: Dict[str, Dict[str, Any]]
    ):
        """
        Feature engineering on diner data.
        This class gets `features` indicating which features to make.
        Unimplemented feature name will raise error with `self.feature_methods`.

        Args:
            review (pd.DataFrame): Review data.
            diner (pd.DataFrame): Diner data.
            feature_param_pair (Dict[str, Dict[str, Any]]): Key is name of engineered feature and
                values are its corresponding parameters.
        """
        self.review = pd.merge(
            left=review,
            right=diner[["diner_idx", "diner_category_large", "diner_category_middle"]],
            how="left",
            on="diner_idx",
        )
        self.diner = diner
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
            {
                "reviewer_id": sorted(review["reviewer_id"].unique())
            }
        )

    def make_features(self: Self) -> None:
        """
        Feature engineer using `self.features`.
        """
        for feat, params in self.feature_param_pair.items():
            self.feature_methods[feat](**params)

    def calculate_categorical_feature_count(self: Self, categorical_feature_names: List[str], **kwargs):
        for feature in categorical_feature_names:
            if feature not in self.review.columns:
                raise ValueError(f"{feature} not in review data columns")
            category_feat = (self.review.groupby(["reviewer_id", feature])
                             .size()
                             .unstack(fill_value=0)
                             .reset_index())
            self.user = pd.merge(
                left=self.user,
                right=category_feat,
                how="inner",
                on="reviewer_id",
            )

    def calculate_user_mean_review_score(self: Self, **kwargs):
        score_feat = (self.review.groupby("reviewer_id")["reviewer_review_score"]
                      .mean()
                      .reset_index())
        self.user = pd.merge(
            left=self.user,
            right=score_feat,
            how="inner",
            on="reviewer_id",
        )

    def _get_user_feature(self: Self) -> pd.DataFrame:
        return self.user
