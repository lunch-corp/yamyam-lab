import datetime as dt
from typing import Any, Dict, List, Self

import pandas as pd

from .base import BaseFeatureStore


class UserFeatureStore(BaseFeatureStore):
    def __init__(
        self: Self,
        review: pd.DataFrame,
        diner: pd.DataFrame,
        all_user_ids: List[int],
        feature_param_pair: Dict[str, Dict[str, Any]],
    ):
        """
        Feature engineering on user data.
        This class gets `feature_param_pair` indicating which features to make with corresponding parameters.
        Unimplemented feature name will raise error with `self.feature_methods`.

        Args:
            review (pd.DataFrame): Review data which will be train dataset.
            diner (pd.DataFrame): Diner data.
            all_user_ids (List[int]): User ids from all of review data, which is train, val and test dataset.
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
            "user_activity_patterns": self.calculate_activity_patterns,
            # "scaled_scores": self.calculate_scaled_scores,
        }
        for feat, arg in feature_param_pair.items():
            if feat not in self.feature_methods.keys():
                raise ValueError(f"{feat} not matched with implemented method")
        self.feature_param_pair = feature_param_pair

        # initial user feature dataframe to merge with other engineered features
        # Note: reviewer_id is initialized from `all_user_ids`, not from train review data.
        # self.user will be continuously left merged with features calculated from train review data.
        # This means that, for cold start users in val and test data, feature value will be nan.
        # If we decide to recommend items to cold start users using most popular items,
        # we do not have to fill na values for cold start users.
        # Filling with zeros will be used as temporary na filling strategy, so please be aware of it.
        self.user = pd.DataFrame({"reviewer_id": sorted(all_user_ids)})

        self.feature_dict = {
            "한식": "korean",
            "중식": "chinese",
            "일식": "japanese",
            "양식": "western",
            "간식": "snack",
            "아시아음식": "asian",
            "패스트푸드": "fastfood",
            "디저트": "dessert",
            "카페": "cafe",
        }

    def make_features(self: Self) -> None:
        """
        Feature engineer using `self.features`.
        """
        for feat, params in self.feature_param_pair.items():
            self.feature_methods[feat](**params)

        # temporary na fill logic, which fills na value with 0.
        # If we do not use most popular items recommendation to cold start users,
        # this line should be updated with customized na filling logic.
        self.user.fillna(0)

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

            category_feat.columns = [
                self.feature_dict.get(col, 0) if col in self.feature_dict else col
                for col in category_feat.columns
            ]

            self.user = pd.merge(
                left=self.user,
                right=category_feat,
                how="left",
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
            how="left",
            on="reviewer_id",
        )

    def calculate_activity_patterns(self: Self, **kwargs) -> None:
        """
        Calculate features related to user's activity patterns.
        """
        # 리뷰 작성 날짜 정보 추출
        review_date = pd.to_datetime(self.review["reviewer_review_date"])

        # 요일별 패턴 (0: 월요일, 6: 일요일)
        day_patterns = pd.get_dummies(review_date.dt.dayofweek).astype(int)
        day_patterns.columns = [f"day_{col}" for col in day_patterns.columns]
        day_patterns["reviewer_id"] = self.review["reviewer_id"]

        # 요일별 합계 계산
        daily_counts = day_patterns.groupby("reviewer_id").sum().reset_index()

        # 최근성 feature
        recency = (
            self.review.groupby("reviewer_id")["reviewer_review_date"]
            .max()
            .reset_index()
        )
        recency["days_since_last_review"] = (
            pd.Timestamp.now() - pd.to_datetime(recency["reviewer_review_date"])
        ).dt.days

        # 요일 패턴과 최근성 정보를 사용자 데이터에 병합
        self.user = pd.merge(
            left=self.user,
            right=daily_counts,
            how="left",
            on="reviewer_id",
        )

        self.user = pd.merge(
            left=self.user,
            right=recency[["reviewer_id", "days_since_last_review"]],
            how="left",
            on="reviewer_id",
        )

    def calculate_scaled_scores(
        self: Self,
        badge_range: tuple = (0.1, 1.0),
        date_range: tuple = (0.1, 1.0),
        score_range: tuple = (-0.5, 1.0),
        decay_period: int = 1095,
        badge_weight: float = 0.3,
        date_weight: float = 0.2,
        score_weight: float = 0.5,
        **kwargs,
    ) -> None:
        """
        Calculate badge_scaled, date_scaled, score_scaled, and combined_score for each user.

        Args:
            badge_range (tuple): The range to scale badge_level to (min, max).
            date_range (tuple): The range to scale date_weight to (min, max).
            score_range (tuple): The range to scale score_diff to (min, max).
            decay_period (int): Date weight decay period (default: 1095 days).
            badge_weight (float): Weight for badge_level (default: 0.3).
            date_weight (float): Weight for date_weight (default: 0.2).
            score_weight (float): Weight for score_diff (default: 0.5).
            **kwargs: Additional keyword arguments.

        Returns:
            None: This function does not return a value. Instead, it calculates and stores the scaled scores for each user.
        """

        # date_weight와 score_diff가 없는 경우 계산
        if (
            "date_weight" not in self.review.columns
            or "score_diff" not in self.review.columns
        ):
            self._calculate_weighted_score(
                decay_period=decay_period,
            )

        # 각 열의 최소, 최대값 계산
        badge_min, badge_max = (
            self.review["badge_level"].min(),
            self.review["badge_level"].max(),
        )
        date_weight_min, date_weight_max = (
            self.review["date_weight"].min(),
            self.review["date_weight"].max(),
        )
        score_diff_min, score_diff_max = (
            self.review["score_diff"].min(),
            self.review["score_diff"].max(),
        )

        # 스케일링된 값 계산
        self.review["badge_scaled"] = self.review["badge_level"].apply(
            lambda x: min_max_scaling(
                x, badge_min, badge_max, badge_range[0], badge_range[1]
            )
        )

        self.review["date_scaled"] = self.review["date_weight"].apply(
            lambda x: min_max_scaling(
                x, date_weight_min, date_weight_max, date_range[0], date_range[1]
            )
        )

        self.review["score_scaled"] = self.review["score_diff"].apply(
            lambda x: min_max_scaling(
                x, score_diff_min, score_diff_max, score_range[0], score_range[1]
            )
        )

        # combined_score 계산 (badge_scaled 포함)
        self.review["combined_score"] = (
            (badge_weight * self.review["badge_scaled"])
            + (date_weight * self.review["date_scaled"])
            + (score_weight * self.review["score_scaled"])
        )

        # 사용자별로 평균 계산하여 사용자 특성으로 추가
        scaled_features = (
            self.review.groupby("reviewer_id")
            .agg(
                {
                    "date_scaled": "mean",
                    "score_scaled": "mean",
                    "combined_score": "mean",
                }
            )
            .reset_index()
        )

        self.user = pd.merge(
            left=self.user,
            right=scaled_features,
            how="left",
            on="reviewer_id",
        )

    def _calculate_weighted_score(
        self: Self,
        decay_period: int = 1095,
    ) -> None:
        """
        Calculate date_weight and score_diff for reviews if they don't exist.

        Args:
            decay_period (int): Date weight decay period in days
        """

        # 필수 컬럼 확인
        required_columns = [
            "badge_level",
            "reviewer_review_score",
            "reviewer_avg",
            "reviewer_review_date",
        ]
        for col in required_columns:
            if col not in self.review.columns:
                raise ValueError(
                    f"{col} not in review data columns. Required for weighted score calculation."
                )

        # 현재 날짜 설정
        current_date = dt.datetime.now()

        def _partly_calculate_weighted_score(row, today=current_date):
            """
            Calculate weighted_score to date-weight review scores

            Returns:
                Tuple of (date_weight, score_diff)
            """
            # 날짜 차이 계산
            days_diff = (today - row["reviewer_review_date"]).days
            date_weight = max(1 - max(0, days_diff - 90) / decay_period, 0.01)

            # 사용자 평균과의 차이 계산
            simple_score = row["reviewer_review_score"] - row["reviewer_avg"]

            return (date_weight, simple_score)

        # 리뷰 데이터에 가중치가 적용된 점수 계산
        self.review[["date_weight", "score_diff"]] = self.review.apply(
            _partly_calculate_weighted_score, axis=1, result_type="expand"
        )

    @property
    def engineered_features(self: Self) -> pd.DataFrame:
        """
        Get engineered features only without original features with primary key.

        Returns (pd.DataFrame):
            Engineered features dataframe.
        """
        return self.user


def min_max_scaling(
    value: float, min_val: float, max_val: float, range_min: float, range_max: float
) -> float:
    """
    Scale a value to a specified range using min-max scaling.

    Args:
        value (float): The value to be scaled.
        min_val (float): The minimum value in the original data.
        max_val (float): The maximum value in the original data.
        range_min (float): The minimum value of the target range.
        range_max (float): The maximum value of the target range.

    Returns:
        float: The scaled value within the specified range.
    """
    if max_val - min_val == 0:  # Zero division 방지
        return range_min
    return ((value - min_val) / (max_val - min_val)) * (
        range_max - range_min
    ) + range_min
