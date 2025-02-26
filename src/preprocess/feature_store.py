from typing import List

import numpy as np
import pandas as pd


# NaN 또는 빈 리스트를 처리할 수 있도록 정의
def extract_statistics(prices: list[int, float]) -> pd.Series:
    if not prices or pd.isna(prices):  # 빈 리스트라면 NaN 반환
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
    # when prices do not include pure float, such as `변동가격`,
    # float(price) raises error
    # todo: preprocess null value
    try:
        prices = [float(price) for price in prices]
    except:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
    return pd.Series(
        [min(prices), max(prices), np.mean(prices), np.median(prices), len(prices)]
    )


# numpy 기반으로 점수 추출 최적화
def extract_scores_array(reviews: str, categories: list[tuple[str, str]]) -> np.ndarray:
    # 리뷰 데이터를 파싱하여 배열로 변환
    parsed = [[] if pd.isna(review) else eval(review) for review in reviews]
    # 카테고리별 점수 초기화 (rows x categories)
    scores = np.zeros((len(reviews), len(categories)), dtype=int)

    # 각 리뷰에서 카테고리 점수 추출
    category_map = {cat: idx for idx, (cat, _) in enumerate(categories)}
    for row_idx, review in enumerate(parsed):
        for cat, score in review:
            if cat in category_map:  # 해당 카테고리가 정의된 경우
                scores[row_idx, category_map[cat]] = score

    return scores


class DinerFeatureStore:
    def __init__(
        self,
        review: pd.DataFrame,
        diner: pd.DataFrame,
        features: List[str],
    ):
        """
        Feature engineering on diner data.
        This class gets `features` indicating which features to make.
        Unimplemented feature name will raise error with `self.feature_methods`.

        Args:
            review (pd.DataFrame): Review data.
            diner (pd.DataFrame): Diner data.
            features (List[str]): List of features to make.
        """
        self.review = review
        self.diner = diner
        self.feature_methods = {"all_review_cnt": self.calculate_all_review_cnt}
        for feature in features:
            if feature not in self.feature_methods.keys():
                raise ValueError(f"{feature} not matched with implemented method")
        self.features = features

    def make_features(self) -> None:
        """
        Feature engineer using `self.features`.
        """
        for feature in self.features:
            featuren_eng_func = self.feature_methods[feature]
            featuren_eng_func()

    def calculate_all_review_cnt(self) -> None:
        """
        Calculate number of review counts for each diner.
        """
        diner_idx2review_cnt = self.review["diner_idx"].value_counts().to_dict()
        self.diner["all_review_cnt"] = self.diner["diner_idx"].map(diner_idx2review_cnt)
