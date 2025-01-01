import numpy as np
import pandas as pd



# NaN 또는 빈 리스트를 처리할 수 있도록 정의
def extract_statistics(prices: list[int, float]) -> pd.Series:
    if not prices:  # 빈 리스트라면 NaN 반환
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
    return pd.Series(
        [min(prices), max(prices), np.mean(prices), np.median(prices), len(prices)]
    )


# numpy 기반으로 점수 추출 최적화
def extract_scores_array(reviews: str, categories: list[tuple[str, str]]) -> np.ndarray:
    # 리뷰 데이터를 파싱하여 배열로 변환
    parsed = [eval(review) for review in reviews]
    # 카테고리별 점수 초기화 (rows x categories)
    scores = np.zeros((len(reviews), len(categories)), dtype=int)

    # 각 리뷰에서 카테고리 점수 추출
    category_map = {cat: idx for idx, (cat, _) in enumerate(categories)}
    for row_idx, review in enumerate(parsed):
        for cat, score in review:
            if cat in category_map:  # 해당 카테고리가 정의된 경우
                scores[row_idx, category_map[cat]] = score

    return scores
