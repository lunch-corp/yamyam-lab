import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# 프로젝트 루트 디렉토리를 Python path에 추가
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))


@st.cache_data(ttl=3600)
def load_data():
    """
    데이터를 로드하고 Streamlit 캐싱 기능을 사용하여 효율적으로 관리합니다.

    Returns:
        tuple: (review_df, diner_df, diner_category_df, reviewer_df) - 각 데이터프레임
    """
    # 데이터 경로 설정
    data_path = Path(__file__).parents[2] / "data"

    # 데이터 로딩 시작
    # start_time = time.time()

    with st.spinner("리뷰 데이터를 불러오는 중..."):
        review_df = pd.read_csv(data_path / "review.csv")

    with st.spinner("식당 데이터를 불러오는 중..."):
        diner_df = pd.read_csv(data_path / "diner.csv")

    with st.spinner("카테고리 데이터를 불러오는 중..."):
        diner_category_df = pd.read_csv(data_path / "diner_category_raw.csv")

    with st.spinner("리뷰어 데이터를 불러오는 중..."):
        reviewer_df = pd.read_csv(data_path / "reviewer.csv")

    diner_df = pd.merge(diner_df, diner_category_df, on="diner_idx", how="left")
    review_df = pd.merge(review_df, reviewer_df, on="reviewer_id", how="left")

    # end_time = time.time()
    # st.success(f"데이터 로딩 완료! (소요 시간: {end_time - start_time:.2f}초)")

    return review_df, diner_df


# @st.cache_data(ttl=3600)
# def get_review_data():
#     """리뷰 데이터만 로드합니다."""
#     data_path = Path(__file__).parents[2] / "data"
#     return pd.read_csv(data_path / "review.csv")


# @st.cache_data(ttl=3600)
# def get_diner_data():
#     """식당 데이터만 로드합니다."""
#     data_path = Path(__file__).parents[2] / "data"
#     return pd.read_csv(data_path / "diner.csv")


# @st.cache_data(ttl=3600)
# def get_category_data():
#     """카테고리 데이터만 로드합니다."""
#     data_path = Path(__file__).parents[2] / "data"
#     return pd.read_csv(data_path / "diner_category_raw.csv")


# @st.cache_data(ttl=3600)
# def get_reviewer_data():
#     """리뷰어 데이터만 로드합니다."""
#     data_path = Path(__file__).parents[2] / "data"
#     return pd.read_csv(data_path / "reviewer.csv")


def get_reviewer_info(
    review_df: pd.DataFrame, reviewer_id: int
) -> Tuple[pd.DataFrame, str]:
    """특정 리뷰어의 데이터와 이름을 반환합니다."""
    target_reviewer = review_df[review_df["reviewer_id"] == reviewer_id]
    if len(target_reviewer) == 0:
        return None, None
    reviewer_name = target_reviewer["reviewer_user_name"].iloc[0]
    target_reviewer["score_diff"] = (
        target_reviewer["reviewer_review_score"] - target_reviewer["reviewer_avg"]
    )
    return target_reviewer, reviewer_name


def calculate_menu_price_avg(price_series: pd.Series) -> float:
    """메뉴 가격의 평균을 계산합니다."""
    all_prices = []
    for prices in price_series.dropna():
        try:
            price_list = eval(prices) if isinstance(prices, str) else prices
            price_list = [
                float(p) for p in price_list if str(p).replace(".", "").isdigit()
            ]
            all_prices.extend(price_list)
        except:
            continue
    return np.mean(all_prices) if all_prices else 0


def analyze_menu_frequency(menu_series: pd.Series) -> pd.Series:
    """메뉴 빈도를 분석합니다."""
    all_menus = []
    for menus in menu_series.dropna():
        try:
            menu_list = eval(menus) if isinstance(menus, str) else menus
            all_menus.extend(menu_list)
        except:
            continue
    return pd.Series(all_menus).value_counts()


def get_category_stats(
    merged_df: pd.DataFrame, category_col: str, top_n: int = 10
) -> pd.DataFrame:
    """카테고리별 통계를 계산합니다."""
    # 전체 카운트
    total_counts = merged_df[category_col].value_counts()

    # 만족하는 케이스 카운트
    satisfied = merged_df[merged_df["score_diff"] > 0]
    satisfied_counts = satisfied[category_col].value_counts()

    # 데이터프레임 생성
    stats_df = pd.DataFrame(
        {
            "총 방문": total_counts,
            "만족": satisfied_counts.reindex(total_counts.index).fillna(0),
        }
    )

    # 만족도 비율 계산
    stats_df["만족도(%)"] = (stats_df["만족"] / stats_df["총 방문"] * 100).round(1)

    return stats_df.head(top_n)


def get_average_scores(merged_df: pd.DataFrame) -> Dict[str, float]:
    """평균 점수들을 계산합니다."""
    return {
        "Diner Avg": round(merged_df["diner_review_avg"].mean(), 2),
        "Bayesian Score": round(merged_df["bayesian_score"].mean(), 2),
        "Reviewer Avg": round(merged_df["reviewer_avg"].mean(), 1),
    }
