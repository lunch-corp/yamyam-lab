import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# 프로젝트 루트 디렉토리를 Python path에 추가
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from src.tools.google_drive import ensure_data_files


@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """리뷰, 식당, 리뷰어 데이터를 로드하고 캐시합니다."""

    # # 필요한 파일 다운로드 확인
    data_paths = ensure_data_files()

    # 식당 데이터 로드 및 형변환
    diner = pd.read_csv(data_paths["diner"])

    # 숫자형 컬럼
    numeric_cols_diner = [
        "diner_idx",
        "diner_review_cnt",
        "diner_blog_review_cnt",
        "diner_review_avg",
        "diner_lat",
        "diner_lon",
        "real_good_review_cnt",
        "real_bad_review_cnt",
        "all_review_cnt",
        "real_good_review_percent",
        "real_bad_review_percent",
        "bayesian_score",
        "rank",
    ]
    diner[numeric_cols_diner] = diner[numeric_cols_diner].apply(pd.to_numeric, errors="coerce")

    # 문자열 컬럼
    string_cols_diner = [
        "diner_name",
        "diner_category_large",
        "diner_category_middle",
        "diner_category_small",
        "diner_category_detail",
        "diner_address",
        "diner_phone",
        "diner_url",
        "diner_open_time",
        "diner_address_constituency",
    ]
    diner[string_cols_diner] = diner[string_cols_diner].astype(str)

    # 불리언 컬럼
    diner["is_small_category_missing"] = diner["is_small_category_missing"].astype(bool)

    # 리스트형 컬럼들
    list_cols_diner = [
        "diner_tag",
        "diner_menu",
        "diner_menu_name",
        "diner_menu_price",
        "diner_review_tags",
    ]
    for col in list_cols_diner:
        diner[col] = diner[col].apply(lambda x: eval(x) if pd.notna(x) and x != "nan" else [])

    # 리뷰어 데이터 로드 및 형변환
    reviewer = pd.read_csv(data_paths["reviewer"])

    # 숫자형 컬럼
    numeric_cols_reviewer = ["reviewer_id", "reviewer_avg", "badge_level"]
    reviewer[numeric_cols_reviewer] = reviewer[numeric_cols_reviewer].apply(
        pd.to_numeric, errors="coerce"
    )

    # 문자열 컬럼
    string_cols_reviewer = ["badge_grade", "reviewer_user_name"]
    reviewer[string_cols_reviewer] = reviewer[string_cols_reviewer].astype(str)

    # 리뷰 데이터 로드 및 형변환
    review_columns = [
        "diner_idx",
        "review_id",
        "reviewer_id",
        "reviewer_review_date",
        "reviewer_review_score",
    ]

    review = pd.read_csv(data_paths["review"], usecols=review_columns)

    # 날짜형 변환
    review["reviewer_review_date"] = pd.to_datetime(review["reviewer_review_date"])

    # 숫자형 변환
    numeric_cols_review = ["diner_idx", "review_id", "reviewer_id"]
    review[numeric_cols_review] = review[numeric_cols_review].apply(pd.to_numeric, errors="coerce")

    # 데이터 정제
    # NA 값 처리
    diner = diner.fillna(
        {
            "diner_open_time": "",
            "diner_phone": "",
            "diner_url": "",
            "diner_address_constituency": "",
        }
    )

    reviewer = reviewer.fillna({"badge_grade": "", "reviewer_user_name": ""})

    review = review.fillna({"reviewer_review": ""})

    # 리뷰 데이터에 리뷰어 정보 병합
    review = pd.merge(review, reviewer, on="reviewer_id", how="left")
    review["score_diff"] = review["reviewer_review_score"] - review["reviewer_avg"]
    
    category_kakao = pd.read_csv(data_paths['category'])
    
    category_frequency = pd.read_csv('/Users/seongrok.kim/Github/yamyam-lab/data/category_frequency.csv')
    return review, diner, category_kakao, category_frequency


def get_reviewer_info(review_df: pd.DataFrame, reviewer_id: int) -> Tuple[pd.DataFrame, str]:
    """특정 리뷰어의 데이터와 이름을 반환합니다."""
    target_reviewer = review_df[review_df["reviewer_id"] == reviewer_id]
    if len(target_reviewer) == 0:
        return None, None
    reviewer_name = target_reviewer["reviewer_user_name"].iloc[0]
    return target_reviewer, reviewer_name


def calculate_menu_price_avg(price_series: pd.Series) -> float:
    """메뉴 가격의 평균을 계산합니다."""
    all_prices = []
    for prices in price_series.dropna():
        try:
            price_list = eval(prices) if isinstance(prices, str) else prices
            price_list = [float(p) for p in price_list if str(p).replace(".", "").isdigit()]
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


def get_category_stats(merged_df: pd.DataFrame, category_col: str, top_n: int = 10) -> pd.DataFrame:
    """카테고리별 통계를 계산합니다."""
    # 전체 카운트
    total_counts = merged_df[category_col].value_counts()

    # 만족하는 케이스 카운트
    satisfied = merged_df[merged_df["score_diff"] > 0]
    satisfied_counts = satisfied[category_col].value_counts()

    # 데이터프레임 생성
    stats_df = pd.DataFrame(
        {"총 방문": total_counts, "만족": satisfied_counts.reindex(total_counts.index).fillna(0)}
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
