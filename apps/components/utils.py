import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

# 프로젝트 루트 디렉토리를 Python path에 추가
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))


# 개별 데이터 로더 함수들
@st.cache_data(ttl=3600)
def load_review_data(columns=None):
    """리뷰 데이터와 리뷰어 데이터를 로드합니다.

    Args:
        columns: 로드할 컬럼 리스트 (None이면 모든 컬럼)
    """
    data_path = Path(__file__).parents[2] / "data"

    with st.spinner("리뷰 데이터를 불러오는 중..."):
        review_df = pd.read_csv(data_path / "review.csv", usecols=columns)

    with st.spinner("리뷰어 데이터를 불러오는 중..."):
        reviewer_df = pd.read_csv(data_path / "reviewer.csv")

    # 리뷰와 리뷰어 데이터 merge
    review_df = pd.merge(review_df, reviewer_df, on="reviewer_id", how="left")

    return review_df


@st.cache_data(ttl=3600)
def load_diner_data(columns=None):
    """식당 데이터와 카테고리 데이터를 로드합니다.

    Args:
        columns: 로드할 컬럼 리스트 (None이면 모든 컬럼)
    """
    data_path = Path(__file__).parents[2] / "data"

    with st.spinner("식당 데이터를 불러오는 중..."):
        diner_df = pd.read_csv(data_path / "diner.csv", usecols=columns)

    with st.spinner("카테고리 데이터를 불러오는 중..."):
        diner_category_df = pd.read_csv(data_path / "diner_category.csv")

    # 식당과 카테고리 데이터 merge
    diner_df = pd.merge(diner_df, diner_category_df, on="diner_idx", how="left")

    return diner_df


@st.cache_data(ttl=3600)
def load_keyword_data():
    """키워드 데이터를 로드합니다."""
    data_path = Path(__file__).parents[2] / "data"

    with st.spinner("리뷰 키워드 데이터를 불러오는 중..."):
        review_keyword = pd.read_csv(data_path / "review_keyword.csv")
        review_keyword["parsed_keywords"] = review_keyword["parsed_keywords"].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

    return review_keyword


# 빠른 데이터 로더 (기본 컬럼만)
@st.cache_data(ttl=3600)
def load_basic_review_data():
    """기본 리뷰 데이터만 빠르게 로드합니다."""
    basic_columns = [
        "review_id",
        "diner_idx",
        "reviewer_id",
        "reviewer_review_score",
        "reviewer_review_date",
    ]
    return load_review_data(columns=basic_columns)


@st.cache_data(ttl=3600)
def load_basic_diner_data():
    """기본 식당 데이터만 빠르게 로드합니다."""
    basic_columns = [
        "diner_idx",
        "diner_name",
        "diner_review_avg",
        "diner_review_cnt",
        "bayesian_score",
    ]
    return load_diner_data(columns=basic_columns)


# 선택적 merge 함수들
@st.cache_data(ttl=1800)  # 더 짧은 TTL 사용
def merge_review_diner(review_df, diner_df):
    """리뷰 데이터와 식당 데이터를 merge합니다."""
    return pd.merge(review_df, diner_df, on="diner_idx", how="left")


@st.cache_data(ttl=1800)
def merge_review_keywords(review_df, keyword_df):
    """리뷰 데이터와 키워드 데이터를 merge합니다."""
    return pd.merge(review_df, keyword_df, on="review_id", how="left")


@st.cache_data(ttl=3600)
def load_data():
    """
    호환성을 위한 기존 함수 - 모든 데이터를 로드합니다.
    새로운 코드에서는 개별 로더 함수들을 사용하는 것을 권장합니다.

    Returns:
        tuple: (review_df, diner_df) - 각 데이터프레임
    """
    # 개별 데이터 로드
    review_df = load_review_data()
    diner_df = load_diner_data()
    keyword_df = load_keyword_data()

    # 키워드 데이터 merge
    review_df = merge_review_keywords(review_df, keyword_df)

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


def analyze_keywords(review_keyword_series, sentiment_threshold=0.4):
    """리뷰 키워드를 분석합니다."""
    keywords = []
    categories = []
    sentiments = []

    for keywords_str in review_keyword_series:
        if pd.isna(keywords_str):
            continue

        try:
            keywords.append(keywords_str["term"])
            categories.append(keywords_str["category"])
            sentiments.append(float(keywords_str["sentiment"]))
        except:
            continue

    keyword_df = pd.DataFrame(
        {
            "keyword": keywords,
            "category": categories,
            "sentiment": sentiments,
        }
    )

    # 긍정/부정 키워드 분류
    positive_keywords = keyword_df[keyword_df["sentiment"] >= sentiment_threshold]
    negative_keywords = keyword_df[keyword_df["sentiment"] < sentiment_threshold]

    return keyword_df, positive_keywords, negative_keywords


def parse_keywords_safely(keywords_df):
    """키워드 DataFrame을 안전하게 파싱합니다."""
    if keywords_df.empty:
        return []

    try:
        # DataFrame을 dictionary 리스트로 변환
        keywords_list = keywords_df.apply(
            lambda x: {
                "term": x["term"],
                "category": x["category"],
                "sentiment": float(x["sentiment"]),
            }
        ).tolist()
        return keywords_list
    except Exception as e:
        print(f"Error parsing keywords: {e}")
        return []


def calculate_sentiment_score(keywords_df):
    """키워드 DataFrame에서 평균 감성 점수를 계산합니다."""
    if keywords_df.empty:
        return None
    try:
        return pd.to_numeric(keywords_df["sentiment"]).mean()
    except (KeyError, ValueError, TypeError):
        return None


def get_word_cloud_data(df, diner_name):
    """키워드 DataFrame에서 워드클라우드 데이터를 준비합니다."""
    if len(df) > 0:
        # 카테고리별 긍정 키워드 분석
        pos_by_category = (
            df.groupby(["category", "keyword"]).size().reset_index(name="count")
        )
        pos_by_category = pos_by_category.sort_values(
            ["category", "count"], ascending=[True, False]
        )

        st.dataframe(pos_by_category)

        word_freq = dict(zip(pos_by_category["keyword"], pos_by_category["count"]))

        # 워드클라우드 생성
        if word_freq:
            st.write("### 키워드 워드클라우드")
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                font_path="/System/Library/Fonts/AppleSDGothicNeo.ttc",  # 한글 폰트 설정
            ).generate_from_frequencies(word_freq)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            # 키워드 카테고리 분포
            fig_category = px.pie(
                pos_by_category.groupby("category")["count"].sum().reset_index(),
                values="count",
                names="category",
                title=f"{diner_name}의 키워드 카테고리 분포",
            )
            st.plotly_chart(fig_category)
