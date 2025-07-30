import pandas as pd
import plotly.express as px
import streamlit as st

from apps.components.utils import load_diner_data, load_review_data


def create_category_treemap(diner_df):
    """
    카테고리 구조를 트리맵으로 시각화
    """
    # 각 카테고리 레벨별 카운트
    df_grouped = (
        diner_df.groupby(
            [
                "diner_category_large",
                "diner_category_middle",
                "diner_category_small",
            ]
        )
        .size()
        .reset_index(name="count")
    )

    # 누락된 값 처리
    df_grouped = df_grouped.fillna("기타")

    # 트리맵 생성
    fig = px.treemap(
        df_grouped,
        path=["diner_category_large", "diner_category_middle", "diner_category_small"],
        values="count",
        color="count",
        color_continuous_scale="Viridis",
        title="카테고리별 식당 분포",
    )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig


def create_category_metrics(diner_df, category_column):
    """
    선택한 카테고리 수준에 따른 메트릭 계산
    """
    diner_df["diner_review_cnt"] = pd.to_numeric(
        diner_df["diner_review_cnt"], errors="coerce"
    )

    # 카테고리별 평균 및 카운트 계산
    metrics = pd.DataFrame(
        {
            "count": diner_df.groupby(category_column).size(),
            "avg_rating": diner_df.groupby(category_column)["diner_review_avg"].mean(),
            "bayesian_avg": diner_df.groupby(category_column)["bayesian_score"].mean(),
            "avg_reviews": diner_df.groupby(category_column)["diner_review_cnt"].mean(),
        }
    )

    # 결측치 처리
    metrics = metrics.fillna(0)
    metrics.columns = ["식당 수", "평균 평점", "베이지안 평균", "평균 리뷰 수"]
    return metrics.sort_values("식당 수", ascending=False)


def create_missing_values_chart(diner_df):
    """
    각 카테고리 수준별 결측치 비율을 시각화합니다.
    """
    category_columns = [
        "diner_category_large",
        "diner_category_middle",
        "diner_category_small",
        "diner_category_detail",
    ]

    # 결측치 계산
    missing_data = {"카테고리 수준": [], "결측치 수": [], "결측치 비율(%)": []}

    total_rows = len(diner_df)

    for col in category_columns:
        missing_count = diner_df[col].isna().sum()
        missing_percent = (missing_count / total_rows) * 100

        missing_data["카테고리 수준"].append(col.replace("diner_category_", ""))
        missing_data["결측치 수"].append(missing_count)
        missing_data["결측치 비율(%)"].append(missing_percent)

    missing_df = pd.DataFrame(missing_data)

    # 막대 그래프 생성
    fig = px.bar(
        missing_df,
        x="카테고리 수준",
        y="결측치 비율(%)",
        text="결측치 수",
        title="카테고리 수준별 결측치 비율",
        color="결측치 비율(%)",
        color_continuous_scale="Reds",
    )

    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(yaxis_range=[0, 100])

    return fig


def category_analysis_page():
    # 주로 diner 데이터만 사용
    diner_df = load_diner_data()

    st.title("카테고리 분석")

    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "카테고리 개요",
            "카테고리별 통계",
            "카테고리 상세 분석",
            "결측치 분석",
        ]
    )

    with tab1:
        st.subheader("카테고리 분포")
        # 트리맵 시각화
        fig = create_category_treemap(diner_df)
        st.plotly_chart(fig, use_container_width=True)

        # 기본 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("대분류 수", diner_df["diner_category_large"].nunique())
        with col2:
            st.metric("중분류 수", diner_df["diner_category_middle"].nunique())
        with col3:
            st.metric("소분류 수", diner_df["diner_category_small"].nunique())

    with tab2:
        st.subheader("카테고리별 통계")
        category_level = st.selectbox(
            "카테고리 레벨 선택:",
            ["middle", "small", "detail"],
            key="stats_category_level",
        )

        category_col = f"diner_category_{category_level}"
        metrics_df = create_category_metrics(diner_df, category_col)

        # 데이터 표시
        st.dataframe(
            metrics_df.style.background_gradient(
                subset=["식당 수", "평균 평점", "베이지안 평균"]
            ),
            use_container_width=True,
        )

        # 시각화
        metric_to_plot = st.selectbox(
            "시각화할 지표 선택:",
            ["식당 수", "평균 평점", "베이지안 평균", "평균 리뷰 수"],
        )

        fig = px.bar(
            metrics_df.reset_index(),
            x=category_col,
            y=metric_to_plot,
            title=f"카테고리별 {metric_to_plot}",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("카테고리 상세 분석")

        # 카테고리 선택
        col1, col2, col3 = st.columns(3)

        with col1:
            middle_categories = ["전체"] + sorted(
                diner_df["diner_category_middle"].dropna().unique().tolist()
            )
            large_cat = st.selectbox("중분류 선택:", middle_categories)

        # 중분류에 따른 필터링
        if large_cat == "전체":
            filtered_df = diner_df
        else:
            filtered_df = diner_df[diner_df["diner_category_middle"] == large_cat]

        # 소분류 선택
        with col2:
            small_categories = ["전체"] + sorted(
                filtered_df["diner_category_small"].dropna().unique().tolist()
            )
            middle_cat = st.selectbox("소분류 선택:", small_categories)

        # 소분류에 따른 필터링
        if middle_cat != "전체":
            filtered_df = filtered_df[filtered_df["diner_category_small"] == middle_cat]

        # 세부분류 선택
        with col3:
            detail_categories = ["전체"] + sorted(
                filtered_df["diner_category_detail"].dropna().unique().tolist()
            )
            small_cat = st.selectbox("세부분류 선택:", detail_categories)

        # 세부분류에 따른 최종 필터링
        if small_cat != "전체":
            filtered_df = filtered_df[filtered_df["diner_category_detail"] == small_cat]

        # 선택된 카테고리 통계
        st.subheader("📊 선택된 카테고리 통계")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("식당 수", len(filtered_df))
        with col2:
            st.metric("평균 평점", f"{filtered_df['diner_review_avg'].mean():.2f}")
        with col3:
            st.metric("평균 리뷰 수", f"{filtered_df['diner_review_cnt'].mean():.0f}")
        with col4:
            st.metric("베이지안 평균", f"{filtered_df['bayesian_score'].mean():.2f}")

        # 상위 식당 목록
        st.subheader("🏆 상위 식당")
        top_restaurants = filtered_df.nlargest(10, "bayesian_score")[
            ["diner_name", "diner_review_avg", "diner_review_cnt", "bayesian_score"]
        ]
        st.dataframe(top_restaurants)

        # 리뷰 점수 분포 - 여기서만 review 데이터 필요
        st.subheader("⭐ 리뷰 점수 분포")
        # 필요할 때만 review 데이터 로드
        review_df = load_review_data()
        category_reviews = review_df[
            review_df["diner_idx"].isin(filtered_df["diner_idx"])
        ]
        fig = px.histogram(
            category_reviews,
            x="reviewer_review_score",
            nbins=10,
            title="리뷰 점수 분포",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("카테고리 결측치 분석")

        # 결측치 비율 시각화
        missing_fig = create_missing_values_chart(diner_df)
        st.plotly_chart(missing_fig, use_container_width=True)


if __name__ == "__main__":
    category_analysis_page()
