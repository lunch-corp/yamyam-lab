import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.utils import load_data


def create_category_treemap(diner_df: pd.DataFrame) -> go.Figure:
    """카테고리 트리맵을 생성합니다."""
    category_counts = (
        diner_df.groupby(
            ["diner_category_large", "diner_category_middle", "diner_category_small"]
        )
        .size()
        .reset_index(name="count")
    )

    fig = px.treemap(
        category_counts,
        path=["diner_category_large", "diner_category_middle", "diner_category_small"],
        values="count",
        title="카테고리 분포",
    )
    fig.update_traces(root_color="lightgrey")
    fig.update_layout(height=600)
    return fig

# Function to create a donut chart for a specific category level
def create_donut_chart(data, category_level, title):
    category_counts = data[category_level].value_counts().reset_index()
    category_counts.columns = ['Category', 'Count']

    fig = px.pie(
        category_counts,
        names='Category',
        values='Count',
        hole=0.5,  # Creates the donut shape
        title=title
    )
    return fig


def create_category_metrics(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    """카테고리별 주요 지표를 계산합니다."""
    metrics = (
        df.groupby(category_col)
        .agg(
            {
                "diner_idx": "count",
                "diner_review_avg": "mean",
                "bayesian_score": "mean",
                "diner_review_cnt": "mean",
            }
        )
        .round(2)
    )

    metrics.columns = ["식당 수", "평균 평점", "베이지안 평균", "평균 리뷰 수"]
    return metrics.sort_values("식당 수", ascending=False)


def category_analysis_page():
    st.title("카테고리 분석")

    # 데이터 로드
    with st.spinner("데이터를 불러오는 중..."):
        review_df, diner_df, category_kakao_df, category_frequency_df = load_data()
        

    kakao_raw_category = pd.merge(diner_df[['diner_idx', 'diner_name', 'diner_tag', 'diner_menu', 'diner_menu_name',
        'diner_menu_price', 'diner_review_cnt', 'diner_blog_review_cnt',
        'diner_review_avg', 'diner_review_tags', 'diner_address', 'diner_phone',
        'diner_lat', 'diner_lon', 'diner_url', 'diner_open_time',
        'diner_address_constituency', 'real_good_review_cnt',
        'real_bad_review_cnt', 'all_review_cnt', 'real_good_review_percent',
        'real_bad_review_percent', 'is_small_category_missing',
        'bayesian_score', 'rank']],
        category_kakao_df, on='diner_idx', how='left')
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(
        ["카테고리 개요", "카테고리별 통계", "카테고리 상세 분석"]
    )

    with tab1:
        st.subheader("카테고리 분포")
        
        # diner와 category_df를 diner_idx를 기준으로 병합
        # kakao_raw_category = pd.merge(diner_df[['diner_idx', 'diner_name']],
        #                 category_kakao_df, on='diner_idx', how='left')
        
        kakao_modified_df = kakao_raw_category.copy()
        
        # 트리맵 시각화
        st.dataframe(category_frequency_df)
        fig = create_category_treemap(kakao_raw_category)
        st.plotly_chart(fig, use_container_width=True)
     
        # Streamlit layout
        st.title("Kakao Category Visualization")

        # Donut charts for each category level
        for category_level, title in zip(
            ['diner_category_large', 'diner_category_middle', 'diner_category_small', 'diner_category_detail'],
            ['Large Categories', 'Middle Categories', 'Small Categories', 'Detail Categories']
        ):
            st.subheader(f"{title}")
            fig = create_donut_chart(kakao_raw_category, category_level, f"Distribution of {title}")
            st.plotly_chart(fig, use_container_width=True)

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
                kakao_raw_category["diner_category_large"].fillna('기타').unique().tolist()
            )
            large_cat = st.selectbox("중분류 선택:", middle_categories)

        # 중분류에 따른 필터링
        if large_cat == "전체":
            filtered_df = kakao_raw_category
        else:
            filtered_df = kakao_raw_category[kakao_raw_category["diner_category_large"] == large_cat]

        # 소분류 선택
        with col2:
            small_categories = ["전체"] + sorted(
                filtered_df["diner_category_middle"].fillna('기타').unique().tolist()
            )
            middle_cat = st.selectbox("소분류 선택:", small_categories)

        # 소분류에 따른 필터링
        if middle_cat != "전체":
            filtered_df = filtered_df[filtered_df["diner_category_middle"] == middle_cat]

        # 세부분류 선택
        with col3:
            detail_categories = ["전체"] + sorted(
                filtered_df["diner_category_small"].fillna('기타').unique().tolist()
            )
            small_cat = st.selectbox("세부분류 선택:", detail_categories)

        # 세부분류에 따른 최종 필터링
        if small_cat != "전체":
            filtered_df = filtered_df[filtered_df["diner_category_small"] == small_cat]

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

        # 리뷰 점수 분포
        st.subheader("⭐ 리뷰 점수 분포")
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


if __name__ == "__main__":
    category_analysis_page()
