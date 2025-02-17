import pandas as pd
import streamlit as st
from components.plots import (create_category_bar_chart,
                              create_menu_frequency_chart,
                              create_scores_comparison_chart,
                              create_time_series_chart)
from components.utils import (analyze_menu_frequency, calculate_menu_price_avg,
                              get_average_scores, get_category_stats,
                              get_reviewer_info, load_data)


def reviewer_analysis_page():
    st.title("리뷰어 분석")

    # 데이터 로드
    with st.spinner("데이터를 불러오는 중..."):
        review_df, diner_df, category_kakao_df = load_data()

    # 사이드바에 리뷰어 ID 입력
    with st.sidebar:
        st.subheader("리뷰어 검색")
        reviewer_id = st.text_input(
            "리뷰어 ID를 입력하세요:",
            value=893438059,
            help="분석하고 싶은 리뷰어의 ID를 입력하세요.",
        )
        search_button = st.button("분석", use_container_width=True)

    if search_button:
        # 리뷰어 정보 가져오기
        target_reviewer, reviewer_name = get_reviewer_info(review_df, int(reviewer_id))

        if target_reviewer is None:
            st.error("해당 리뷰어의 데이터가 없습니다.")
            return

        # 데이터 병합
        merged_df = pd.merge(target_reviewer, diner_df, on="diner_idx")

        # 메인 컨텐츠 영역
        st.header(f"📊 {reviewer_name}({reviewer_id})님의 분석")

        # 기본 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 리뷰 수", len(merged_df))
        with col2:
            satisfaction_rate = (merged_df["score_diff"] > 0).mean() * 100
            st.metric("전체 만족도", f"{satisfaction_rate:.1f}%")
        with col3:
            avg_score = merged_df["reviewer_review_score"].mean()
            st.metric("평균 평점", f"{avg_score:.1f}")

        # 카테고리 분석
        st.subheader("📋 카테고리 분석")
        tab1, tab2 = st.tabs(["중분류", "소분류"])

        with tab1:
            middle_stats = get_category_stats(merged_df, "diner_category_middle")
            st.plotly_chart(
                create_category_bar_chart(middle_stats, "중분류 카테고리별 방문 및 만족도"),
                use_container_width=True,
            )

        with tab2:
            small_stats = get_category_stats(merged_df, "diner_category_small")
            st.plotly_chart(
                create_category_bar_chart(small_stats, "소분류 카테고리별 방문 및 만족도"),
                use_container_width=True,
            )

        # 메뉴 분석
        st.subheader("🍽️ 메뉴 분석")
        menu_counts = analyze_menu_frequency(merged_df["diner_menu_name"]).head(10)
        st.plotly_chart(create_menu_frequency_chart(menu_counts), use_container_width=True)

        # 평균 점수 비교
        st.subheader("⭐ 평균 점수 비교")
        scores = get_average_scores(merged_df)
        st.plotly_chart(create_scores_comparison_chart(scores), use_container_width=True)

        # 시간별 방문 패턴
        st.subheader("📅 시간별 방문 패턴")
        st.plotly_chart(create_time_series_chart(merged_df), use_container_width=True)

        # 메뉴 가격 정보
        st.subheader("💰 메뉴 가격 정보")
        menu_price_avg = calculate_menu_price_avg(merged_df["diner_menu_price"])
        st.metric("방문한 식당 평균 메뉴 가격", f"{menu_price_avg:,.0f}원")

        # 추가 정보 표시
        with st.expander("상세 정보"):
            st.dataframe(
                merged_df[
                    [
                        "diner_name",
                        "diner_category_small",
                        "diner_category_detail",
                        "diner_url",
                        "diner_review_cnt",
                        "diner_blog_review_cnt",
                        "diner_review_avg",
                        "bayesian_score",
                        "reviewer_review_score",
                        "reviewer_review_date",
                        "score_diff",
                    ]
                ]
            )


if __name__ == "__main__":
    reviewer_analysis_page()
