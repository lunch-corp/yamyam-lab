import pandas as pd
import plotly.express as px
import streamlit as st

from apps.components.plots import (
    create_category_bar_chart,
    create_menu_frequency_chart,
    create_scores_comparison_chart,
    create_time_series_chart,
)
from apps.components.utils import (
    analyze_menu_frequency,
    calculate_menu_price_avg,
    calculate_sentiment_score,
    get_average_scores,
    get_category_stats,
    get_reviewer_info,
    load_diner_data,
    load_review_data,
    merge_review_diner,
    parse_keywords_safely,
)


def reviewer_analysis_page():
    # 개별 데이터 로드
    review_df = load_review_data()
    diner_df = load_diner_data()

    st.title("리뷰어 분석")

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

        # 데이터 병합 - 필요할 때만 merge
        merged_df = merge_review_diner(target_reviewer, diner_df)

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
                create_category_bar_chart(
                    middle_stats, "중분류 카테고리별 방문 및 만족도"
                ),
                use_container_width=True,
            )

        with tab2:
            small_stats = get_category_stats(merged_df, "diner_category_small")
            st.plotly_chart(
                create_category_bar_chart(
                    small_stats, "소분류 카테고리별 방문 및 만족도"
                ),
                use_container_width=True,
            )

        # 메뉴 분석
        st.subheader("🍽️ 메뉴 분석")
        menu_counts = analyze_menu_frequency(merged_df["diner_menu_name"]).head(10)
        st.plotly_chart(
            create_menu_frequency_chart(menu_counts), use_container_width=True
        )

        # 평균 점수 비교
        st.subheader("⭐ 평균 점수 비교")
        scores = get_average_scores(merged_df)
        st.plotly_chart(
            create_scores_comparison_chart(scores), use_container_width=True
        )

        # 시간별 방문 패턴
        st.subheader("📅 시간별 방문 패턴")
        st.plotly_chart(create_time_series_chart(merged_df), use_container_width=True)

        # 메뉴 가격 정보
        st.subheader("💰 메뉴 가격 정보")
        menu_price_avg = calculate_menu_price_avg(merged_df["diner_menu_price"])
        st.metric("방문한 식당 평균 메뉴 가격", f"{menu_price_avg:,.0f}원")

        # 키워드 분석
        st.subheader("🔍 리뷰 키워드 분석")

        # 시간에 따른 키워드 감성 변화
        st.write("### 시간에 따른 리뷰 감성 변화")

        # 리뷰별 평균 감성 점수 계산
        keywords_df = merged_df.dropna(subset=["parsed_keywords"])

        # parsed_keywords를 리뷰 ID별로 그룹화
        keywords_grouped = keywords_df.groupby("review_id")["parsed_keywords"].apply(
            lambda x: pd.DataFrame(
                [
                    {
                        "term": kw["term"],
                        "category": kw["category"],
                        "sentiment": float(kw["sentiment"]),
                    }
                    for kw in x
                ]
            )
        )

        sentiment_by_date = keywords_df.copy()
        sentiment_by_date["review_sentiment"] = keywords_grouped.apply(
            calculate_sentiment_score
        )

        # 날짜별로 그룹화하여 평균 감성 점수 계산
        sentiment_by_date["reviewer_review_date"] = pd.to_datetime(
            sentiment_by_date["reviewer_review_date"]
        )
        monthly_sentiment = sentiment_by_date.groupby(
            pd.Grouper(key="reviewer_review_date", freq="M")
        )["review_sentiment"].mean()

        # 시각화
        fig = px.line(
            monthly_sentiment.reset_index(),
            x="reviewer_review_date",
            y="review_sentiment",
            title="월별 평균 리뷰 감성 점수 변화",
            labels={
                "review_sentiment": "평균 감성 점수",
                "reviewer_review_date": "날짜",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        # 리뷰 목록을 날짜순으로 정렬
        review_list = keywords_df.sort_values("reviewer_review_date", ascending=False)

        # 리뷰 ID별로 그룹화
        for review_id, review_group in review_list.groupby("review_id"):
            review = review_group.iloc[0]  # 리뷰 기본 정보
            with st.expander(
                f"리뷰 {review['reviewer_review_date']} - {review['diner_name']}"
            ):
                # 리뷰 원문 표시
                st.write("**리뷰 원문:**")
                st.write(review["reviewer_review"])

                # 키워드 분석 결과 표시
                st.write("**추출된 키워드:**")
                keywords = parse_keywords_safely(review_group["parsed_keywords"])

                if keywords:
                    # 키워드를 카테고리별로 그룹화
                    keywords_by_category = {}
                    for keyword in keywords:
                        category = keyword.get("category", "기타")
                        if category not in keywords_by_category:
                            keywords_by_category[category] = []
                        keywords_by_category[category].append(keyword)

                    # 카테고리별로 키워드 표시
                    for category, category_keywords in keywords_by_category.items():
                        st.write(f"*{category}:*")
                        for keyword in category_keywords:
                            sentiment = float(keyword["sentiment"])
                            sentiment_color = (
                                "green"
                                if sentiment > 0.5
                                else "red"
                                if sentiment < 0.5
                                else "gray"
                            )
                            st.markdown(
                                f"- {keyword['term']} "
                                f"(감성점수: <span style='color: {sentiment_color}'>{sentiment:.2f}</span>)",
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("추출된 키워드가 없습니다.")

                # 구분선 추가
                st.markdown("---")

        # 추가 정보 표시
        with st.expander("상세 정보"):
            st.dataframe(
                merged_df[
                    [
                        "diner_name",
                        "diner_category_small",
                        "diner_category_detail",
                        "diner_review_cnt",
                        "diner_blog_review_cnt",
                        "diner_review_avg",
                        "bayesian_score",
                        "reviewer_review_score",
                        "reviewer_review_date",
                        "reviewer_review",
                    ]
                ]
            )


if __name__ == "__main__":
    reviewer_analysis_page()
