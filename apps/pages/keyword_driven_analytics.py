import pandas as pd
import plotly.express as px
import streamlit as st

from apps.components.utils import (
    analyze_keywords,
    get_word_cloud_data,
    load_diner_data,
    load_keyword_data,
    load_review_data,
    merge_review_keywords,
)


def extract_city_info(address):
    """주소에서 시도와 구 정보를 추출합니다."""
    if pd.isna(address):
        return "기타", "기타"

    address_list = address.split(" ")

    return address_list[0], address_list[1]


def process_keyword_data(review_df, diner_df):
    # 키워드 데이터 로드 및 merge
    keyword_df = load_keyword_data()
    review_with_keywords = merge_review_keywords(review_df, keyword_df)

    # parsed_keywords가 null이 아닌 데이터만 필터링
    keyword_df = review_with_keywords[
        review_with_keywords["parsed_keywords"].notna()
    ].copy()

    # diner 정보 조인
    keyword_df = keyword_df.merge(
        diner_df[
            [
                "diner_idx",
                "diner_name",
                "diner_category_large",
                "diner_category_middle",
                "diner_num_address",
            ]
        ],
        on="diner_idx",
        how="left",
    )

    # 지역 정보 추출 (시도와 구)
    city_info = keyword_df["diner_num_address"].apply(extract_city_info)
    keyword_df["region"] = city_info.apply(lambda x: x[0])
    keyword_df["city"] = city_info.apply(
        lambda x: x[1]
    )  # 기존 호환성을 위해 구 정보는 city에 저장

    return keyword_df


def analyze_keywords_by_city(keyword_df):
    """개선된 지역별 키워드 분석 (워드 클라우드 및 카테고리 분석 포함)"""

    # 지역 선택 기능
    filtered_df, region_name = analyze_keywords_by_city_with_selection(keyword_df)
    if filtered_df is None:
        return

    # 탭으로 분석 결과 구분
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 분포 분석",
            "☁️ 워드 클라우드",
            "🏪 카테고리별 분석",
            "📈 상세 통계",
        ]
    )

    with tab1:
        st.subheader("키워드 분포 분석")

        # 지역별 키워드 카운트 (시도 단위로 표시)
        city_keywords = []
        for _, row in filtered_df.iterrows():
            keyword = row["parsed_keywords"]
            city_keywords.append(
                {
                    "region": row["region"],
                    "city": row["city"],
                    "category": keyword["category"],
                    "sentiment": "긍정"
                    if float(keyword["sentiment"]) > 0.5
                    else "부정",
                    "count": 1,
                }
            )

        city_df = pd.DataFrame(city_keywords)

        # 시도별 키워드 카테고리 분포
        if len(city_df["region"].unique()) > 1:
            fig_category = px.bar(
                city_df.groupby(["region", "category"])["count"].sum().reset_index(),
                x="region",
                y="count",
                color="category",
                title=f"{region_name} - 시도별 키워드 카테고리 분포",
            )
            st.plotly_chart(fig_category, use_container_width=True)

        # 긍/부정 분포
        sentiment_df = (
            city_df.groupby(["region", "sentiment"])["count"].sum().reset_index()
        )

        # 퍼센트 계산
        total_by_region = sentiment_df.groupby("region")["count"].transform("sum")
        sentiment_df["percent"] = sentiment_df["count"] / total_by_region

        fig_sentiment = px.bar(
            sentiment_df,
            x="region",
            y="count",
            color="sentiment",
            title=f"{region_name} - 긍/부정 분포",
            text=sentiment_df["percent"].apply(lambda x: f"{x:.0%}"),
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with tab2:
        st.subheader("워드 클라우드")
        st.dataframe(filtered_df)
        keyword_df, positive_keywords, negative_keywords = analyze_keywords(
            filtered_df["parsed_keywords"]
        )
        col1, col2 = st.columns(2)

        with col1:
            st.write("**긍정 키워드**")
            get_word_cloud_data(positive_keywords, region_name)

        with col2:
            st.write("**부정 키워드**")
            get_word_cloud_data(negative_keywords, region_name)

    with tab3:
        st.subheader("음식점 카테고리별 분석")

        # 카테고리별 키워드 분석
        category_keywords = []
        for _, row in filtered_df.iterrows():
            keyword = row["parsed_keywords"]
            category_keywords.append(
                {
                    "diner_category": row["diner_category_large"]
                    if pd.notna(row["diner_category_large"])
                    else "기타",
                    "keyword_category": keyword["category"],
                    "sentiment": "긍정"
                    if float(keyword["sentiment"]) > 0.5
                    else "부정",
                    "count": 1,
                }
            )

        category_df = pd.DataFrame(category_keywords)

        # 음식점 카테고리별 긍/부정 분포
        fig_category_sentiment = px.bar(
            category_df.groupby(["diner_category", "sentiment"])["count"]
            .sum()
            .reset_index(),
            x="diner_category",
            y="count",
            color="sentiment",
            title=f"{region_name} - 음식점 카테고리별 긍/부정 분포",
        )
        fig_category_sentiment.update_xaxes(tickangle=45)
        st.plotly_chart(fig_category_sentiment, use_container_width=True)

        # 음식점 카테고리별 키워드 카테고리 분포
        fig_keyword_category = px.bar(
            category_df.groupby(["diner_category", "keyword_category"])["count"]
            .sum()
            .reset_index(),
            x="diner_category",
            y="count",
            color="keyword_category",
            title=f"{region_name} - 음식점 카테고리별 키워드 카테고리 분포",
        )
        fig_keyword_category.update_xaxes(tickangle=45)
        st.plotly_chart(fig_keyword_category, use_container_width=True)

    with tab4:
        st.subheader("상세 통계")

        # 기본 통계
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("총 키워드 수", f"{len(filtered_df):,}")

        with col2:
            positive_count = len(
                [
                    1
                    for _, row in filtered_df.iterrows()
                    if float(row["parsed_keywords"]["sentiment"]) > 0.5
                ]
            )
            positive_rate = positive_count / len(filtered_df) * 100
            st.metric("긍정 비율", f"{positive_rate:.1f}%")

        with col3:
            unique_diners = filtered_df["diner_idx"].nunique()
            st.metric("분석 음식점 수", f"{unique_diners:,}")

        with col4:
            unique_categories = filtered_df["diner_category_large"].nunique()
            st.metric("음식점 카테고리 수", f"{unique_categories}")

        # 상세 데이터 테이블
        st.subheader("키워드 상세 데이터")
        display_df = filtered_df[
            [
                "diner_name",
                "region",
                "city",
                "diner_category_large",
                "parsed_keywords",
            ]
        ].copy()
        display_df["키워드"] = display_df["parsed_keywords"].apply(
            lambda x: x.get("keyword", "") if isinstance(x, dict) else ""
        )
        display_df["카테고리"] = display_df["parsed_keywords"].apply(
            lambda x: x.get("category", "") if isinstance(x, dict) else ""
        )
        display_df["감정점수"] = display_df["parsed_keywords"].apply(
            lambda x: f"{float(x.get('sentiment', 0)):.2f}"
            if isinstance(x, dict)
            else ""
        )

        st.dataframe(
            display_df[
                [
                    "diner_name",
                    "region",
                    "city",
                    "diner_category_large",
                    "키워드",
                    "카테고리",
                    "감정점수",
                ]
            ].rename(
                columns={
                    "diner_name": "음식점명",
                    "region": "시도",
                    "city": "구/군",
                    "diner_category_large": "음식점 카테고리",
                }
            ),
            use_container_width=True,
        )


def analyze_keywords_by_city_with_selection(keyword_df):
    """지역 선택 기능이 있는 키워드 분석"""
    st.subheader("📍 지역별 키워드 분석")

    # 사이드바에 지역 선택 옵션 추가
    col1, col2 = st.columns(2)

    with col1:
        # 시도 선택
        available_region = sorted(keyword_df["region"].unique())
        selected_region = st.selectbox(
            "시도 선택", options=["전체"] + available_region, index=0
        )

    with col2:
        # 선택된 시도에 따른 구 필터링
        if selected_region == "전체":
            available_citys = sorted(keyword_df["city"].unique())
        else:
            available_citys = sorted(
                keyword_df[keyword_df["region"] == selected_region]["city"].unique()
            )

        selected_city = st.selectbox(
            "구/군 선택", options=["전체"] + available_citys, index=0
        )

    # 데이터 필터링
    filtered_df = keyword_df.copy()
    if selected_region != "전체":
        filtered_df = filtered_df[filtered_df["region"] == selected_region]
    if selected_city != "전체":
        filtered_df = filtered_df[filtered_df["city"] == selected_city]

    if len(filtered_df) == 0:
        st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
        return None, None

    # 선택된 지역 정보 표시
    region_name = f"{selected_region}" if selected_region != "전체" else "전국"
    if selected_city != "전체":
        region_name += f" {selected_city}"

    st.info(f"📍 분석 지역: **{region_name}** (총 {len(filtered_df):,}개 키워드)")

    return filtered_df, region_name


def main():
    st.title("🔍 키워드 기반 분석")

    st.markdown("""
    이 페이지에서는 리뷰 키워드 분석 결과를 지역별, 카테고리별로 상세히 분석할 수 있습니다.

    **주요 기능:**
    - 🗺️ 시도/구 단위 지역 선택
    - ☁️ 긍정/부정 키워드 워드 클라우드
    - 🏪 음식점 카테고리별 키워드 분석
    - 📊 상세 통계 및 데이터 탐색
    """)

    # 개별 데이터 로드
    with st.spinner("데이터를 로딩하는 중..."):
        review_df = load_review_data()
        diner_df = load_diner_data()

        # 키워드 데이터 처리
        keyword_df = process_keyword_data(review_df, diner_df)

    st.success(f"총 {len(keyword_df):,}개의 키워드 데이터를 로드했습니다.")

    # 통합 분석 실행
    analyze_keywords_by_city(keyword_df)


if __name__ == "__main__":
    main()
