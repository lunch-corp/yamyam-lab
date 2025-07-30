import pandas as pd
import streamlit as st

from apps.components.utils import (
    analyze_keywords,
    get_word_cloud_data,
    load_diner_data,
    load_keyword_data,
    load_review_data,
)


def diner_analysis_page():
    # 개별 데이터 로드
    review_df = load_review_data()
    diner_df = load_diner_data()
    review_keyword_df = load_keyword_data()

    st.title("식당 분석")

    search_button = False

    # 사이드바 - 식당 검색 기능
    with st.sidebar:
        st.subheader("식당 검색")
        search_method = st.radio("검색 방법", ["식당명", "카테고리"])

        if search_method == "식당명":
            diner_name = st.text_input("식당명을 입력하세요:")
            if diner_name:
                filtered_diners = diner_df[
                    diner_df["diner_name"].str.contains(diner_name, na=False)
                ]

                if len(filtered_diners) == 0:
                    st.error("검색 결과가 없습니다.")
                    return

                selected_diner = st.selectbox(
                    "식당을 선택하세요:", filtered_diners["diner_name"].unique()
                )
                search_button = st.button("분석", use_container_width=True)

        else:  # 카테고리로 검색
            categories = {
                "대분류": "diner_category_large",
                "중분류": "diner_category_middle",
                "소분류": "diner_category_small",
            }

            selected_category_type = st.selectbox(
                "카테고리 단계 선택:", list(categories.keys())
            )

            category_col = categories[selected_category_type]
            available_categories = diner_df[category_col].unique()

            selected_category = st.selectbox(
                f"{selected_category_type} 선택:", available_categories
            )

            filtered_diners = diner_df[diner_df[category_col] == selected_category]
            selected_diner = st.selectbox(
                "식당을 선택하세요:", filtered_diners["diner_name"].unique()
            )
            search_button = st.button("분석", use_container_width=True)

    # 메인 페이지에 분석 결과 표시
    if search_button:
        # 선택된 식당 정보
        diner_info = diner_df[diner_df["diner_name"] == selected_diner].iloc[0]
        diner_reviews = review_df[review_df["diner_idx"] == diner_info["diner_idx"]]

        # 분석 결과 표시
        st.header(f"📍 {selected_diner}")

        # 기본 정보 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("전체 리뷰 수", diner_info["diner_review_cnt"])
        with col2:
            st.metric("평균 평점", f"{diner_info['diner_review_avg']:.1f}")
        with col3:
            st.metric("베이지안 평균", f"{diner_info['bayesian_score']:.2f}")

        # 카테고리 정보
        st.subheader("🏷️ 카테고리 정보")
        cat_col1, cat_col2, cat_col3 = st.columns(3)
        with cat_col1:
            st.write("대분류:", diner_info["diner_category_large"])
        with cat_col2:
            st.write("중분류:", diner_info["diner_category_middle"])
        with cat_col3:
            st.write("소분류:", diner_info["diner_category_small"])

        # 메뉴 분석
        st.subheader("🍽️ 메뉴 정보")
        menu_names = diner_info.get("diner_menu_name", [])
        menu_prices = diner_info.get("diner_menu_price", [])

        if menu_names and isinstance(menu_names, list):
            # 메뉴와 가격을 함께 표시
            menu_data = pd.DataFrame(
                {
                    "메뉴": menu_names,
                    "가격": (
                        menu_prices
                        if len(menu_prices) == len(menu_names)
                        else ["N/A"] * len(menu_names)
                    ),
                }
            )
            st.dataframe(menu_data, use_container_width=True)
        else:
            st.info("메뉴 정보가 없습니다.")

        # 태그 정보
        st.subheader("🏷️ 태그")
        tags = diner_info.get("diner_tag", [])
        if tags and isinstance(tags, list):
            st.write(", ".join(tags))
        else:
            st.info("태그 정보가 없습니다.")

        # 리뷰 분석
        if len(diner_reviews) > 0:
            st.subheader("📝 리뷰 분석")

            # 시간별 평점 추이
            diner_reviews["reviewer_review_date"] = pd.to_datetime(
                diner_reviews["reviewer_review_date"]
            )
            monthly_scores = diner_reviews.groupby(
                pd.Grouper(key="reviewer_review_date", freq="M")
            )["reviewer_review_score"].mean()

            st.line_chart(monthly_scores)

            # 키워드 분석
            st.subheader("🔍 키워드 분석")

            # 키워드 데이터 분석
            diner_reviews = pd.merge(diner_reviews, review_keyword_df, on="review_id")
            keyword_df, positive_keywords, negative_keywords = analyze_keywords(
                diner_reviews["parsed_keywords"]
            )
            # 긍정/부정 키워드 탭
            keyword_tab1, keyword_tab2 = st.tabs(["긍정 키워드", "부정 키워드"])

            with keyword_tab1:
                if len(positive_keywords) > 0:
                    get_word_cloud_data(positive_keywords, selected_diner)
                else:
                    st.info("긍정 키워드가 없습니다.")

            with keyword_tab2:
                if len(negative_keywords) > 0:
                    get_word_cloud_data(negative_keywords, selected_diner)
                else:
                    st.info("부정 키워드가 없습니다.")

        # 위치 정보
        st.subheader("📍 위치 정보")
        if pd.notna(diner_info["diner_lat"]) and pd.notna(diner_info["diner_lon"]):
            st.map(
                pd.DataFrame(
                    {
                        "lat": [diner_info["diner_lat"]],
                        "lon": [diner_info["diner_lon"]],
                    }
                )
            )
        st.write(diner_info["diner_road_address"])


if __name__ == "__main__":
    diner_analysis_page()
