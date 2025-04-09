import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from apps.components.utils import load_data


def analyze_diner_review_counts(diner_df):
    """
    diner_review_cnt가 null이거나 0 이하인 음식점 데이터를 분석합니다.
    """
    # null 또는 0 이하 값을 가진 음식점 확인
    diner_df["diner_review_cnt"] = pd.to_numeric(
        diner_df["diner_review_cnt"], errors="coerce"
    )

    # 세 그룹으로 나누기
    null_reviews = diner_df[diner_df["diner_review_cnt"].isna()]
    zero_reviews = diner_df[
        (diner_df["diner_review_cnt"] == 0) | (diner_df["diner_review_cnt"] < 0)
    ]
    valid_reviews = diner_df[
        (diner_df["diner_review_cnt"] > 0) & (~diner_df["diner_review_cnt"].isna())
    ]

    # 비율 계산
    null_count = len(null_reviews)
    zero_count = len(zero_reviews)
    valid_count = len(valid_reviews)

    # 파이 차트 데이터
    labels = ["리뷰 수 NULL", "리뷰 수 0 이하", "리뷰 있음"]
    values = [null_count, zero_count, valid_count]

    # 파이 차트 생성
    fig_pie = px.pie(
        names=labels,
        values=values,
        title="음식점 리뷰 수 현황",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )

    # 텍스트에 개수와 비율 표시
    fig_pie.update_traces(
        textinfo="percent+value", texttemplate="%{percent:.1f}% (%{value:,})"
    )

    # 카테고리별 리뷰 수 분포 (boxplot)
    if "diner_category_large" in diner_df.columns:
        # 상위 10개 카테고리만 선택
        top_categories = (
            diner_df["diner_category_large"].value_counts().head(10).index.tolist()
        )
        filtered_df = diner_df[diner_df["diner_category_large"].isin(top_categories)]

        # 박스플롯 생성
        fig_box = px.box(
            filtered_df,
            x="diner_category_large",
            y="diner_review_cnt",
            title="주요 카테고리별 리뷰 수 분포",
            color="diner_category_large",
            log_y=True,  # 로그 스케일 사용 (분포 차이가 클 경우)
        )
        fig_box.update_layout(showlegend=False)

        return fig_pie, fig_box

    return fig_pie, None


def analyze_reviewer_counts(review_df):
    """
    reviewer당 작성한 리뷰 수를 분석합니다.
    """
    # reviewer별 리뷰 수 계산
    reviewer_counts = review_df["reviewer_id"].value_counts().reset_index()
    reviewer_counts.columns = ["reviewer_id", "review_count"]

    # 리뷰 수별 reviewer 수 계산
    count_distribution = (
        reviewer_counts["review_count"].value_counts().sort_index().reset_index()
    )
    count_distribution.columns = ["리뷰 작성 수", "리뷰어 수"]

    # 리뷰 작성 수가 10개 이하인 데이터만 필터링 (대부분이 이 범위에 있을 것으로 예상)
    count_distribution_filtered = count_distribution[
        count_distribution["리뷰 작성 수"] <= 10
    ]

    # 막대그래프 생성
    fig_bar = px.bar(
        count_distribution_filtered,
        x="리뷰 작성 수",
        y="리뷰어 수",
        title="리뷰어별 리뷰 작성 수 분포 (10개 이하)",
        color="리뷰어 수",
        text="리뷰어 수",
    )
    fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside")

    # 리뷰 수 구간별 비율 계산
    bins = [1, 2, 3, 5, 10, 20, 50, 100, float("inf")]
    labels = [
        "1개",
        "2개",
        "3-4개",
        "5-9개",
        "10-19개",
        "20-49개",
        "50-99개",
        "100개 이상",
    ]

    reviewer_counts["review_count_bin"] = pd.cut(
        reviewer_counts["review_count"], bins=bins, labels=labels
    )

    bin_counts = reviewer_counts["review_count_bin"].value_counts().sort_index()

    # 파이 차트 생성
    fig_pie = px.pie(
        names=bin_counts.index,
        values=bin_counts.values,
        title="리뷰어별 리뷰 작성 수 분포 (구간별)",
        hole=0.4,
    )
    fig_pie.update_traces(
        textinfo="percent+value", texttemplate="%{percent:.1f}% (%{value:,})"
    )

    # 누적 분포 계산
    total_reviewers = len(reviewer_counts)
    cumulative_data = []

    for i, (count, label) in enumerate(zip(bins[:-1], labels)):
        reviewer_count = len(reviewer_counts[reviewer_counts["review_count"] >= count])
        percentage = reviewer_count / total_reviewers * 100
        cumulative_data.append(
            {
                "최소 리뷰 수": label,
                "리뷰어 수": reviewer_count,
                "비율(%)": percentage,
            }
        )

    cumulative_df = pd.DataFrame(cumulative_data)

    # 누적 분포 차트
    fig_line = px.line(
        cumulative_df,
        x="최소 리뷰 수",
        y="비율(%)",
        title="최소 N개 이상 리뷰를 작성한 리뷰어 비율",
        markers=True,
    )

    fig_line.update_layout(yaxis_range=[0, 100])
    fig_line.add_trace(
        go.Scatter(
            x=cumulative_df["최소 리뷰 수"],
            y=cumulative_df["비율(%)"],
            mode="markers+text",
            text=cumulative_df["리뷰어 수"].apply(lambda x: f"{x:,}명"),
            textposition="top center",
        )
    )

    return fig_bar, fig_pie, fig_line


def data_overview_page():
    review_df, diner_df = load_data()

    st.title("📊 데이터 개요")

    # 탭 생성
    overview_tab, schema_tab, statistics_tab, review_analysis_tab = st.tabs(
        [
            "데이터셋 소개",
            "스키마 설명",
            "데이터 통계",
            "리뷰 분포 분석",
        ]
    )

    with overview_tab:
        st.write("""
        ## 1. 데이터셋 소개 (Overview)

        이 프로젝트는 음식점 정보와 리뷰 데이터를 분석하여 인사이트를 도출하는 대시보드입니다.
        다음 네 가지 주요 테이블을 사용합니다:
        """)

        overview_data = {
            "테이블명": ["diner", "diner_category", "review", "reviewer"],
            "설명": [
                "음식점의 기본 정보 (이름, 주소, 평점, 메뉴 등)",
                "음식점의 카테고리 정보 (대/중/소 분류)",
                "유저가 남긴 음식점 리뷰 데이터",
                "리뷰어(사용자)에 대한 정보",
            ],
            "주요 컬럼": [
                "diner.diner_idx → diner_category.diner_idx, review.diner_idx",
                "diner_category.diner_idx → diner.diner_idx",
                "review.reviewer_id → reviewer.reviewer_id, diner.diner_idx",
                "reviewer.reviewer_id → review.reviewer_id",
            ],
        }

        st.dataframe(pd.DataFrame(overview_data), hide_index=True)

    with schema_tab:
        st.write("""
        ## 2. 각 테이블의 컬럼 설명 (Schema Details)
        """)

        # 탭을 사용하여 각 테이블의 스키마 표시
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "diner",
                "diner_category",
                "review",
                "reviewer",
            ]
        )

        with tab1:
            st.write("### diner 테이블")
            diner_schema = {
                "컬럼명": [
                    "diner_idx",
                    "diner_name",
                    "diner_tag",
                    "diner_menu_name",
                    "diner_menu_price",
                    "diner_review_cnt",
                    "diner_review_avg",
                    "diner_blog_review_cnt",
                    "diner_review_tags",
                    "diner_road_address",
                    "diner_num_address",
                    "diner_phone",
                    "diner_lat",
                    "diner_lon",
                    "diner_open_time",
                    "diner_open_time_titles",
                    "diner_open_time_hours",
                    "diner_open_time_off_days_title",
                    "diner_open_time_off_days_hours",
                    "bayesian_score",
                ],
                "설명": [
                    "음식점 고유 ID",
                    "음식점 이름",
                    "음식점 태그 (리스트)",
                    "메뉴 이름 (리스트)",
                    "메뉴 가격 (리스트)",
                    "리뷰 수",
                    "평균 평점",
                    "블로그 리뷰 수",
                    "리뷰 태그 (리스트)",
                    "도로명 주소",
                    "지번 주소",
                    "전화번호",
                    "위도",
                    "경도",
                    "영업 시간",
                    "영업 시간 제목 (리스트)",
                    "영업 시간 상세 (리스트)",
                    "휴무일 제목 (리스트)",
                    "휴무일 상세 (리스트)",
                    "베이지안 평점 (가중치 적용)",
                ],
                "데이터 타입": [
                    "float",
                    "string",
                    "list[string]",
                    "list[string]",
                    "list[int]",
                    "int",
                    "float",
                    "float",
                    "list[string]",
                    "string",
                    "string",
                    "string",
                    "float",
                    "float",
                    "string",
                    "list[string]",
                    "list[string]",
                    "list[string]",
                    "list[string]",
                    "float",
                ],
            }
            st.dataframe(pd.DataFrame(diner_schema), hide_index=True)

        with tab2:
            st.write("### diner_category 테이블")
            diner_category_schema = {
                "컬럼명": [
                    "diner_idx",
                    "industry_category",
                    "diner_category_large",
                    "diner_category_large",
                    "diner_category_small",
                ],
                "설명": [
                    "음식점 고유 ID",
                    "산업 카테고리",
                    "대분류 카테고리",
                    "중분류 카테고리",
                    "소분류 카테고리",
                ],
                "데이터 타입": ["float", "string", "string", "string", "string"],
            }
            st.dataframe(pd.DataFrame(diner_category_schema))

        with tab3:
            st.write("### review 테이블")
            review_schema = {
                "컬럼명": [
                    "review_id",
                    "diner_idx",
                    "reviewer_id",
                    "reviewer_review",
                    "reviewer_review_date",
                    "reviewer_review_score",
                ],
                "설명": [
                    "리뷰 고유 ID",
                    "음식점 고유 ID",
                    "리뷰어 고유 ID",
                    "리뷰 텍스트",
                    "리뷰 작성 날짜",
                    "리뷰 평점",
                ],
                "데이터 타입": ["int", "float", "int", "string", "string", "float"],
            }
            st.dataframe(pd.DataFrame(review_schema))

        with tab4:
            st.write("### reviewer 테이블")
            reviewer_schema = {
                "컬럼명": [
                    "reviewer_id",
                    "reviewer_level",
                    "reviewer_review_cnt",
                    "reviewer_avg",
                    "reviewer_follower",
                ],
                "설명": [
                    "리뷰어 고유 ID",
                    "리뷰어 레벨 (등급)",
                    "리뷰어가 작성한 리뷰 수",
                    "리뷰어가 부여한 평균 평점",
                    "리뷰어 팔로워 수",
                ],
                "데이터 타입": ["int", "string", "int", "float", "float"],
            }
            st.dataframe(pd.DataFrame(reviewer_schema))

    with statistics_tab:
        st.write("""
        ## 3. 데이터 규모 및 통계 정보
        """)

        # 데이터 로드 및 통계 계산
        diner_count = len(diner_df["diner_idx"].unique())
        review_count = len(review_df)
        reviewer_count = len(review_df["reviewer_id"].unique())
        category_count = (
            len(diner_df["diner_category_large"].unique())
            + len(diner_df["diner_category_large"].unique())
            + len(diner_df["diner_category_small"].unique())
        )

        # 평점 분포 계산
        rating_counts = review_df["reviewer_review_score"].value_counts().sort_index()
        rating_data = {
            "평점": [str(rate) for rate in rating_counts.index],
            "리뷰 수": rating_counts.values,
        }

        # 카테고리 분포 계산
        if "diner_category_large" in diner_df.columns:
            category_counts = diner_df["diner_category_large"].value_counts().head(10)
            category_data = {
                "카테고리": category_counts.index,
                "음식점 수": category_counts.values,
            }
        else:
            # 예시 데이터를 사용
            category_data = {
                "카테고리": [
                    "한식",
                    "양식",
                    "일식",
                    "중식",
                    "아시아음식",
                    "간식",
                    "술집",
                    "치킨",
                    "퓨전요리",
                    "기타",
                ],
                "음식점 수": [
                    23956,
                    4645,
                    3927,
                    3290,
                    1318,
                    5057,
                    6389,
                    2832,
                    660,
                    970,
                ],
            }

        # 데이터 규모 표시
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="음식점 수", value=f"~{diner_count:,}개")
            st.metric(label="리뷰 수", value=f"~{review_count:,}개")

        with col2:
            st.metric(label="리뷰어 수", value=f"~{reviewer_count:,}명")
            st.metric(label="카테고리 수", value=f"~{category_count}개")

        # 데이터 분포 시각화
        st.write("### 카테고리 대분류별 음식점 분포")

        fig = px.bar(
            pd.DataFrame(category_data),
            x="카테고리",
            y="음식점 수",
            color="음식점 수",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("### 평점 분포")

        fig = px.pie(
            pd.DataFrame(rating_data),
            names="평점",
            values="리뷰 수",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Viridis,
        )
        st.plotly_chart(fig, use_container_width=True)

    with review_analysis_tab:
        st.write("""
        ## 4. 리뷰 분포 분석
        """)

        # 1. 음식점 리뷰 수 분석
        st.markdown("### 1. 음식점 리뷰 수 분석")
        st.write("음식점의 리뷰 수가 null이거나 0 이하인 경우를 분석합니다.")

        review_pie, review_box = analyze_diner_review_counts(diner_df)

        st.plotly_chart(review_pie, use_container_width=True)

        if review_box is not None:
            st.write("주요 카테고리별 리뷰 수 분포:")
            st.plotly_chart(review_box, use_container_width=True)

        # 2. 리뷰어별 리뷰 작성 수 분석
        st.markdown("### 2. 리뷰어별 리뷰 작성 수 분석")
        st.write("각 리뷰어가 작성한 리뷰 수 분포를 분석합니다.")

        reviewer_bar, reviewer_pie, reviewer_line = analyze_reviewer_counts(review_df)

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(reviewer_bar, use_container_width=True)

        with col2:
            st.plotly_chart(reviewer_pie, use_container_width=True)

        st.plotly_chart(reviewer_line, use_container_width=True)

        # 추가 설명
        st.info("""
        **리뷰 분포 분석 결과 해석:**

        1. **음식점 리뷰 수:** null이나 0 이하의 리뷰 수를 가진 음식점은 평점 데이터가 없거나 불완전한 데이터로 볼 수 있습니다.
        2. **리뷰어별 리뷰 작성 수:** 대부분의 리뷰어가 소수의 리뷰만 작성했다면, 소수의 활발한 리뷰어에 의해 데이터가 편향될 수 있습니다.
        3. **최소 N개 이상 리뷰를 작성한 리뷰어 비율** 차트는 일정 수준 이상의 활동성을 가진 리뷰어 비율을 보여줍니다.
        """)


if __name__ == "__main__":
    data_overview_page()
