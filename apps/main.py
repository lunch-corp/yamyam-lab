import streamlit as st

from apps.components.utils import load_data

st.set_page_config(page_title="맛집 분석 대시보드", page_icon="🍽️", layout="wide")


def main():
    review_df, diner_df = load_data()

    # 페이지 구성
    data_overview = st.Page(
        "./pages/data_overview.py",
        title="데이터 개요",
        icon=":material/dashboard:",
    )
    reviewer_analysis = st.Page(
        "./pages/reviewer_analysis.py",
        title="리뷰어 분석",
        icon=":material/person:",
    )
    diner_analysis = st.Page(
        "./pages/diner_analysis.py",
        title="식당 분석",
        icon=":material/restaurant:",
    )
    category_analysis = st.Page(
        "./pages/category_analysis.py",
        title="카테고리 분석",
        icon=":material/category:",
    )

    # 네비게이션 설정
    pages = [data_overview, reviewer_analysis, diner_analysis, category_analysis]

    pg = st.navigation({"맛집 분석 대시보드": pages}, position="sidebar")

    # 페이지 실행
    pg.run()


if __name__ == "__main__":
    main()
