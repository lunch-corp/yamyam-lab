import streamlit as st
from pages.reviewer_analysis import reviewer_analysis_page
from pages.diner_analysis import diner_analysis_page
from pages.category_analysis import category_analysis_page

st.set_page_config(page_title="맛집 분석 대시보드", page_icon="🍽️", layout="wide")


def main():
    # 사이드바에 페이지 선택
    page = st.sidebar.selectbox(
        "분석 페이지 선택", ["리뷰어 분석", "식당 분석", "카테고리 분석"]
    )

    # 선택된 페이지 표시
    if page == "리뷰어 분석":
        reviewer_analysis_page()
    elif page == "식당 분석":
        diner_analysis_page()
    else:
        category_analysis_page()


if __name__ == "__main__":
    main()
