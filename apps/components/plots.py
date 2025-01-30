import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict


def create_category_bar_chart(
    stats_df: pd.DataFrame, title: str = "카테고리별 방문 및 만족도"
) -> go.Figure:
    """
    카테고리별 전체 방문 수와 만족한 방문 수를 보여주는 이중 막대 그래프를 생성합니다.

    Args:
        stats_df: 카테고리별 통계가 담긴 데이터프레임 ('총 방문', '만족' 컬럼 필요)
        title: 그래프 제목

    Returns:
        plotly Figure 객체
    """
    fig = go.Figure(
        data=[
            go.Bar(
                name="총 방문",
                x=stats_df.index,
                y=stats_df["총 방문"],
                text=stats_df["총 방문"],
                textposition="auto",
            ),
            go.Bar(
                name="만족",
                x=stats_df.index,
                y=stats_df["만족"],
                text=stats_df["만족도(%)"].apply(lambda x: f"{x:.1f}%"),
                textposition="auto",
            ),
        ]
    )

    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="카테고리",
        yaxis_title="방문 횟수",
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def create_menu_frequency_chart(
    menu_counts: pd.Series, title: str = "메뉴별 주문 빈도"
) -> go.Figure:
    """
    메뉴 주문 빈도를 보여주는 막대 그래프를 생성합니다.

    Args:
        menu_counts: 메뉴별 주문 횟수가 담긴 Series
        title: 그래프 제목

    Returns:
        plotly Figure 객체
    """
    fig = px.bar(x=menu_counts.index, y=menu_counts.values, text=menu_counts.values)

    fig.update_layout(
        title=title,
        xaxis_title="메뉴",
        yaxis_title="주문 횟수",
        height=400,
        showlegend=False,
    )

    fig.update_traces(textposition="auto")

    return fig


def create_scores_comparison_chart(
    scores: Dict[str, float], title: str = "평균 점수 비교"
) -> go.Figure:
    """
    여러 점수들을 비교하는 막대 그래프를 생성합니다.

    Args:
        scores: 점수 이름과 값이 담긴 딕셔너리
        title: 그래프 제목

    Returns:
        plotly Figure 객체
    """
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(scores.keys()),
                y=list(scores.values()),
                text=[f"{v:.2f}" for v in scores.values()],
                textposition="auto",
                marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="점수 유형",
        yaxis_title="점수",
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, 5]),  # 점수는 보통 0-5 사이
    )

    return fig


def create_time_series_chart(
    merged_df: pd.DataFrame, title: str = "시간별 방문 패턴"
) -> go.Figure:
    """
    시간에 따른 방문 패턴을 보여주는 선 그래프를 생성합니다.

    Args:
        merged_df: 리뷰 데이터가 담긴 데이터프레임
        title: 그래프 제목

    Returns:
        plotly Figure 객체
    """
    # 날짜별 방문 횟수 계산
    merged_df["reviewer_review_date"] = pd.to_datetime(
        merged_df["reviewer_review_date"]
    )
    daily_visits = (
        merged_df.groupby("reviewer_review_date").size().reset_index(name="visits")
    )

    fig = go.Figure(
        data=go.Scatter(
            x=daily_visits["reviewer_review_date"],
            y=daily_visits["visits"],
            mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title="방문 횟수",
        height=400,
        showlegend=False,
    )

    return fig
