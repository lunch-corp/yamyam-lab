import pandera as pa
from pandera.typing import Series

# 값의 범위 설정
PRICE_MIN: int = 0  # 최소 가격
PRICE_MAX: int = 1_000_000  # 최대 가격
LAT_MIN: float = 33.0  # 최소 위도
LAT_MAX: float = 38.0  # 최대 위도
LON_MIN: float = 125.0  # 최소 경도
LON_MAX: float = 132.0  # 최대 경도
REVIEW_SCORE_MIN: float = 0.0  # 최소 리뷰 점수
REVIEW_SCORE_MAX: float = 5.0  # 최대 리뷰 점수


class ReviewerSchema(pa.DataFrameModel):
    reviewer_id: Series[int] = pa.Field(nullable=False, unique=True)
    reviewer_user_name: Series[str] = pa.Field(nullable=True)
    reviewer_avg: Series[float] = pa.Field(nullable=True)
    badge_grade: Series[str] = pa.Field(nullable=True)
    badge_level: Series[int] = pa.Field(nullable=True)


class CategorySchema(pa.DataFrameModel):
    diner_idx: Series[float] = pa.Field(nullable=False, unique=True, coerce=True)
    industry_category: Series[str] = pa.Field(nullable=True)
    diner_category_large: Series[str] = pa.Field(nullable=True)
    diner_category_middle: Series[str] = pa.Field(nullable=True)
    diner_category_small: Series[str] = pa.Field(nullable=True)


class ReviewSchema(pa.DataFrameModel):
    diner_idx: Series[float] = pa.Field(nullable=False, unique=False, coerce=True)
    reviewer_id: Series[int] = pa.Field(nullable=False)
    review_id: Series[int] = pa.Field(nullable=False, unique=True)
    reviewer_review: Series[str] = pa.Field(nullable=True)
    reviewer_review_date: Series[str] = pa.Field(nullable=True)
    reviewer_review_score: Series[float] = pa.Field(
        nullable=True,
        in_range={"min_value": REVIEW_SCORE_MIN, "max_value": REVIEW_SCORE_MAX},
    )


class DinerSchema(pa.DataFrameModel):
    diner_idx: Series[float] = pa.Field(nullable=False, unique=True, coerce=True)
    diner_name: Series[str] = pa.Field(nullable=True)
    diner_tag: Series[str] = pa.Field(nullable=True)
    diner_menu_name: Series[str] = pa.Field(nullable=True)
    diner_menu_price: Series[str] = pa.Field(nullable=True)
    # "[1, 2, 3]" , 변환전이라 검증이 안됨
    # diner_menu_price: Series[float] = pa.Field(
    #     nullable=True,
    #     in_range={"min_value": PRICE_MIN, "max_value": PRICE_MAX}
    # )
    diner_review_cnt: Series[int] = pa.Field(nullable=True, coerce=True)
    diner_review_avg: Series[float] = pa.Field(nullable=True)
    diner_blog_review_cnt: Series[float] = pa.Field(nullable=True)
    diner_review_tags: Series[str] = pa.Field(nullable=True)
    diner_road_address: Series[str] = pa.Field(nullable=True)
    diner_num_address: Series[str] = pa.Field(nullable=True)
    diner_phone: Series[str] = pa.Field(nullable=True)
    diner_lat: Series[float] = pa.Field(
        nullable=True, in_range={"min_value": LAT_MIN, "max_value": LAT_MAX}
    )
    diner_lon: Series[float] = pa.Field(
        nullable=True, in_range={"min_value": LON_MIN, "max_value": LON_MAX}
    )
    diner_open_time: Series[str] = pa.Field(nullable=True)
    diner_open_time_titles: Series[str] = pa.Field(nullable=True)
    diner_open_time_hours: Series[str] = pa.Field(nullable=True)
    diner_open_time_off_days_title: Series[str] = pa.Field(nullable=True)
    diner_open_time_off_days_hours: Series[str] = pa.Field(nullable=True)
    bayesian_score: Series[float] = pa.Field(nullable=True)
