import pandera as pa
from pandera.typing import Series
from datetime import datetime

class ReviewerSchema(pa.DataFrameModel):
    reviewer_id: Series[int] = pa.Field(nullable=False, unique=True)
    reviewer_user_name: Series[str] = pa.Field(nullable=True)
    reviewer_avg: Series[float] = pa.Field(nullable=True)
    badge_grade: Series[str] = pa.Field(nullable=True)
    badge_level: Series[int] = pa.Field(nullable=True)

class CategorySchema(pa.DataFrameModel):
    diner_idx: Series[int] = pa.Field(nullable=False, unique=True)
    industry_category: Series[str] = pa.Field(nullable=True)
    diner_category_large: Series[str] = pa.Field(nullable=True)
    diner_category_middle: Series[str] = pa.Field(nullable=True)
    diner_category_small: Series[str] = pa.Field(nullable=True)

class ReviewSchema(pa.DataFrameModel):
    diner_idx: Series[int] = pa.Field(nullable=False)
    reviewer_id: Series[int] = pa.Field(nullable=False)
    review_id: Series[int] = pa.Field(nullable=False, unique=True)
    reviewer_review: Series[str] = pa.Field(nullable=True)
    reviewer_review_date: Series[datetime] = pa.Field(nullable=True)
    reviewer_review_score: Series[float] = pa.Field(nullable=True)

class DinerSchema(pa.DataFrameModel):
    diner_idx: Series[int] = pa.Field(nullable=False, unique=True)
    diner_name: Series[str] = pa.Field(nullable=True)
    diner_tag: Series[str] = pa.Field(nullable=True)
    diner_menu_name: Series[str] = pa.Field(nullable=True)
    diner_menu_price: Series[str] = pa.Field(nullable=True)
    diner_review_cnt: Series[int] = pa.Field(nullable=True)
    diner_review_avg: Series[float] = pa.Field(nullable=True)
    diner_blog_review_cnt: Series[float] = pa.Field(nullable=True)
    diner_review_tags: Series[str] = pa.Field(nullable=True)
    diner_road_address: Series[str] = pa.Field(nullable=True)
    diner_num_address: Series[str] = pa.Field(nullable=True)
    diner_phone: Series[str] = pa.Field(nullable=True)
    diner_lat: Series[float] = pa.Field(nullable=True)
    diner_lon: Series[float] = pa.Field(nullable=True)
    diner_open_time: Series[str] = pa.Field(nullable=True)
    diner_open_time_titles: Series[str] = pa.Field(nullable=True)
    diner_open_time_hours: Series[str] = pa.Field(nullable=True)
    diner_open_time_off_days_title: Series[str] = pa.Field(nullable=True)
    diner_open_time_off_days_hours: Series[str] = pa.Field(nullable=True)
    bayesian_score: Series[str] = pa.Field(nullable=True)