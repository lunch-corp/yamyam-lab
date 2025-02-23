import pandera as pa
from pandera import Column, DataFrameSchema

# 데이터 스키마 정의
reviewer_schema = DataFrameSchema(
    {
        "reviewer_id": Column(int, nullable=False, unique=True),
        "reviewer_user_name": Column(str, nullable=True),
        "reviewer_avg": Column(float, nullable=True),
        "badge_grade": Column(str, nullable=True),
        "badge_level": Column(int, nullable=True),
    }
)

category_schema = DataFrameSchema(
    {
        "diner_idx": Column(int, nullable=False, unique=True),
        "industry_category": Column(str, nullable=True),
        "diner_category_large": Column(str, nullable=True),
        "diner_category_middle": Column(str, nullable=True),
        "diner_category_small": Column(str, nullable=True),
    }
)

review_schema = DataFrameSchema(
    {
        "diner_idx": Column(int, nullable=False, unique=False),
        "reviewer_id": Column(int, nullable=False, unique=False),
        "review_id": Column(int, nullable=False, unique=True),
        "reviewer_review": Column(str, nullable=True),
        "reviewer_review_date": Column(pa.DateTime, nullable=True),
        "reviewer_review_score": Column(float, nullable=True),
    }
)


diner_schema = DataFrameSchema(
    {
        "diner_idx": Column(int, nullable=False, unique=True),
        "diner_name": Column(str, nullable=True),
        "diner_tag": Column(str, nullable=True),
        "diner_menu_name": Column(str, nullable=True),
        "diner_menu_price": Column(str, nullable=True),
        "diner_review_cnt": Column(int, nullable=True),
        "diner_review_avg": Column(float, nullable=True),
        "diner_blog_review_cnt": Column(float, nullable=True),
        "diner_review_tags": Column(str, nullable=True),
        "diner_road_address": Column(str, nullable=True),
        "diner_num_address": Column(str, nullable=True),
        "diner_phone": Column(str, nullable=True),
        "diner_lat": Column(float, nullable=True),
        "diner_lon": Column(float, nullable=True),
        "diner_open_time": Column(str, nullable=True),
        "diner_open_time_titles": Column(str, nullable=True),
        "diner_open_time_hours": Column(str, nullable=True),
        "diner_open_time_off_days_title": Column(str, nullable=True),
        "diner_open_time_off_days_hours": Column(str, nullable=True),
        "bayesian_score": Column(str, nullable=True),
    }
)
