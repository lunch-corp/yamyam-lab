import ast
from typing import Literal

import numpy as np
import pandas as pd

from . import schemas


class DataValidator:
    def __init__(self):
        self.reviewer_schema = schemas.ReviewerSchema
        self.category_schema = schemas.CategorySchema
        self.review_schema = schemas.ReviewSchema
        self.diner_schema = schemas.DinerSchema

    def validate(
        self,
        df: pd.DataFrame,
        name_of_df: Literal["reviewer", "category", "review", "diner"],
    ) -> pd.DataFrame:
        match name_of_df:
            case "reviewer":
                return self.reviewer_schema.validate(df)
            case "category":
                return self.category_schema.validate(df)
            case "review":
                return self.review_schema.validate(df)
            case "diner":
                df = self._simple_diner_preprocess(diner_df=df)
                return self.diner_schema.validate(df)
            case _:
                raise ValueError(
                    f"❌ Unsupported DataFrame name: {name_of_df}. Choose from ['reviewer', 'category', 'review', 'diner']."
                )

    def _safe_convert_to_int(self, value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return np.nan  # 변환 불가 시 NaN으로 변경

    def _simple_diner_preprocess(self, diner_df: pd.DataFrame):
        # int 변환 적용
        diner_df["diner_review_cnt"] = diner_df["diner_review_cnt"].apply(
            self._safe_convert_to_int
        )

        # NaN 값 제거
        diner_df = diner_df.dropna(subset=["diner_review_cnt"])

        # 변환할 컬럼 목록
        columns_to_convert = [
            "diner_menu_name",
            "diner_tag",
            "diner_menu_price",
            "diner_review_tags",
            "diner_open_time_titles",
            "diner_open_time_hours",
            "diner_open_time_off_days_title",
            "diner_open_time_off_days_hours",
        ]

        # NaN을 고려하여 안전하게 ast.literal_eval 적용
        for col in columns_to_convert:
            diner_df.loc[:, col] = diner_df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            diner_df.loc[:, col] = diner_df[col].apply(
                lambda x: [] if isinstance(x, float) and pd.isna(x) else x
            )

        return diner_df
