from typing import Literal

import pandas as pd

from . import schemas


class DataValidator:
    def __init__(self):
        self.reviewer_schema = schemas.ReviewerSchema
        self.category_schema = schemas.CategorySchema
        self.review_schema = schemas.ReviewerSchema
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
                return self.diner_schema.validate(df)
            case _:
                raise ValueError(
                    f"❌ Unsupported DataFrame name: {name_of_df}. Choose from ['reviewer', 'category', 'review', 'diner']."
                )
