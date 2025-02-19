from pathlib import Path

import yaml


class CategoryProcessor:
    def __init__(self, df):
        """
        카테고리 처리를 위한 프로세서 초기화

        Args:
            df (pd.DataFrame): 처리할 데이터프레임
        """
        self.df = df.copy()
        self.mappings = self._load_category_mappings()

    def _load_category_mappings(self):
        """카테고리 매핑 설정을 YAML 파일에서 로드"""
        config_path = (
            Path(__file__).parents[2] / "config" / "data" / "category_mappings.yaml"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def process_all(self):
        """모든 카테고리 처리를 순차적으로 실행"""
        return (
            self.process_lowering_categories()
            .process_partly_lowering_categories()
            .process_chicken_categories()
        )

    def process_lowering_categories(self):
        """대분류 카테고리 조정 처리"""
        mappings = self.mappings["lowering_large_categories"]
        for after_category, before_categories in mappings.items():
            target_rows = self.df["diner_category_large"].isin(before_categories)
            self._shift_categories_down(target_rows)
            self.df.loc[target_rows, "diner_category_large"] = after_category
        return self

    def process_partly_lowering_categories(self):
        """부분적 카테고리 조정 처리"""
        mappings = self.mappings["partly_lowering_large_categories"]
        for after_category, category_info in mappings.items():
            # 설정 추출
            part_large = category_info["partly_lowering_large_category"]
            part_middle = category_info["partly_lowering_middle_categories"]
            new_middle = category_info["changing_middle_categorie"]
            before_categories = category_info["lowering_large_categories"]

            # 부분 카테고리 업데이트
            update_condition = (self.df["diner_category_large"] == part_large) & (
                self.df["diner_category_middle"].isin(part_middle)
            )
            self.df.loc[update_condition, "diner_category_large"] = after_category

            # 전체 카테고리 업데이트
            target_rows = self.df["diner_category_large"].isin(before_categories)
            self._shift_categories_down(target_rows)
            self.df.loc[target_rows, "diner_category_middle"] = new_middle
            self.df.loc[target_rows, "diner_category_large"] = after_category
        return self

    def process_chicken_categories(self, target_categories=["치킨"]):
        """치킨 카테고리 특수 처리"""
        chicken_config = self.mappings["chicken_category"]

        target_rows = (self.df["diner_category_large"].isin(target_categories)) & (
            ~self.df["diner_category_middle"].isin(
                ["프라이드치킨", "구운치킨", "닭강정"]
            )
        )

        self.df.loc[target_rows, "diner_category_small"] = self.df.loc[
            target_rows, "diner_category_middle"
        ]

        grilled_chicken = chicken_config["구이"]
        is_grilled = self.df["diner_category_middle"].isin(grilled_chicken)

        self.df.loc[target_rows & is_grilled, "diner_category_middle"] = "구운치킨"
        self.df.loc[target_rows & ~is_grilled, "diner_category_middle"] = "프라이드치킨"

        return self

    def _shift_categories_down(self, target_rows):
        """카테고리를 한 단계씩 아래로 이동하는 헬퍼 메서드"""
        self.df.loc[target_rows, "diner_category_detail"] = self.df.loc[
            target_rows, "diner_category_small"
        ]
        self.df.loc[target_rows, "diner_category_small"] = self.df.loc[
            target_rows, "diner_category_middle"
        ]
