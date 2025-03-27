import logging
from pathlib import Path
from typing import Any, Dict, List, Self

import pandas as pd
import yaml

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CategoryProcessor:
    """
    카테고리 처리 클래스로, YAML 설정 파일에 따라 데이터프레임의 카테고리 정보를 변환합니다.

    Attributes:
        df (pd.DataFrame): 처리할 데이터프레임 복사본.
        mappings (Dict[str, Any]): YAML 파일에서 로드한 카테고리 매핑 정보.

    Examples:
        >>> processor = CategoryProcessor(df)
        >>> processed_df = (
        ...     processor.process_lowering_categories()
        ...              .process_partly_lowering_categories()
        ...              .process_chicken_categories()
        ...              .df
        ... )
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        카테고리 프로세서를 초기화합니다.

        Args:
            df (pd.DataFrame): 전처리할 데이터프레임.
        """
        self.df: pd.DataFrame = df.copy()
        self.mappings: Dict[str, Any] = self._load_category_mappings()
        self.category_depth: List[List[str]] = [
            ["detail", "small"],
            ["small", "middle"],
            ["middle", "large"],
        ]

    def _load_category_mappings(self) -> Dict[str, Any]:
        """
        config/data/category_mappings.yaml에서 카테고리 매핑 설정을 로드합니다.

        Returns:
            Dict[str, Any]: 로드된 카테고리 매핑 정보.

        Raises:
            FileNotFoundError: 설정 파일이 존재하지 않는 경우.
            yaml.YAMLError: YAML 파일 파싱 오류 발생 시.
        """
        config_path: Path = (
            Path(__file__).parents[2] / "config" / "data" / "category_mappings.yaml"
        )
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                mappings = yaml.safe_load(f)
            return mappings
        except Exception as e:
            logger.error(f"Failed to load category mappings from {config_path}: {e}")
            raise

    def process_all(self) -> Self:
        """
        모든 카테고리 처리 함수를 순차적으로 실행합니다.

        Returns:
            CategoryProcessor: 연쇄 호출이 가능한 self.
        """
        return (
            self.process_chicken_categories()
            .process_lowering_categories()
            .process_partly_lowering_categories()
        )

    def process_lowering_categories(self) -> Self:
        """
        대분류 카테고리 조정을 처리합니다.
        mappings의 lowering_large_categories 설정에 따라, 해당 before 카테고리를 after 카테고리로 변경하고,
        기존의 중분류와 소분류는 한 단계 아래로 이동시킵니다.

        Returns:
            CategoryProcessor: 연쇄 호출이 가능한 self.
        """
        mappings: Dict[str, List[str]] = self.mappings["lowering_large_categories"]
        for after_category, before_categories in mappings.items():
            target_rows: pd.Series = self.df["diner_category_large"].isin(
                before_categories
            )
            self._shift_categories_down(target_rows)
            self.df.loc[target_rows, "diner_category_large"] = after_category
        return self

    def process_partly_lowering_categories(self) -> Self:
        """
        부분적 카테고리 조정을 처리합니다.
        mappings의 partly_lowering_large_categories 설정에 따라,
        특정 조건을 만족하는 경우 대분류 및 중분류를 변경합니다.

        Returns:
            CategoryProcessor: 연쇄 호출이 가능한 self.
        """
        mappings: Dict[str, Any] = self.mappings["partly_lowering_large_categories"]
        for after_category, category_info in mappings.items():
            # 설정 추출
            part_large: str = category_info["partly_lowering_large_category"]
            part_middle: List[str] = category_info["partly_lowering_middle_categories"]
            new_middle: str = category_info["changing_middle_category"]
            before_categories: List[str] = category_info["lowering_large_categories"]

            # 부분 카테고리 업데이트: 기존 대분류가 part_large이면서 중분류가 part_middle인 경우
            update_condition: pd.Series = (
                self.df["diner_category_large"] == part_large
            ) & (self.df["diner_category_middle"].isin(part_middle))
            self.df.loc[update_condition, "diner_category_large"] = after_category

            # 전체 카테고리 업데이트: before_categories에 해당하는 경우
            target_rows: pd.Series = self.df["diner_category_large"].isin(
                before_categories
            )
            self._shift_categories_down(target_rows)
            self.df.loc[target_rows, "diner_category_middle"] = new_middle
            self.df.loc[target_rows, "diner_category_large"] = after_category
        return self

    def process_chicken_categories(
        self, target_categories: List[str] = ["치킨"]
    ) -> Self:
        """
        치킨 카테고리에 대한 특수 처리를 수행합니다.
        - '치킨' 대분류의 경우, 소분류를 기존 중분류 값으로 이동합니다.
        - mappings의 chicken_category 설정을 참고하여, '구이'에 해당하는 경우 '구운치킨'으로,
          그렇지 않으면 '프라이드치킨'으로 중분류를 업데이트합니다.

        Args:
            target_categories (List[str], optional): 처리 대상 대분류 목록. 기본값은 ["치킨"].

        Returns:
            CategoryProcessor: 연쇄 호출이 가능한 self.
        """
        chicken_config: Dict[str, Any] = self.mappings["chicken_category"]
        target_rows: pd.Series = self.df["diner_category_large"].isin(
            target_categories
        ) & (
            ~self.df["diner_category_middle"].isin(
                ["프라이드치킨", "구운치킨", "닭강정"]
            )
        )

        # 기존 중분류 값을 소분류로 이동
        self.df.loc[target_rows, "diner_category_small"] = self.df.loc[
            target_rows, "diner_category_middle"
        ]

        grilled_chicken: List[str] = chicken_config["구이"]
        is_grilled: pd.Series = self.df["diner_category_middle"].isin(grilled_chicken)

        self.df.loc[target_rows & is_grilled, "diner_category_middle"] = "구이"
        self.df.loc[target_rows & ~is_grilled, "diner_category_middle"] = "프라이드"

        return self

    def _shift_categories_down(
        self, target_rows: pd.Series, target_category: str = "diner_category_large"
    ) -> None:
        """
        대상 행에 대해 카테고리를 한 단계 아래로 이동시킵니다.
        변경되는 대상 카테고리에 따라 이동 패턴이 달라집니다:
        - diner_category_large일 경우: 'middle→small', 'small→detail' 이동
        - diner_category_middle일 경우: 'small→detail' 이동만 수행

        Args:
            target_rows (pd.Series): 이동할 행을 나타내는 불리언 시리즈.
            target_category (str): 이동 대상 카테고리 (기본값: "diner_category_large")
        """

        if target_category == "diner_category_middle":
            # 중분류 변경 시 소분류→상세분류 이동만 수행
            category_depth = self.category_depth[:2]
        else:
            category_depth = self.category_depth

        for cat in category_depth:
            self.df.loc[target_rows, f"diner_category_{cat[0]}"] = self.df.loc[
                target_rows, f"diner_category_{cat[1]}"
            ]
