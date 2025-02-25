import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

    def process_all(self) -> "CategoryProcessor":
        """
        모든 카테고리 처리 함수를 순차적으로 실행합니다.

        Returns:
            CategoryProcessor: 연쇄 호출이 가능한 self.
        """
        return (
            self.process_lowering_categories()
            .process_partly_lowering_categories()
            .process_chicken_categories()
        )

    def process_lowering_categories(self) -> "CategoryProcessor":
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

    def process_partly_lowering_categories(self) -> "CategoryProcessor":
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
    ) -> "CategoryProcessor":
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

        self.df.loc[target_rows & is_grilled, "diner_category_middle"] = "구운치킨"
        self.df.loc[target_rows & ~is_grilled, "diner_category_middle"] = "프라이드치킨"

        return self

    def _shift_categories_down(self, target_rows: pd.Series, target_category) -> None:
        """
        대상 행에 대해 카테고리를 한 단계 아래로 이동시킵니다.
        기존 중분류 값은 소분류로, 상위 대분류 값은 중분류로 이동합니다.

        Args:
            target_rows (pd.Series): 이동할 행을 나타내는 불리언 시리즈.
        """
        target_depth = {
            "diner_category_large": [
                "diner_category_detail",
                "diner_category_small",
                "diner_category_middle",
            ],
            "diner_category_middle": ["diner_category_detail", "diner_category_small"],
        }

        category_list = target_depth[target_category]
        self.df.loc[target_rows, "diner_category_detail"] = self.df.loc[
            target_rows, "diner_category_small"
        ]
        self.df.loc[target_rows, "diner_category_small"] = self.df.loc[
            target_rows, "diner_category_middle"
        ]


@dataclass
class DataValidationConfig:
    """
    데이터 검증에 필요한 설정값을 담은 데이터 클래스.

    Attributes:
        PRICE_MIN (int): 가격의 최소 허용값.
        PRICE_MAX (int): 가격의 최대 허용값.
        LAT_MIN (float): 위도의 최소 허용값 (한국 최남단).
        LAT_MAX (float): 위도의 최대 허용값 (한국 최북단).
        LON_MIN (float): 경도의 최소 허용값 (한국 최서단).
        LON_MAX (float): 경도의 최대 허용값 (한국 최동단).
        REVIEW_SCORE_MIN (float): 리뷰 점수의 최소값.
        REVIEW_SCORE_MAX (float): 리뷰 점수의 최대값.
    """

    PRICE_MIN: int = 0
    PRICE_MAX: int = 1_000_000
    LAT_MIN: float = 33.0
    LAT_MAX: float = 38.0
    LON_MIN: float = 125.0
    LON_MAX: float = 132.0
    REVIEW_SCORE_MIN: float = 0.0
    REVIEW_SCORE_MAX: float = 5.0


class DataTypeConverter:
    """
    데이터 타입 변환을 위한 헬퍼 클래스.
    문자열 또는 리스트 형태의 데이터를 적절한 파이썬 타입으로 변환합니다.
    """

    @staticmethod
    def convert_to_list(value: Any) -> List[str]:
        """
        문자열이나 리스트 형태의 데이터를 파이썬 리스트로 변환합니다.

        Args:
            value (Any): 변환할 값.

        Returns:
            List[str]: 변환된 문자열 리스트.
        """
        if pd.isna(value):
            return []

        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]

        if isinstance(value, str):
            try:
                if value.startswith("[") and value.endswith("]"):
                    items = ast.literal_eval(value)
                    return [str(item).strip() for item in items if str(item).strip()]
                elif "@" in value:
                    return [item.strip() for item in value.split("@") if item.strip()]
                else:
                    return [value.strip()] if value.strip() else []
            except Exception as e:
                logger.warning(f"List conversion error: {e}, value: {value}")
                return []

        return []

    @staticmethod
    def convert_price_list(
        value: Any, config: DataValidationConfig
    ) -> Optional[List[int]]:
        """
        가격 데이터를 정수 리스트로 변환합니다.
        문자열 또는 리스트 형태의 가격 정보를 파싱하여 검증 후 정수 리스트로 반환합니다.

        Args:
            value (Any): 변환할 가격 정보.
            config (DataValidationConfig): 가격 검증을 위한 설정값.

        Returns:
            Optional[List[int]]: 검증된 가격 리스트, 유효한 가격이 없으면 None.
        """
        try:
            if pd.isna(value):
                return None

            if isinstance(value, str):
                prices = ast.literal_eval(value)
            elif isinstance(value, list):
                prices = value
            else:
                return None

            validated_prices: List[int] = []
            for price in prices:
                try:
                    price_int = int("".join(filter(str.isdigit, str(price))))
                    if config.PRICE_MIN <= price_int <= config.PRICE_MAX:
                        validated_prices.append(price_int)
                except ValueError:
                    continue

            return sorted(list(set(validated_prices))) if validated_prices else None

        except Exception as e:
            logger.warning(f"Price conversion error: {e}, value: {value}")
            return None


class DataFrameValidator:
    """
    데이터프레임 내의 수치형 데이터를 검증하는 클래스.
    위도, 경도, 리뷰 점수 등의 이상치를 검출하여 NaN 처리합니다.
    """

    def __init__(self, config: DataValidationConfig) -> None:
        """
        초기화 메서드.

        Args:
            config (DataValidationConfig): 검증에 필요한 설정값.
        """
        self.config: DataValidationConfig = config

    def validate_numeric_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        수치형 데이터의 범위를 검증하고, 범위를 벗어나는 값은 NaN으로 처리합니다.

        Args:
            df (pd.DataFrame): 검증할 데이터프레임.

        Returns:
            pd.DataFrame: 검증 후 데이터프레임.
        """
        df_validated: pd.DataFrame = df.copy()

        # 위도/경도 검증
        mask: pd.Series = df["diner_lat"].between(
            self.config.LAT_MIN, self.config.LAT_MAX
        ) & df["diner_lon"].between(self.config.LON_MIN, self.config.LON_MAX)
        invalid_coords: pd.Series = ~mask
        if invalid_coords.any():
            logger.warning(f"Found {invalid_coords.sum()} invalid coordinates")
            df_validated.loc[invalid_coords, ["diner_lat", "diner_lon"]] = np.nan

        # 리뷰 점수 검증
        invalid_scores: pd.Series = ~df["diner_review_avg"].between(
            self.config.REVIEW_SCORE_MIN, self.config.REVIEW_SCORE_MAX
        )
        if invalid_scores.any():
            logger.warning(f"Found {invalid_scores.sum()} invalid review scores")
            df_validated.loc[invalid_scores, "diner_review_avg"] = np.nan

        return df_validated


def process_diner_data(
    df: pd.DataFrame, config: DataValidationConfig = DataValidationConfig()
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    메인 데이터 처리 함수.
    1. 문자열 컬럼의 전처리 및 결측치 처리
    2. 리스트 형태의 컬럼 변환 (diner_tag, diner_menu_name, diner_menu_price)
    3. 수치형 데이터의 타입 변환 및 범위 검증
    4. 최종적으로 검증 결과와 처리된 데이터프레임 반환

    Args:
        df (pd.DataFrame): 처리할 원본 데이터프레임.
        config (DataValidationConfig, optional): 데이터 검증 설정. 기본값은 DataValidationConfig().

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - 처리된 데이터프레임.
            - 검증 결과 사전 (null_counts, data_types, row_count).
    """
    try:
        logger.info("Starting data processing...")
        converter = DataTypeConverter()
        validator = DataFrameValidator(config)

        # 데이터프레임 복사
        df_processed: pd.DataFrame = df.copy()

        # 문자열 컬럼 처리: nan 문자열을 None으로 변경
        str_columns: List[str] = [
            "diner_name",
            "diner_road_address",
            "diner_num_address",
            "diner_phone",
            "diner_open_time",
        ]
        for col in str_columns:
            df_processed[col] = df_processed[col].astype(str).replace("nan", None)

        # 리스트 형태 컬럼 처리
        df_processed["diner_tag"] = df_processed["diner_tag"].apply(
            converter.convert_to_list
        )
        df_processed["diner_menu_name"] = df_processed["diner_menu_name"].apply(
            converter.convert_to_list
        )
        df_processed["diner_menu_price"] = df_processed["diner_menu_price"].apply(
            lambda x: converter.convert_price_list(x, config)
        )

        # 수치형 데이터 처리 및 검증
        numeric_columns: List[str] = [
            "diner_review_avg",
            "diner_blog_review_cnt",
            "diner_lat",
            "diner_lon",
            "bayesian_score",
        ]
        for col in numeric_columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")

        df_processed = validator.validate_numeric_range(df_processed)

        # 정수형 컬럼 처리
        df_processed["diner_review_cnt"] = pd.to_numeric(
            df_processed["diner_review_cnt"], errors="coerce"
        ).astype("Int64")

        # 검증 결과 생성
        validation_results: Dict[str, Any] = {
            "null_counts": df_processed.isnull().sum().to_dict(),
            "data_types": df_processed.dtypes.to_dict(),
            "row_count": len(df_processed),
        }

        logger.info("Data processing completed successfully")
        return df_processed, validation_results

    except Exception as e:
        logger.error(f"Error during data processing: {e}", exc_info=True)
        raise
