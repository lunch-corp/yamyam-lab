import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MiddleCategorySimplifier:
    """
    중분류 카테고리를 간소화하는 전처리기

    원본 중분류를 간소화된 중분류로 변환합니다.
    브랜드/체인점 중심 분류를 음식 종류 중심으로 정리합니다.

    Args:
        config_root_path (str): Config 파일의 root 경로
        data_path (str): Data 파일의 root 경로
    """

    def __init__(
        self,
        config_root_path: Optional[str] = None,
        data_path: Optional[str] = None,
        logger: logging.Logger = logger,
    ):
        self.config_root_path = Path(config_root_path)
        self.data_path = Path(data_path)
        self.logger = logger

        # 간소화 매핑 로드
        self.simplify_mapping = self._load_simplify_mapping()
        # null 채우기 매핑 로드
        self.null_fill_mapping = self._load_null_fill_mapping()

    def _load_simplify_mapping(self) -> Dict[str, Dict[str, list]]:
        """
        간소화 매핑 규칙을 YAML 파일에서 로드합니다.

        Returns:
            Dict[str, Dict[str, list]]: 간소화 매핑 딕셔너리

        Raises:
            FileNotFoundError: YAML 파일이 없거나 simplify_mapping이 없는 경우
        """
        config_path = self.config_root_path / "data" / "category_mappings.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Category mappings file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                mappings = yaml.safe_load(f)

            if "simplify_mapping" not in mappings:
                raise ValueError(
                    f"'simplify_mapping' section not found in {config_path}"
                )

            self.logger.info(f"Loaded simplify_mapping from {config_path}")
            return mappings["simplify_mapping"]

        except Exception as e:
            self.logger.error(
                f"Failed to load simplify_mapping from {config_path}: {e}"
            )
            raise

    def _load_null_fill_mapping(self) -> Dict[str, str]:
        """
        diner_name과 diner_tag 기반 중분류 매핑 규칙을 YAML 파일에서 로드합니다.
        null_fill_mapping이 없으면 simplify_mapping을 기반으로 자동 생성합니다.

        Returns:
            Dict[str, str]: 키워드 -> 중분류 매핑 딕셔너리
        """
        config_path = self.config_root_path / "data" / "category_mappings.yaml"

        if not config_path.exists():
            self.logger.warning(
                f"Category mappings file not found: {config_path}, building from simplify_mapping"
            )
            return self._build_null_fill_mapping_from_simplify()

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                mappings = yaml.safe_load(f)

            if "null_fill_mapping" in mappings:
                self.logger.info(f"Loaded null_fill_mapping from {config_path}")
                return mappings["null_fill_mapping"]
            else:
                self.logger.info(
                    f"'null_fill_mapping' section not found in {config_path}, "
                    f"building from simplify_mapping"
                )
                return self._build_null_fill_mapping_from_simplify()

        except Exception as e:
            self.logger.warning(
                f"Failed to load null_fill_mapping from {config_path}: {e}, "
                f"building from simplify_mapping"
            )
            return self._build_null_fill_mapping_from_simplify()

    def _build_null_fill_mapping_from_simplify(self) -> Dict[str, str]:
        """
        simplify_mapping을 기반으로 null_fill_mapping을 자동 생성합니다.
        각 원본 중분류를 키워드로, 간소화된 중분류를 값으로 매핑합니다.

        같은 키워드가 여러 번 나타나면, 더 긴(구체적인) 키워드를 우선시합니다.

        Returns:
            Dict[str, str]: 키워드 -> 중분류 매핑
        """
        null_fill_mapping = {}
        keyword_lengths = {}  # 키워드의 길이를 추적하여 더 긴 키워드를 우선시

        # simplify_mapping 구조: {대분류: {간소화된_중분류: [원본_중분류_리스트]}}
        for large_category, middle_mappings in self.simplify_mapping.items():
            for simplified_middle, original_list in middle_mappings.items():
                for original in original_list:
                    # 원본 중분류를 키워드로 사용
                    # 쉼표로 구분된 경우 각각을 별도 키워드로 추가
                    if "," in original:
                        keywords = [k.strip() for k in original.split(",")]
                        for keyword in keywords:
                            if keyword:  # 빈 문자열 제외
                                self._add_keyword_to_mapping(
                                    null_fill_mapping,
                                    keyword_lengths,
                                    keyword,
                                    simplified_middle,
                                )
                    else:
                        keyword = original.strip()
                        if keyword:
                            self._add_keyword_to_mapping(
                                null_fill_mapping,
                                keyword_lengths,
                                keyword,
                                simplified_middle,
                            )

        self.logger.info(
            f"Built null_fill_mapping from simplify_mapping with {len(null_fill_mapping)} keywords"
        )
        return null_fill_mapping

    def _add_keyword_to_mapping(
        self,
        null_fill_mapping: Dict[str, str],
        keyword_lengths: Dict[str, int],
        keyword: str,
        simplified_middle: str,
    ) -> None:
        """
        키워드를 매핑에 추가합니다. 이미 존재하는 키워드면 첫 번째 매칭을 유지합니다.

        Args:
            null_fill_mapping: 키워드 -> 중분류 매핑 딕셔너리
            keyword_lengths: 키워드 길이 추적 딕셔너리 (사용하지 않지만 호환성을 위해 유지)
            keyword: 추가할 키워드
            simplified_middle: 간소화된 중분류
        """
        if keyword not in null_fill_mapping:
            # 새로운 키워드면 추가
            null_fill_mapping[keyword] = simplified_middle
            keyword_lengths[keyword] = len(keyword)
        # 이미 존재하는 키워드는 첫 번째 매칭을 유지 (덮어쓰지 않음)

    def fill_null_middle_category(
        self,
        diner_name: str,
        diner_tag: Any,
        large_category: str,
        diner_menu: Any = None,
    ) -> Optional[str]:
        """
        diner_name, diner_tag, diner_menu를 기반으로 중분류를 추론합니다.

        Args:
            diner_name: 식당 이름
            diner_tag: 식당 태그 (리스트 또는 문자열)
            large_category: 대분류
            diner_menu: 식당 메뉴 (리스트 또는 문자열)

        Returns:
            추론된 중분류 (매칭되지 않으면 None)
        """
        if pd.isna(diner_name):
            diner_name = ""
        else:
            diner_name = str(diner_name).lower()

        # diner_tag 처리 (리스트일 수 있음)
        tag_text = ""
        if not pd.isna(diner_tag):
            if isinstance(diner_tag, list):
                tag_text = " ".join(str(tag).lower() for tag in diner_tag)
            else:
                tag_text = str(diner_tag).lower()

        # diner_menu 처리 (리스트 또는 문자열)
        menu_text = ""
        if not pd.isna(diner_menu) and diner_menu is not None:
            if isinstance(diner_menu, str):
                # 문자열 형태의 리스트인 경우 파싱 시도
                try:
                    import ast

                    parsed_menu = ast.literal_eval(diner_menu)
                    if isinstance(parsed_menu, list):
                        menu_text = " ".join(str(menu).lower() for menu in parsed_menu)
                    else:
                        menu_text = str(diner_menu).lower()
                except:
                    menu_text = str(diner_menu).lower()
            elif isinstance(diner_menu, list):
                menu_text = " ".join(str(menu).lower() for menu in diner_menu)
            else:
                menu_text = str(diner_menu).lower()

        # 검색할 텍스트 합치기
        search_text = f"{diner_name} {tag_text} {menu_text}".strip()

        # 키워드 매칭 (긴 키워드부터 우선 매칭)
        sorted_keywords = sorted(self.null_fill_mapping.keys(), key=len, reverse=True)

        for keyword in sorted_keywords:
            if keyword.lower() in search_text:
                inferred_middle = self.null_fill_mapping[keyword]
                # 대분류와 호환성 확인 (간단한 체크)
                # 대분류가 있고 simplify_mapping에도 있으면 호환성 확인
                if (
                    pd.notna(large_category)
                    and large_category != ""
                    and large_category in self.simplify_mapping
                ):
                    large_mapping = self.simplify_mapping[large_category]
                    # 추론된 중분류가 해당 대분류의 간소화된 중분류 목록에 있는지 확인
                    if inferred_middle in large_mapping:
                        return inferred_middle
                    # 매핑에 없어도 일단 반환 (나중에 처리될 수 있음)
                # 대분류가 없거나 매핑에 없어도 반환
                return inferred_middle

        return None

    def simplify_middle_category(
        self, large_category: str, middle_category: str
    ) -> str:
        """
        원본 중분류를 간소화된 중분류로 변환합니다.

        Args:
            large_category: 대분류
            middle_category: 원본 중분류

        Returns:
            간소화된 중분류 (매핑되지 않으면 원본 반환)
        """
        if pd.isna(middle_category) or middle_category == "":
            return middle_category

        # 대분류가 "패밀리레스토랑" 또는 "샐러드"인 경우 "양식" 섹션의 매핑을 사용
        if large_category in ["패밀리레스토랑", "샐러드"]:
            large_category = "양식"

        # 먼저 현재 대분류의 매핑에서 찾기
        large_mapping = self.simplify_mapping.get(large_category, {})

        # 정확히 일치하는 경우
        for simplified, originals in large_mapping.items():
            if middle_category == originals:
                return simplified

        # 부분 일치 확인 (키워드 포함)
        for simplified, originals in large_mapping.items():
            for original in originals:
                if (
                    original.lower() in str(middle_category).lower()
                    or str(middle_category).lower() in original.lower()
                ):
                    return simplified

        # 현재 대분류에서 찾지 못한 경우, 모든 대분류의 매핑에서 찾기
        # (원본 데이터의 대분류가 잘못된 경우 대비)
        for cat, cat_mapping in self.simplify_mapping.items():
            if cat == large_category:
                continue  # 이미 확인했음
            for simplified, originals in cat_mapping.items():
                for original in originals:
                    if (
                        original.lower() in str(middle_category).lower()
                        or str(middle_category).lower() in original.lower()
                    ):
                        return simplified

        # 매핑되지 않으면 원본 반환
        return middle_category

    def process(
        self,
        category_df: Optional[pd.DataFrame] = None,
        inplace: bool = True,
    ) -> pd.DataFrame:
        """
        카테고리 데이터프레임의 중분류를 간소화합니다.
        원본 diner_category_middle 컬럼을 간소화된 값으로 직접 변경합니다.

        Args:
            category_df: 카테고리 데이터프레임 (None이면 파일에서 로드)
            inplace: 원본 컬럼을 직접 수정할지 여부 (기본값: True)

        Returns:
            간소화된 중분류가 적용된 데이터프레임
        """
        if category_df is None:
            category_df = pd.read_csv(self.data_path / "diner_category_raw.csv")
        else:
            category_df = category_df.copy() if not inplace else category_df

        # diner_name, diner_tag, diner_menu_name이 없으면 diner.csv에서 로드하여 merge
        has_diner_name = "diner_name" in category_df.columns
        has_diner_tag = "diner_tag" in category_df.columns
        has_diner_menu = "diner_menu_name" in category_df.columns

        if not has_diner_name or not has_diner_tag or not has_diner_menu:
            diner_file = self.data_path / "diner.csv"
            if diner_file.exists():
                missing_cols = []
                if not has_diner_name:
                    missing_cols.append("diner_name")
                if not has_diner_tag:
                    missing_cols.append("diner_tag")
                if not has_diner_menu:
                    missing_cols.append("diner_menu_name")
                self.logger.info(
                    f"Missing columns {missing_cols}, loading from {diner_file}"
                )
                diner_df = pd.read_csv(diner_file, low_memory=False)

                # 필요한 컬럼만 선택 (없는 것만 merge)
                merge_columns = ["diner_idx"]
                if not has_diner_name and "diner_name" in diner_df.columns:
                    merge_columns.append("diner_name")
                if not has_diner_tag and "diner_tag" in diner_df.columns:
                    merge_columns.append("diner_tag")
                if not has_diner_menu and "diner_menu_name" in diner_df.columns:
                    merge_columns.append("diner_menu_name")

                # merge_columns에 diner_idx 외의 컬럼이 있으면 merge 수행
                if len(merge_columns) > 1:
                    category_df = pd.merge(
                        category_df,
                        diner_df[merge_columns],
                        on="diner_idx",
                        how="left",
                    )
                    self.logger.info(
                        f"Merged diner data. Added columns: {[c for c in merge_columns if c != 'diner_idx']}"
                    )
                else:
                    self.logger.info(
                        "All required columns already exist, skipping merge"
                    )
            else:
                self.logger.warning(
                    f"diner.csv not found at {diner_file}, "
                    f"null filling will be skipped if diner_name/diner_tag/diner_menu_name columns are missing"
                )

        original_values = category_df["diner_category_middle"].copy()

        self.logger.info(f"Original category_df shape: {category_df.shape}")
        self.logger.info(
            f"Null middle categories: {category_df['diner_category_middle'].isna().sum()}"
        )

        # 중분류가 null인 경우 diner_name과 diner_tag를 기반으로 채우기
        null_middle_mask = category_df["diner_category_middle"].isna()
        if null_middle_mask.any():
            null_count_before = null_middle_mask.sum()

            # diner_name, diner_tag, diner_menu_name 컬럼이 있는지 확인
            has_diner_name = "diner_name" in category_df.columns
            has_diner_tag = "diner_tag" in category_df.columns
            has_diner_menu = "diner_menu_name" in category_df.columns

            if has_diner_name or has_diner_tag or has_diner_menu:
                category_df.loc[null_middle_mask, "diner_category_middle"] = (
                    category_df.loc[null_middle_mask].apply(
                        lambda row: self.fill_null_middle_category(
                            row.get("diner_name", ""),
                            row.get("diner_tag", ""),
                            row.get("diner_category_large", ""),
                            row.get("diner_menu_name", None),
                        ),
                        axis=1,
                    )
                )

                null_count_after = category_df["diner_category_middle"].isna().sum()
                filled_count = null_count_before - null_count_after

                self.logger.info(
                    f"Filled {filled_count} null middle categories using diner_name/diner_tag/diner_menu_name "
                    f"({null_count_after} remaining null)"
                )
            else:
                self.logger.warning(
                    "diner_name or diner_tag columns not found, skipping null fill"
                )

        # 대분류가 "간식"이고 중분류가 "닭강정"인 경우 "치킨" > "닭강정"으로 변경 (간소화 전에 처리)
        chicken_gangjeong_mask = (category_df["diner_category_large"] == "간식") & (
            category_df["diner_category_middle"] == "닭강정"
        )
        if chicken_gangjeong_mask.any():
            category_df.loc[chicken_gangjeong_mask, "diner_category_large"] = "치킨"
            self.logger.info(
                f"Changed large category from '간식' to '치킨' for {chicken_gangjeong_mask.sum()} rows "
                f"with middle category '닭강정' (before simplification)"
            )

        # 간소화 적용 - 원본 컬럼을 직접 변경
        category_df["diner_category_middle"] = category_df.apply(
            lambda row: self.simplify_middle_category(
                row["diner_category_large"], row["diner_category_middle"]
            ),
            axis=1,
        )

        # 대분류가 "치킨"이고 간소화 후 중분류가 "디저트"가 된 경우 (원본이 "닭강정"이었을 수 있음)
        # 중분류를 "닭강정"으로 복원
        chicken_dessert_mask = (category_df["diner_category_large"] == "치킨") & (
            category_df["diner_category_middle"] == "디저트"
        )
        # 원본이 "닭강정"이었던 경우만 처리
        original_chicken_gangjeong = original_values == "닭강정"
        chicken_gangjeong_restore_mask = (
            chicken_dessert_mask & original_chicken_gangjeong
        )
        if chicken_gangjeong_restore_mask.any():
            category_df.loc[chicken_gangjeong_restore_mask, "diner_category_middle"] = (
                "닭강정"
            )
            self.logger.info(
                f"Restored middle category to '닭강정' for {chicken_gangjeong_restore_mask.sum()} rows "
                f"with large category '치킨' (originally was '닭강정')"
            )

        # 샤브샤브, 칼국수인 경우 대분류를 한식으로 변경
        shabu_shabu_mask = category_df["diner_category_middle"].isin(
            ["샤브샤브", "칼국수"]
        )
        if shabu_shabu_mask.any():
            category_df.loc[shabu_shabu_mask, "diner_category_large"] = "한식"
            self.logger.info(
                f"Changed large category to '한식' for {shabu_shabu_mask.sum()} rows "
                f"with middle category '샤브샤브' or '칼국수'"
            )

        # 대분류가 "샐러드"인 경우 "양식"으로 변경
        salad_mask = category_df["diner_category_large"] == "샐러드"
        if salad_mask.any():
            category_df.loc[salad_mask, "diner_category_large"] = "양식"
            self.logger.info(
                f"Changed large category from '샐러드' to '양식' for {salad_mask.sum()} rows"
            )

        # 패밀리레스토랑, 스테이크하우스, 이탈리안인 경우 대분류를 양식으로 변경
        family_restaurant_mask = category_df["diner_category_middle"].isin(
            ["패밀리레스토랑", "스테이크하우스", "이탈리안"]
        )
        if family_restaurant_mask.any():
            original_large_values = category_df.loc[
                family_restaurant_mask, "diner_category_large"
            ].copy()
            category_df.loc[family_restaurant_mask, "diner_category_large"] = "양식"
            changed_large_count = (
                original_large_values
                != category_df.loc[family_restaurant_mask, "diner_category_large"]
            ).sum()
            self.logger.info(
                f"Changed large category to '양식' for {changed_large_count} rows "
                f"with middle category '패밀리레스토랑', '스테이크하우스', or '이탈리안'"
            )

        # null 값 채우기 로직
        self._fill_null_categories(category_df)

        # null 채우기에 사용된 diner_name, diner_tag, diner_menu_name 컬럼 제거
        columns_to_drop = []
        if "diner_name" in category_df.columns:
            columns_to_drop.append("diner_name")
        if "diner_tag" in category_df.columns:
            columns_to_drop.append("diner_tag")
        if "diner_menu_name" in category_df.columns:
            columns_to_drop.append("diner_menu_name")

        if columns_to_drop:
            category_df = category_df.drop(columns=columns_to_drop)
            self.logger.info(
                f"Dropped columns used for null filling: {columns_to_drop}"
            )

        return category_df

    def _fill_null_categories(self, category_df: pd.DataFrame) -> None:
        """
        중분류 카테고리의 null 값을 채웁니다.
        diner_name과 diner_tag를 기반으로 중분류를 추론합니다.

        Args:
            category_df: 처리할 카테고리 데이터프레임
        """
        # 중분류 컬럼이 있는지 확인
        has_middle = "diner_category_middle" in category_df.columns

        # 중분류 null 채우기 (diner_name/diner_tag 기반)
        if not has_middle:
            self.logger.warning(
                "diner_category_middle column not found, skipping null fill"
            )
        else:
            middle_null = category_df["diner_category_middle"].isna()
            if middle_null.sum() > 0:
                # diner_name, diner_tag, diner_menu_name 컬럼이 있는지 확인
                has_diner_name = "diner_name" in category_df.columns
                has_diner_tag = "diner_tag" in category_df.columns
                has_diner_menu = "diner_menu_name" in category_df.columns

                if has_diner_name or has_diner_tag or has_diner_menu:
                    null_count_before = middle_null.sum()

                    category_df.loc[middle_null, "diner_category_middle"] = (
                        category_df.loc[middle_null].apply(
                            lambda row: self.fill_null_middle_category(
                                row.get("diner_name", ""),
                                row.get("diner_tag", ""),
                                row.get("diner_category_large", ""),
                                row.get("diner_menu_name", None),
                            ),
                            axis=1,
                        )
                    )

                    null_count_after = category_df["diner_category_middle"].isna().sum()
                    filled_count = null_count_before - null_count_after

                    self.logger.info(
                        f"Filled {filled_count} null middle categories using diner_name/diner_tag/diner_menu_name "
                        f"in _fill_null_categories ({null_count_after} remaining null)"
                    )
                else:
                    self.logger.warning(
                        "diner_name/diner_tag/diner_menu_name columns not found in _fill_null_categories, "
                        "skipping null fill"
                    )


class CategoryProcessor:
    """
    카테고리 처리 클래스로, YAML 설정 파일에 따라 데이터프레임의 카테고리 정보를 변환합니다.

    Attributes:
        df (pd.DataFrame): 처리할 데이터프레임 복사본.
        config_root_path (str): Root path for config file.

    Examples:
        >>> processor = CategoryProcessor(df)
        >>> processor.process_all()
        >>> processed_df = processor.category_preprocessed_diners
    """

    def __init__(self, df: pd.DataFrame, config_root_path: str) -> None:
        """
        카테고리 프로세서를 초기화합니다.

        Args:
            df (pd.DataFrame): 전처리할 데이터프레임.
        """
        self.df: pd.DataFrame = df.copy()
        self.config_root_path = Path(config_root_path)
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
        config_path: Path = self.config_root_path / "data" / "category_mappings.yaml"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                mappings = yaml.safe_load(f)
            return mappings
        except Exception as e:
            logger.error(f"Failed to load category mappings from {config_path}: {e}")
            raise

    @property
    def category_preprocessed_diners(self) -> pd.DataFrame:
        """
        카테고리 처리가 완료된 데이터프레임을 반환합니다.

        Returns:
            pd.DataFrame: 카테고리 처리가 완료된 데이터프레임.
        """
        return self.df

    def process_all(self) -> None:
        """
        모든 카테고리 처리 함수를 순차적으로 실행합니다.

        Returns:
            CategoryProcessor: 연쇄 호출이 가능한 self.
        """
        self.process_chicken_categories()
        self.process_lowering_categories(level="large")
        self.process_lowering_categories(level="middle")
        self.process_partly_lowering_categories()
        self.integrate_diner_category_middle()
        self.rename_diner_category_middle()

    def process_lowering_categories(self, level: str = "large") -> None:
        """
        대/중분류 카테고리 조정을 처리합니다.

        level 매개변수에 따라 다음과 같이 동작합니다.
        - "large": YAML의 ``lowering_large_categories`` 매핑을 사용하여 대분류를 하향 조정
        - "middle": YAML의 ``lowering_middle_categories`` 매핑을 사용하여 중분류를 하향 조정

        after 카테고리로 변경한 뒤, 기존 하위 카테고리는 한 단계씩 아래로 이동합니다.

        Args:
            level (str, optional): 처리할 카테고리 깊이 ("large" 또는 "middle"). 기본값 "large".

        Returns:
            CategoryProcessor: 연쇄 호출이 가능한 self.
        """

        if level not in {"large", "middle"}:
            raise ValueError("level must be either 'large' or 'middle'")

        # level별 설정값 정의
        mapping_key = (
            "lowering_large_categories"
            if level == "large"
            else "lowering_middle_categories"
        )
        column_name = f"diner_category_{level}"

        # 매핑이 없을 경우 그대로 반환
        mappings: Dict[str, List[str]] = self.mappings.get(mapping_key, {})
        if not mappings:
            return self

        for after_category, before_categories in mappings.items():
            target_rows: pd.Series = self.df[column_name].isin(before_categories)

            if not target_rows.any():
                continue

            # level에 따라 이동 방식 결정
            self._shift_categories_down(target_rows, target_category=column_name)

            # 카테고리 업데이트
            self.df.loc[target_rows, column_name] = after_category

    def process_partly_lowering_categories(self) -> None:
        """
        부분적 카테고리 조정을 처리합니다.
        mappings의 partly_lowering_large_categories 설정에 따라,
        특정 조건을 만족하는 경우 대분류와 중분류를 변경합니다.

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

    def process_chicken_categories(
        self, target_categories: List[str] = ["치킨"]
    ) -> None:
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

    def integrate_diner_category_middle(self) -> None:
        """
        diner_category_large별로 diner_category_middle을 통합합니다.
        """
        integration_config = self.mappings["integrate_diner_category_middle"]
        for diner_category_large, config in integration_config.items():
            for new_middle, old_middles in config.items():
                target_rows = self.df["diner_category_large"] == diner_category_large
                target_rows &= self.df["diner_category_middle"].isin(old_middles)
                self.df.loc[target_rows, "diner_category_middle"] = new_middle

    def rename_diner_category_middle(self) -> None:
        """
        diner_category_middle의 이름을 재정의합니다.
        """
        rename_config = self.mappings["rename_diner_category_middle"]

        flat_mapping = {
            (large, middle): new_middle
            for large, middles in rename_config.items()
            for middle, new_middle in middles.items()
        }

        # 튜플 (diner_category_large, diner_category_middle)을 기반으로 매핑
        self.df["diner_category_middle"] = self.df.apply(
            lambda row: flat_mapping.get(
                (row["diner_category_large"], row["diner_category_middle"]),
                row["diner_category_middle"],
            ),
            axis=1,
        )

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

        for to_category, from_category in category_depth:
            self.df.loc[target_rows, f"diner_category_{to_category}"] = self.df.loc[
                target_rows, f"diner_category_{from_category}"
            ]
