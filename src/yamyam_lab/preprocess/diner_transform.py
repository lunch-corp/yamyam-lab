import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# transformers는 테스트 환경에서는 import하지 않음
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    # pytest 실행 중이거나 CI 환경에서는 transformers를 import하지 않음
    AutoModelForCausalLM = None
    AutoTokenizer = None

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

        # 양식 대분류인 경우, 치킨 섹션도 확인 (치킨이 lowering_large_categories에서 양식으로 변경되었을 수 있음)
        chicken_mapping = None
        if large_category == "양식":
            chicken_mapping = self.simplify_mapping.get("치킨", {})

        # 먼저 현재 대분류의 매핑에서 찾기
        large_mapping = self.simplify_mapping.get(large_category, {})

        # 정확히 일치하는 경우 (현재 대분류)
        for simplified, originals in large_mapping.items():
            if middle_category in originals:
                return simplified

        # 양식 대분류인 경우, 치킨 섹션에서도 정확히 일치하는지 확인
        if chicken_mapping:
            for simplified, originals in chicken_mapping.items():
                if middle_category in originals:
                    return simplified

        # 부분 일치 확인 (키워드 포함) - 현재 대분류
        for simplified, originals in large_mapping.items():
            for original in originals:
                if (
                    original.lower() in str(middle_category).lower()
                    or str(middle_category).lower() in original.lower()
                ):
                    return simplified

        # 양식 대분류인 경우, 치킨 섹션에서도 부분 일치 확인
        if chicken_mapping:
            for simplified, originals in chicken_mapping.items():
                for original in originals:
                    if (
                        original.lower() in str(middle_category).lower()
                        or str(middle_category).lower() in original.lower()
                    ):
                        return simplified

        # YAML이 기준이므로, 현재 대분류의 매핑에서만 찾고
        # 매핑되지 않으면 원본 반환 (다른 대분류의 매핑을 확인하지 않음)
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

        self.logger.info(f"Original category_df shape: {category_df.shape}")
        self.logger.info(
            f"Null middle categories: {category_df['diner_category_middle'].isna().sum()}"
        )

        # 간식 대분류의 "닭강정"을 치킨 대분류로 변경
        chicken_gangjeong_mask = (category_df["diner_category_large"] == "간식") & (
            category_df["diner_category_middle"] == "닭강정"
        )
        if chicken_gangjeong_mask.any():
            category_df.loc[chicken_gangjeong_mask, "diner_category_large"] = "치킨"
            self.logger.info(
                f"Changed large category from '간식' to '치킨' for {chicken_gangjeong_mask.sum()} rows "
                f"with middle category '닭강정' (before simplification)"
            )

        # 한식 대분류의 "분식" 중분류를 "분식" 대분류로 이동 (간소화 전에 처리)
        # 이때 소분류나 상세분류를 확인해서 원본 중분류를 더 구체적인 값으로 변경
        bunsik_before_mask = (category_df["diner_category_large"] == "한식") & (
            category_df["diner_category_middle"] == "분식"
        )
        if bunsik_before_mask.any():
            category_df.loc[bunsik_before_mask, "diner_category_large"] = "분식"
            # 소분류나 상세분류를 확인해서 원본 중분류를 더 구체적인 값으로 변경
            bunsik_indices = category_df[bunsik_before_mask].index
            for idx in bunsik_indices:
                small_category = category_df.loc[idx, "diner_category_small"]
                detail_category = category_df.loc[idx, "diner_category_detail"]

                # 소분류나 상세분류가 있으면 그것을 중분류로 사용
                if pd.notna(small_category) and small_category != "":
                    category_df.loc[idx, "diner_category_middle"] = small_category
                elif pd.notna(detail_category) and detail_category != "":
                    category_df.loc[idx, "diner_category_middle"] = detail_category
                # 둘 다 없으면 "분식" 그대로 유지 (나중에 간소화에서 처리)

            self.logger.info(
                f"Changed large category from '한식' to '분식' for {bunsik_before_mask.sum()} rows "
                f"and updated middle category using small/detail categories (before simplification)"
            )

        # 간소화 적용 - 원본 컬럼을 직접 변경
        category_df["diner_category_middle"] = category_df.apply(
            lambda row: self.simplify_middle_category(
                row["diner_category_large"], row["diner_category_middle"]
            ),
            axis=1,
        )

        # 분식 대분류의 중분류가 여전히 "분식"인 경우, 소분류/상세분류를 확인해서 매핑 재적용
        bunsik_still_bunsik_mask = (category_df["diner_category_large"] == "분식") & (
            category_df["diner_category_middle"] == "분식"
        )
        if bunsik_still_bunsik_mask.any():
            bunsik_indices = category_df[bunsik_still_bunsik_mask].index
            for idx in bunsik_indices:
                # 소분류나 상세분류를 확인하여 분식 섹션의 매핑 적용
                small_category = category_df.loc[idx, "diner_category_small"]
                detail_category = category_df.loc[idx, "diner_category_detail"]

                # 소분류나 상세분류를 중분류로 사용하여 매핑 시도
                target_middle = None
                if pd.notna(small_category) and small_category != "":
                    target_middle = self.simplify_middle_category(
                        "분식", small_category
                    )
                    # 매핑이 적용되었으면 사용
                    if target_middle != small_category:
                        category_df.loc[idx, "diner_category_middle"] = target_middle
                        continue

                if pd.notna(detail_category) and detail_category != "":
                    target_middle = self.simplify_middle_category(
                        "분식", detail_category
                    )
                    # 매핑이 적용되었으면 사용
                    if target_middle != detail_category:
                        category_df.loc[idx, "diner_category_middle"] = target_middle
                        continue

                # 소분류나 상세분류도 매핑되지 않으면 그대로 유지 (나중에 KNN으로 처리)

            self.logger.info(
                f"Reapplied simplify_mapping for {bunsik_still_bunsik_mask.sum()} rows "
                f"with '분식' middle category using small/detail categories"
            )

        # 간식 대분류의 "닭강정"이 간소화 후에도 남아있는 경우 치킨 대분류로 변경하고 중분류를 "닭강정"으로 유지
        chicken_gangjeong_after_mask = (
            category_df["diner_category_large"] == "간식"
        ) & (category_df["diner_category_middle"] == "닭강정")
        if chicken_gangjeong_after_mask.any():
            category_df.loc[chicken_gangjeong_after_mask, "diner_category_large"] = (
                "치킨"
            )
            self.logger.info(
                f"Changed large category from '간식' to '치킨' for {chicken_gangjeong_after_mask.sum()} rows "
                f"with middle category '닭강정' (after simplification)"
            )

        # 한식 대분류의 "일반도시락" 중분류를 "도시락" 대분류로 이동
        dosirak_mask = (category_df["diner_category_large"] == "한식") & (
            category_df["diner_category_middle"] == "일반도시락"
        )
        if dosirak_mask.any():
            category_df.loc[dosirak_mask, "diner_category_large"] = "도시락"
            self.logger.info(
                f"Changed large category from '한식' to '도시락' for {dosirak_mask.sum()} rows "
                f"with middle category '일반도시락'"
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

        # 일식 대분류의 "전통일식집"과 "기타" 중분류를 "기타" 대분류로 이동
        japanese_traditional_mask = (category_df["diner_category_large"] == "일식") & (
            category_df["diner_category_middle"] == "전통일식집"
        )
        if japanese_traditional_mask.any():
            category_df.loc[japanese_traditional_mask, "diner_category_large"] = "기타"
            self.logger.info(
                f"Changed large category from '일식' to '기타' for {japanese_traditional_mask.sum()} rows "
                f"with middle category '전통일식집'"
            )

        japanese_etc_mask = (category_df["diner_category_large"] == "일식") & (
            category_df["diner_category_middle"] == "기타"
        )
        if japanese_etc_mask.any():
            category_df.loc[japanese_etc_mask, "diner_category_large"] = "기타"
            self.logger.info(
                f"Changed large category from '일식' to '기타' for {japanese_etc_mask.sum()} rows "
                f"with middle category '기타'"
            )

        return category_df


class MiddleCategoryKNNImputer:
    """
    KNN 기반 중분류 null 값 imputation 클래스

    각 대분류별로 별도의 KNN 모델을 학습하여 중분류를 예측합니다.
    대분류, 소분류, 상세분류 등의 정보를 활용하여 중분류를 예측합니다.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        data_path: Optional[str] = None,
        logger: logging.Logger = logger,
    ):
        """
        Args:
            n_neighbors: KNN에서 사용할 이웃 수 (기본값: 5)
            data_path: diner.csv 파일 경로 (diner_name, diner_tag 등 사용 시)
            logger: 로거 인스턴스
        """
        self.n_neighbors = n_neighbors
        self.data_path = Path(data_path) if data_path else None
        self.logger = logger
        # 대분류별 모델 저장
        self.knn_models: Dict[str, KNeighborsClassifier] = {}
        self.label_encoders: Dict[
            str, Dict[str, LabelEncoder]
        ] = {}  # {large_category: {feature_col: encoder}}
        self.feature_columns: Dict[
            str, List[str]
        ] = {}  # {large_category: [feature_cols]}
        self.target_encoders: Dict[str, LabelEncoder] = {}  # {large_category: encoder}
        # TF-IDF vectorizers 저장 (대분류별)
        self.tfidf_vectorizers: Dict[
            str, Dict[str, TfidfVectorizer]
        ] = {}  # {large_category: {name/tag: vectorizer}}

    def _prepare_features(
        self,
        df: pd.DataFrame,
        use_diner_info: bool = False,
        large_category: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        KNN을 위한 feature 준비

        Args:
            df: 카테고리 데이터프레임
            use_diner_info: diner.csv의 정보(diner_name, diner_tag 등) 사용 여부
            large_category: 대분류 (대분류별로 이미 필터링된 데이터의 경우)

        Returns:
            feature 데이터프레임
        """
        features = df.copy()

        # 기본 카테고리 정보 사용 (대분류는 이미 필터링되어 있으므로 제외)
        feature_cols = []
        if "diner_category_small" in df.columns:
            feature_cols.append("diner_category_small")
        if "diner_category_detail" in df.columns:
            feature_cols.append("diner_category_detail")

        # diner.csv 정보 병합 (선택적)
        if use_diner_info and self.data_path:
            diner_path = self.data_path / "diner.csv"
            if diner_path.exists():
                try:
                    diner_df = pd.read_csv(diner_path)
                    # diner_idx로 병합
                    if "diner_idx" in df.columns and "diner_idx" in diner_df.columns:
                        features = features.merge(
                            diner_df[["diner_idx", "diner_name", "diner_tag"]],
                            on="diner_idx",
                            how="left",
                        )
                        # diner_name, diner_tag에 대해 TF-IDF 처리를 하여 feature로 추가
                        if large_category not in self.tfidf_vectorizers:
                            self.tfidf_vectorizers[large_category] = {}

                    if "diner_name" in features.columns:
                        name_values = features["diner_name"].fillna("").astype(str)

                        if large_category and "name" in self.tfidf_vectorizers.get(
                            large_category, {}
                        ):
                            # 이미 학습된 vectorizer 사용
                            tfidf_name = self.tfidf_vectorizers[large_category]["name"]
                            name_tfidf = tfidf_name.transform(name_values)
                        else:
                            # 새로운 vectorizer 학습
                            tfidf_name = TfidfVectorizer(max_features=100)
                            name_tfidf = tfidf_name.fit_transform(name_values)
                            if large_category:
                                if large_category not in self.tfidf_vectorizers:
                                    self.tfidf_vectorizers[large_category] = {}
                                self.tfidf_vectorizers[large_category]["name"] = (
                                    tfidf_name
                                )

                        name_tfidf_df = pd.DataFrame(
                            name_tfidf.toarray(),
                            columns=[
                                f"diner_name_tfidf_{i}"
                                for i in range(name_tfidf.shape[1])
                            ],
                            index=features.index,
                        )
                        features = pd.concat([features, name_tfidf_df], axis=1)
                        feature_cols.extend(name_tfidf_df.columns.tolist())

                    if "diner_tag" in features.columns:
                        # diner_tag는 리스트 형태일 수 있으므로 문자열로 변환
                        tag_values = features["diner_tag"].apply(
                            lambda x: " ".join(x)
                            if isinstance(x, list)
                            else str(x)
                            if pd.notna(x)
                            else ""
                        )

                        if large_category and "tag" in self.tfidf_vectorizers.get(
                            large_category, {}
                        ):
                            # 이미 학습된 vectorizer 사용
                            tfidf_tag = self.tfidf_vectorizers[large_category]["tag"]
                            tag_tfidf = tfidf_tag.transform(tag_values)
                        else:
                            # 새로운 vectorizer 학습
                            tfidf_tag = TfidfVectorizer(max_features=10)
                            tag_tfidf = tfidf_tag.fit_transform(tag_values)
                            if large_category:
                                if large_category not in self.tfidf_vectorizers:
                                    self.tfidf_vectorizers[large_category] = {}
                                self.tfidf_vectorizers[large_category]["tag"] = (
                                    tfidf_tag
                                )

                        tag_tfidf_df = pd.DataFrame(
                            tag_tfidf.toarray(),
                            columns=[
                                f"diner_tag_tfidf_{i}"
                                for i in range(tag_tfidf.shape[1])
                            ],
                            index=features.index,
                        )
                        features = pd.concat([features, tag_tfidf_df], axis=1)
                        feature_cols.extend(tag_tfidf_df.columns.tolist())
                except Exception as e:
                    self.logger.warning(f"Failed to load diner.csv: {e}")

        # feature 컬럼 선택
        available_cols = [col for col in feature_cols if col in features.columns]

        # 대분류별로 feature 컬럼 저장
        if large_category:
            self.feature_columns[large_category] = available_cols

        return (
            features[available_cols] if available_cols else pd.DataFrame(index=df.index)
        )

    def _encode_features(
        self, df: pd.DataFrame, large_category: str, is_training: bool = False
    ) -> np.ndarray:
        """
        범주형 feature를 수치형으로 인코딩

        Args:
            df: feature 데이터프레임
            large_category: 대분류
            is_training: 학습 단계인지 여부 (fit 시 True)

        Returns:
            인코딩된 feature 배열
        """
        # feature_columns가 없으면 자동으로 초기화
        if large_category not in self.feature_columns:
            self.feature_columns[large_category] = (
                list(df.columns) if len(df.columns) > 0 else []
            )

        if large_category not in self.label_encoders:
            self.label_encoders[large_category] = {}

        feature_cols = self.feature_columns[large_category]

        if len(feature_cols) == 0:
            # feature가 없는 경우 더미 feature 생성
            return np.zeros((len(df), 1))

        encoded_features = []

        for col in feature_cols:
            # TF-IDF feature는 이미 수치형이므로 그대로 사용
            if col.startswith("diner_name_tfidf_") or col.startswith(
                "diner_tag_tfidf_"
            ):
                # 수치형 feature는 그대로 사용 (null은 0으로 처리)
                encoded_col = df[col].fillna(0).values.astype(float)
                encoded_features.append(encoded_col)
            else:
                # 범주형 feature는 LabelEncoder로 인코딩
                if col not in self.label_encoders[large_category]:
                    self.label_encoders[large_category][col] = LabelEncoder()
                    # null 값을 별도로 처리
                    non_null_mask = df[col].notna()
                    if non_null_mask.any():
                        self.label_encoders[large_category][col].fit(
                            df.loc[non_null_mask, col]
                        )
                    else:
                        # 모든 값이 null인 경우 더미 인코더 생성
                        self.label_encoders[large_category][col] = LabelEncoder()
                        self.label_encoders[large_category][col].fit([""])

                # 인코딩 (null은 -1로 처리)
                encoded_col = np.full(len(df), -1, dtype=float)
                non_null_mask = df[col].notna()
                if non_null_mask.any():
                    values_to_encode = df.loc[non_null_mask, col]
                    # 학습 시 보지 못한 값 처리
                    known_values_mask = values_to_encode.isin(
                        self.label_encoders[large_category][col].classes_
                    )

                    if known_values_mask.any():
                        # 학습된 값들만 인코딩
                        non_null_positions = np.where(non_null_mask)[0]
                        known_positions = non_null_positions[known_values_mask]
                        encoded_col[known_positions] = self.label_encoders[
                            large_category
                        ][col].transform(values_to_encode[known_values_mask])

                    # 학습 시 보지 못한 값이 있으면 경고 (이론적으로는 발생하지 않아야 함)
                    if not known_values_mask.all():
                        unknown_values = values_to_encode[~known_values_mask].unique()
                        self.logger.warning(
                            f"Column '{col}' (large_category='{large_category}') contains "
                            f"previously unseen values: {unknown_values.tolist()[:5]}. "
                            f"These will be encoded as -1 (missing value)."
                        )
                encoded_features.append(encoded_col)

        return (
            np.column_stack(encoded_features)
            if encoded_features
            else np.zeros((len(df), 1))
        )

    def fit(
        self,
        category_df: pd.DataFrame,
        use_diner_info: bool = False,
        test_size: float = 0.2,
    ) -> Dict[str, Dict[str, float]]:
        """
        각 대분류별로 KNN 모델 학습 및 평가

        Args:
            category_df: 카테고리 데이터프레임
            use_diner_info: diner.csv 정보 사용 여부
            test_size: 평가용 데이터 비율

        Returns:
            대분류별 평가 메트릭 딕셔너리 {large_category: {metric: value}}
        """
        # 중분류가 null이 아닌 데이터만 사용
        train_df = category_df[category_df["diner_category_middle"].notna()].copy()

        if len(train_df) == 0:
            raise ValueError("No non-null middle category data for training")

        # 대분류별로 데이터 분리
        all_metrics = {}
        large_categories = train_df["diner_category_large"].unique()

        self.logger.info("=" * 60)
        self.logger.info("Training KNN Models by Large Category")
        self.logger.info("=" * 60)
        self.logger.info(f"Total large categories: {len(large_categories)}")
        self.logger.info("")

        for large_cat in sorted(large_categories):
            if pd.isna(large_cat):
                continue

            # 해당 대분류의 데이터만 필터링
            large_cat_df = train_df[
                train_df["diner_category_large"] == large_cat
            ].copy()

            if len(large_cat_df) < 2:
                self.logger.warning(
                    f"Skipping '{large_cat}': insufficient data ({len(large_cat_df)} samples)"
                )
                continue

            # Feature 준비
            features_df = self._prepare_features(
                large_cat_df, use_diner_info, large_category=large_cat
            )
            X = self._encode_features(features_df, large_cat, is_training=True)

            # Target (중분류)
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(large_cat_df["diner_category_middle"])
            self.target_encoders[large_cat] = target_encoder

            # 중분류가 1개만 있으면 학습 불가
            unique_middles = large_cat_df["diner_category_middle"].nunique()
            if unique_middles < 2:
                self.logger.warning(
                    f"Skipping '{large_cat}': only {unique_middles} unique middle category"
                )
                continue

            # Train/Test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            if len(X_train) == 0:
                self.logger.warning(
                    f"Skipping '{large_cat}': no training data after split"
                )
                continue

            # KNN 모델 학습
            knn_model = KNeighborsClassifier(
                n_neighbors=min(self.n_neighbors, len(X_train)), weights="distance"
            )
            knn_model.fit(X_train, y_train)
            self.knn_models[large_cat] = knn_model

            # 평가
            if len(X_test) > 0:
                y_pred = knn_model.predict(X_test)
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision_macro": precision_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                    "recall_macro": recall_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                    "f1_macro": f1_score(
                        y_test, y_pred, average="macro", zero_division=0
                    ),
                    "precision_weighted": precision_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    ),
                    "recall_weighted": recall_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    ),
                    "f1_weighted": f1_score(
                        y_test, y_pred, average="weighted", zero_division=0
                    ),
                }
                all_metrics[large_cat] = metrics

                self.logger.info("=" * 60)
                self.logger.info(f"Large Category: {large_cat}")
                self.logger.info("=" * 60)
                self.logger.info(f"Training samples: {len(X_train)}")
                self.logger.info(f"Test samples: {len(X_test)}")
                self.logger.info(
                    f"Features used: {self.feature_columns.get(large_cat, [])}"
                )
                self.logger.info(
                    f"Number of neighbors: {min(self.n_neighbors, len(X_train))}"
                )
                unique_middle_list = sorted(
                    large_cat_df["diner_category_middle"].unique()
                )
                self.logger.info(f"Unique middle categories: {unique_middles}")
                self.logger.info(f"  Categories: {', '.join(unique_middle_list)}")
                self.logger.info("-" * 60)
                self.logger.info("Metrics (Macro Average):")
                self.logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
                self.logger.info(f"  Precision: {metrics['precision_macro']:.4f}")
                self.logger.info(f"  Recall:    {metrics['recall_macro']:.4f}")
                self.logger.info(f"  F1-Score:  {metrics['f1_macro']:.4f}")
                self.logger.info("-" * 60)
                self.logger.info("Metrics (Weighted Average):")
                self.logger.info(f"  Precision: {metrics['precision_weighted']:.4f}")
                self.logger.info(f"  Recall:    {metrics['recall_weighted']:.4f}")
                self.logger.info(f"  F1-Score:  {metrics['f1_weighted']:.4f}")
                self.logger.info("=" * 60)

            else:
                self.logger.warning(
                    f"No test data for '{large_cat}', skipping evaluation"
                )

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(f"Total models trained: {len(self.knn_models)}")
        self.logger.info("=" * 60)

        return all_metrics

    def impute(
        self,
        category_df: pd.DataFrame,
        use_diner_info: bool = False,
    ) -> pd.DataFrame:
        """
        중분류 null 값을 대분류별 모델로 imputation

        Args:
            category_df: 카테고리 데이터프레임
            use_diner_info: diner.csv 정보 사용 여부

        Returns:
            imputation이 적용된 데이터프레임
        """
        if not self.knn_models:
            raise ValueError(
                "Models not trained. Call fit() first or use fit_and_impute()"
            )

        result_df = category_df.copy()

        # 중분류가 null인 행 찾기
        null_mask = result_df["diner_category_middle"].isna()

        if not null_mask.any():
            self.logger.info("No null values to impute")
            return result_df

        # 대분류별로 처리
        total_filled = 0

        for large_cat in self.knn_models.keys():
            # 해당 대분류이면서 중분류가 null인 행 찾기
            large_cat_mask = result_df["diner_category_large"] == large_cat
            null_and_large_mask = null_mask & large_cat_mask

            if not null_and_large_mask.any():
                continue

            # 해당 대분류의 데이터만 추출
            large_cat_df = result_df[null_and_large_mask].copy()

            # Feature 준비 (대분류 정보는 이미 알고 있으므로 제외)
            features_df = self._prepare_features(
                large_cat_df, use_diner_info, large_category=large_cat
            )

            X = self._encode_features(features_df, large_cat, is_training=False)

            if len(X) > 0:
                try:
                    # 예측
                    y_pred_encoded = self.knn_models[large_cat].predict(X)
                    y_pred = self.target_encoders[large_cat].inverse_transform(
                        y_pred_encoded
                    )

                    # Imputation 적용
                    result_df.loc[null_and_large_mask, "diner_category_middle"] = y_pred

                    filled_count = null_and_large_mask.sum()
                    total_filled += filled_count
                    self.logger.info(
                        f"Imputed {filled_count:,} null middle category values for '{large_cat}' using KNN"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to impute for '{large_cat}': {e}")

        if total_filled > 0:
            self.logger.info(
                f"Total imputed {total_filled:,} null middle category values across all large categories"
            )
        else:
            self.logger.info("No values were imputed (no matching models or data)")

        return result_df

    def fit_and_impute(
        self,
        category_df: pd.DataFrame,
        use_diner_info: bool = False,
        test_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """
        모델 학습 및 imputation을 한 번에 수행

        Args:
            category_df: 카테고리 데이터프레임
            use_diner_info: diner.csv 정보 사용 여부
            test_size: 평가용 데이터 비율

        Returns:
            (imputation된 데이터프레임, 대분류별 평가 메트릭)
        """
        # 학습 및 평가
        metrics = self.fit(category_df, use_diner_info, test_size)

        # Imputation
        result_df = self.impute(category_df, use_diner_info)

        return result_df, metrics


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

        self.df.loc[target_rows & is_grilled, "diner_category_middle"] = "숯불치킨"
        self.df.loc[target_rows & ~is_grilled, "diner_category_middle"] = "치킨전문점"

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


class MiddleCategoryLLMImputer:
    """
    LLM 기반 중분류 null 값 imputation 클래스

    각 대분류별로 few-shot prompt를 구성하여 오픈소스 LLM으로 중분류를 예측합니다.
    대분류, 소분류, 상세분류, diner_name, diner_tag 등의 정보를 활용합니다.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        data_path: Optional[str] = None,
        n_few_shot: int = 5,
        max_new_tokens: int = 10,
        temperature: float = 0.1,
        device: Optional[str] = None,
        batch_size: int = 8,
        logger: logging.Logger = logger,
    ):
        """
        Args:
            model_name: 사용할 LLM 모델 이름 (HuggingFace 모델 ID)
            data_path: diner.csv 파일 경로 (diner_name, diner_tag 등 사용 시)
            n_few_shot: Few-shot 예제 개수 (기본값: 5)
            max_new_tokens: 생성할 최대 토큰 수 (기본값: 10)
            temperature: 생성 시 temperature (기본값: 0.1, 낮을수록 결정적)
            device: 사용할 디바이스 ("cuda", "cpu", None=자동)
            batch_size: 배치 크기 (기본값: 8)
            logger: 로거 인스턴스
        """
        self.model_name = model_name
        self.data_path = Path(data_path) if data_path else None
        self.n_few_shot = n_few_shot
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.logger = logger

        # 디바이스 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 모델 및 토크나이저 (lazy loading)
        # 타입 힌트는 Any로 설정 (테스트 환경에서 AutoTokenizer가 None일 수 있음)
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None

        # 대분류별 few-shot 예제 저장
        self.few_shot_examples: Dict[str, List[Dict[str, str]]] = {}

        # 대분류별 가능한 중분류 목록 저장
        self.possible_middle_categories: Dict[str, List[str]] = {}

        # 프롬프트 템플릿 로드
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """
        프롬프트 템플릿 파일을 로드합니다.

        Returns:
            프롬프트 템플릿 문자열
        """
        # 프로젝트 루트 경로 찾기 (src/yamyam_lab/preprocess/diner_transform.py -> 프로젝트 루트)
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        prompt_path = project_root / "prompt" / "middle_category_imputation.txt"

        if not prompt_path.exists():
            self.logger.warning(
                f"Prompt template not found at {prompt_path}, using default template"
            )
            # 기본 템플릿 반환
            return (
                "당신은 음식점 카테고리 분류 전문가입니다.\n"
                "주어진 음식점 정보를 바탕으로 '{large_category}' 대분류에 속하는 중분류를 예측하세요.\n\n"
                "중요:\n"
                "- 예제들을 참고하여 유사한 패턴을 찾으세요\n"
                "- 소분류, 상세분류, 식당명, 태그 등의 정보를 종합적으로 고려하세요\n"
                "- 중분류 이름만 정확히 답변하세요 (설명 없이)\n"
                "{possible_middles_section}\n\n"
                "예제:\n{examples_section}\n\n"
                "예측할 음식점:\n정보: {query_features}\n중분류:\n"
            )

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()
            self.logger.info(f"Loaded prompt template from {prompt_path}")
            return template
        except Exception as e:
            self.logger.error(f"Failed to load prompt template: {e}")
            raise

    def _load_model(self):
        """LLM 모델과 토크나이저를 로드합니다 (lazy loading)"""
        # 테스트 환경에서는 transformers를 사용할 수 없으므로 스킵
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            self.logger.warning(
                "transformers is not available (likely in test environment). "
                "Skipping model loading."
            )
            return

        if self.tokenizer is None or self.model is None:
            self.logger.info(f"Loading LLM model: {self.model_name}")
            self.logger.info(f"Using device: {self.device}")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)

                self.logger.info(f"Successfully loaded model: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load model {self.model_name}: {e}")
                raise

    def _prepare_features_text(
        self,
        row: pd.Series,
        use_diner_info: bool = False,
    ) -> str:
        """
        행의 정보를 텍스트로 변환하여 prompt에 사용

        Args:
            row: 데이터프레임의 한 행
            use_diner_info: diner.csv 정보 사용 여부

        Returns:
            feature 텍스트
        """
        features = []

        # 대분류는 이미 알고 있으므로 제외
        if pd.notna(row.get("diner_category_small")):
            features.append(f"소분류: {row['diner_category_small']}")
        if pd.notna(row.get("diner_category_detail")):
            features.append(f"상세분류: {row['diner_category_detail']}")

        if use_diner_info:
            if pd.notna(row.get("diner_name")):
                features.append(f"식당명: {row['diner_name']}")
            if pd.notna(row.get("diner_tag")):
                tag = row["diner_tag"]
                if isinstance(tag, list):
                    tag = ", ".join(tag)
                features.append(f"태그: {tag}")
            if pd.notna(row.get("diner_menu_name")):
                menu = row["diner_menu_name"]
                if isinstance(menu, list):
                    menu = ", ".join(menu)
                features.append(f"메뉴: {menu}")

        return ", ".join(features) if features else "정보 없음"

    def _build_few_shot_prompt(
        self,
        large_category: str,
        examples: List[Dict[str, str]],
        query_features: str,
        possible_middles: Optional[List[str]] = None,
    ) -> str:
        """
        Few-shot prompt 구성

        Args:
            large_category: 대분류
            examples: Few-shot 예제 리스트 [{"features": "...", "middle": "..."}, ...]
            query_features: 예측할 행의 feature 텍스트
            possible_middles: 가능한 중분류 목록 (프롬프트에 포함)

        Returns:
            구성된 prompt
        """
        # 가능한 중분류 목록 섹션 구성
        if possible_middles:
            possible_middles_section = (
                f"- 가능한 중분류: {', '.join(sorted(possible_middles))}\n"
            )
        else:
            possible_middles_section = ""

        # Few-shot 예제 섹션 구성
        examples_section = ""
        for i, ex in enumerate(examples, 1):
            examples_section += (
                f"\n예제 {i}:\n정보: {ex['features']}\n중분류: {ex['middle']}\n"
            )

        # 템플릿에 값 채우기
        prompt = self.prompt_template.format(
            large_category=large_category,
            possible_middles_section=possible_middles_section,
            examples_section=examples_section,
            query_features=query_features,
        )

        return prompt

    def _predict_with_llm(
        self, prompt: str, possible_middles: Optional[List[str]] = None
    ) -> str:
        """
        LLM을 사용하여 중분류 예측 (단일)

        Args:
            prompt: 입력 prompt
            possible_middles: 가능한 중분류 목록 (매칭용)

        Returns:
            예측된 중분류
        """
        results = self._predict_with_llm_batch([prompt], possible_middles)
        return results[0] if results else ""

    def _predict_with_llm_batch(
        self,
        prompts: List[str],
        possible_middles: Optional[List[str]] = None,
    ) -> List[str]:
        """
        LLM을 사용하여 중분류 예측 (배치)

        Args:
            prompts: 입력 prompt 리스트
            possible_middles: 가능한 중분류 목록 (매칭용)

        Returns:
            예측된 중분류 리스트
        """
        if self.tokenizer is None or self.model is None:
            self._load_model()

        if not prompts:
            return []

        # Prompt를 모델 형식에 맞게 변환
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]

            # Qwen2 형식에 맞게 포맷팅
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt

            formatted_prompts.append(formatted_prompt)

        # 배치 토크나이징
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True,
        ).to(self.device)

        # 배치 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 배치 디코딩
        input_lengths = inputs["input_ids"].shape[1]
        predictions = []
        for i, output in enumerate(outputs):
            generated_text = self.tokenizer.decode(
                output[input_lengths:], skip_special_tokens=True
            )

            # 중분류만 추출 (첫 줄 또는 첫 단어)
            predicted = generated_text.strip().split("\n")[0].split(",")[0].strip()

            # 가능한 중분류 목록이 있으면 매칭 시도
            if possible_middles:
                # 정확히 일치하는 경우
                if predicted in possible_middles:
                    predictions.append(predicted)
                    continue

                # 부분 일치 확인
                matched = None
                for middle in possible_middles:
                    if predicted in middle or middle in predicted:
                        matched = middle
                        break

                if matched:
                    predictions.append(matched)
                else:
                    # 매칭 실패 시 첫 번째 가능한 값 반환 (fallback)
                    predictions.append(
                        possible_middles[0] if possible_middles else predicted
                    )
            else:
                predictions.append(predicted)

        return predictions

    def _prepare_few_shot_examples(
        self,
        large_cat_df: pd.DataFrame,
        use_diner_info: bool = False,
    ) -> List[Dict[str, str]]:
        """
        대분류별 few-shot 예제 준비

        Args:
            large_cat_df: 해당 대분류의 데이터프레임 (중분류가 있는 데이터)
            use_diner_info: diner.csv 정보 사용 여부

        Returns:
            Few-shot 예제 리스트
        """
        # 중분류가 있는 데이터만 사용
        valid_df = large_cat_df[large_cat_df["diner_category_middle"].notna()].copy()

        if len(valid_df) == 0:
            return []

        # 중분류별로 샘플링하여 다양성 확보
        examples = []
        middle_categories = valid_df["diner_category_middle"].unique()

        samples_per_category = max(1, self.n_few_shot // len(middle_categories))

        for middle_cat in middle_categories:
            middle_df = valid_df[valid_df["diner_category_middle"] == middle_cat]
            sample_size = min(samples_per_category, len(middle_df))
            sampled = middle_df.sample(n=sample_size, random_state=42)

            for _, row in sampled.iterrows():
                features = self._prepare_features_text(row, use_diner_info)
                examples.append(
                    {
                        "features": features,
                        "middle": str(row["diner_category_middle"]),
                    }
                )

        # 전체 예제가 n_few_shot보다 적으면 추가 샘플링
        if len(examples) < self.n_few_shot:
            remaining = self.n_few_shot - len(examples)
            additional = valid_df.sample(
                n=min(remaining, len(valid_df)), random_state=42
            )
            for _, row in additional.iterrows():
                features = self._prepare_features_text(row, use_diner_info)
                examples.append(
                    {
                        "features": features,
                        "middle": str(row["diner_category_middle"]),
                    }
                )

        # 최대 n_few_shot개만 반환
        return examples[: self.n_few_shot]

    def fit(
        self,
        category_df: pd.DataFrame,
        use_diner_info: bool = False,
        test_size: float = 0.2,
    ) -> Dict[str, Dict[str, float]]:
        """
        각 대분류별로 few-shot 예제 준비 및 평가

        Args:
            category_df: 카테고리 데이터프레임
            use_diner_info: diner.csv 정보 사용 여부
            test_size: 평가용 데이터 비율

        Returns:
            대분류별 평가 메트릭 딕셔너리 {large_category: {metric: value}}
        """
        # diner.csv 정보 병합 (필요한 경우)
        if use_diner_info and self.data_path:
            diner_path = self.data_path / "diner.csv"
            if diner_path.exists():
                try:
                    diner_df = pd.read_csv(diner_path)
                    if (
                        "diner_idx" in category_df.columns
                        and "diner_idx" in diner_df.columns
                    ):
                        # diner_menu_name이 있는 경우에만 포함
                        merge_cols = ["diner_idx", "diner_name", "diner_tag"]
                        if "diner_menu_name" in diner_df.columns:
                            merge_cols.append("diner_menu_name")
                        category_df = category_df.merge(
                            diner_df[merge_cols],
                            on="diner_idx",
                            how="left",
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to load diner.csv: {e}")

        # 중분류가 null이 아닌 데이터만 사용
        train_df = category_df[category_df["diner_category_middle"].notna()].copy()

        if len(train_df) == 0:
            raise ValueError("No non-null middle category data for training")

        # 대분류별로 데이터 분리
        all_metrics = {}
        large_categories = train_df["diner_category_large"].unique()

        self.logger.info("=" * 60)
        self.logger.info("Preparing LLM Few-Shot Examples by Large Category")
        self.logger.info("=" * 60)
        self.logger.info(f"Total large categories: {len(large_categories)}")
        self.logger.info("")

        for large_cat in tqdm(
            sorted(large_categories), desc="Processing categories", leave=True
        ):
            if pd.isna(large_cat):
                continue

            # 해당 대분류의 데이터만 필터링
            large_cat_df = train_df[
                train_df["diner_category_large"] == large_cat
            ].copy()

            if len(large_cat_df) < 2:
                self.logger.warning(
                    f"Skipping '{large_cat}': insufficient data ({len(large_cat_df)} samples)"
                )
                continue

            # 가능한 중분류 목록 저장
            unique_middles = large_cat_df["diner_category_middle"].unique().tolist()
            self.possible_middle_categories[large_cat] = unique_middles

            # Few-shot 예제 준비
            examples = self._prepare_few_shot_examples(large_cat_df, use_diner_info)
            self.few_shot_examples[large_cat] = examples

            # Train/Test split
            try:
                train_split, test_split = train_test_split(
                    large_cat_df,
                    test_size=test_size,
                    random_state=42,
                    stratify=large_cat_df["diner_category_middle"],
                )
            except ValueError:
                # stratify가 실패하면 stratify 없이 split
                train_split, test_split = train_test_split(
                    large_cat_df, test_size=test_size, random_state=42
                )

            if len(test_split) == 0:
                self.logger.warning(
                    f"No test data for '{large_cat}', skipping evaluation"
                )
                continue

            # 평가
            y_true = []
            y_pred = []

            self.logger.info("=" * 60)
            self.logger.info(f"Large Category: {large_cat}")
            self.logger.info("=" * 60)
            self.logger.info(f"Training samples: {len(train_split)}")
            self.logger.info(f"Test samples: {len(test_split)}")
            self.logger.info(f"Few-shot examples: {len(examples)}")
            self.logger.info(f"Unique middle categories: {len(unique_middles)}")

            # 테스트 데이터에 대해 배치 예측
            test_rows = list(test_split.iterrows())
            prompts_list = []
            indices_list = []

            for idx, row in test_rows:
                query_features = self._prepare_features_text(row, use_diner_info)
                prompt = self._build_few_shot_prompt(
                    large_cat, examples, query_features, unique_middles
                )
                prompts_list.append(prompt)
                indices_list.append((idx, row))

            # 배치 단위로 예측
            y_true = []
            y_pred = []
            for i in tqdm(
                range(0, len(prompts_list), self.batch_size),
                desc=f"Predicting {large_cat}",
                leave=False,
            ):
                batch_prompts = prompts_list[i : i + self.batch_size]
                batch_indices = indices_list[i : i + self.batch_size]

                try:
                    batch_predictions = self._predict_with_llm_batch(
                        batch_prompts, unique_middles
                    )

                    for (idx, row), predicted in zip(batch_indices, batch_predictions):
                        y_true.append(str(row["diner_category_middle"]))
                        y_pred.append(predicted)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to predict batch for '{large_cat}': {e}"
                    )
                    # Fallback: 첫 번째 가능한 값 사용
                    fallback = unique_middles[0] if unique_middles else ""
                    for idx, row in batch_indices:
                        y_true.append(str(row["diner_category_middle"]))
                        y_pred.append(fallback)

            if len(y_true) > 0 and len(y_pred) > 0:
                metrics = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision_macro": precision_score(
                        y_true, y_pred, average="macro", zero_division=0
                    ),
                    "recall_macro": recall_score(
                        y_true, y_pred, average="macro", zero_division=0
                    ),
                    "f1_macro": f1_score(
                        y_true, y_pred, average="macro", zero_division=0
                    ),
                    "precision_weighted": precision_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    ),
                    "recall_weighted": recall_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    ),
                    "f1_weighted": f1_score(
                        y_true, y_pred, average="weighted", zero_division=0
                    ),
                }
                all_metrics[large_cat] = metrics

                self.logger.info("-" * 60)
                self.logger.info("Metrics (Macro Average):")
                self.logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
                self.logger.info(f"  Precision: {metrics['precision_macro']:.4f}")
                self.logger.info(f"  Recall:    {metrics['recall_macro']:.4f}")
                self.logger.info(f"  F1-Score:  {metrics['f1_macro']:.4f}")
                self.logger.info("-" * 60)
                self.logger.info("Metrics (Weighted Average):")
                self.logger.info(f"  Precision: {metrics['precision_weighted']:.4f}")
                self.logger.info(f"  Recall:    {metrics['recall_weighted']:.4f}")
                self.logger.info(f"  F1-Score:  {metrics['f1_weighted']:.4f}")
                self.logger.info("=" * 60)

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(f"Total categories prepared: {len(self.few_shot_examples)}")
        self.logger.info("=" * 60)

        return all_metrics

    def impute(
        self,
        category_df: pd.DataFrame,
        use_diner_info: bool = False,
    ) -> pd.DataFrame:
        """
        중분류 null 값을 대분류별 LLM 모델로 imputation

        Args:
            category_df: 카테고리 데이터프레임
            use_diner_info: diner.csv 정보 사용 여부

        Returns:
            imputation이 적용된 데이터프레임
        """
        if not self.few_shot_examples:
            raise ValueError(
                "Few-shot examples not prepared. Call fit() first or use fit_and_impute()"
            )

        # diner.csv 정보 병합 (필요한 경우)
        if use_diner_info and self.data_path:
            diner_path = self.data_path / "diner.csv"
            if diner_path.exists():
                try:
                    diner_df = pd.read_csv(diner_path)
                    if (
                        "diner_idx" in category_df.columns
                        and "diner_idx" in diner_df.columns
                    ):
                        # diner_menu_name이 있는 경우에만 포함
                        merge_cols = ["diner_idx", "diner_name", "diner_tag"]
                        if "diner_menu_name" in diner_df.columns:
                            merge_cols.append("diner_menu_name")
                        category_df = category_df.merge(
                            diner_df[merge_cols],
                            on="diner_idx",
                            how="left",
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to load diner.csv: {e}")

        result_df = category_df.copy()

        # 중분류가 null인 행 찾기
        null_mask = result_df["diner_category_middle"].isna()

        if not null_mask.any():
            self.logger.info("No null values to impute")
            return result_df

        # 대분류별로 처리
        total_filled = 0

        for large_cat in tqdm(
            self.few_shot_examples.keys(),
            desc="Imputing by category",
            leave=True,
        ):
            # 해당 대분류이면서 중분류가 null인 행 찾기
            large_cat_mask = result_df["diner_category_large"] == large_cat
            null_and_large_mask = null_mask & large_cat_mask

            if not null_and_large_mask.any():
                continue

            # 해당 대분류의 데이터만 추출
            large_cat_df = result_df[null_and_large_mask].copy()
            examples = self.few_shot_examples[large_cat]
            possible_middles = self.possible_middle_categories.get(large_cat, [])

            # 배치 단위로 처리
            rows_list = list(large_cat_df.iterrows())
            prompts_list = []
            indices_list = []

            for idx, row in rows_list:
                query_features = self._prepare_features_text(row, use_diner_info)
                prompt = self._build_few_shot_prompt(
                    large_cat, examples, query_features, possible_middles
                )
                prompts_list.append(prompt)
                indices_list.append(idx)

            # 배치 단위로 예측
            filled_count = 0
            for i in tqdm(
                range(0, len(prompts_list), self.batch_size),
                desc=f"Imputing {large_cat}",
                leave=False,
            ):
                batch_prompts = prompts_list[i : i + self.batch_size]
                batch_indices = indices_list[i : i + self.batch_size]

                try:
                    batch_predictions = self._predict_with_llm_batch(
                        batch_prompts, possible_middles
                    )

                    for idx, predicted in zip(batch_indices, batch_predictions):
                        if predicted:
                            result_df.loc[idx, "diner_category_middle"] = predicted
                            filled_count += 1
                except Exception as e:
                    self.logger.warning(
                        f"Failed to impute batch for '{large_cat}': {e}"
                    )

            total_filled += filled_count
            if filled_count > 0:
                self.logger.info(
                    f"Imputed {filled_count:,} null middle category values for '{large_cat}' using LLM"
                )

        if total_filled > 0:
            self.logger.info(
                f"Total imputed {total_filled:,} null middle category values across all large categories"
            )
        else:
            self.logger.info("No values were imputed")

        return result_df

    def fit_and_impute(
        self,
        category_df: pd.DataFrame,
        use_diner_info: bool = False,
        test_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """
        Few-shot 예제 준비 및 imputation을 한 번에 수행

        Args:
            category_df: 카테고리 데이터프레임
            use_diner_info: diner.csv 정보 사용 여부
            test_size: 평가용 데이터 비율

        Returns:
            (imputation된 데이터프레임, 대분류별 평가 메트릭)
        """
        # Few-shot 예제 준비 및 평가
        metrics = self.fit(category_df, use_diner_info, test_size)

        # Imputation
        result_df = self.impute(category_df, use_diner_info)

        return result_df, metrics
