import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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

        original_values = category_df["diner_category_middle"].copy()

        self.logger.info(f"Original category_df shape: {category_df.shape}")
        self.logger.info(
            f"Null middle categories: {category_df['diner_category_middle'].isna().sum()}"
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
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            except ValueError:
                # stratify가 실패하면 (데이터가 적은 경우) stratify 없이 split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
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
                self.logger.info(f"Unique middle categories: {unique_middles}")
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
