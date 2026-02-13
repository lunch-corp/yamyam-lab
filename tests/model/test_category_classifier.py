"""Integration tests for category classifier pipeline using mock data."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from yamyam_lab.data.category_loader import CategoryData
from yamyam_lab.model.classification import CatBoostCategoryClassifier
from yamyam_lab.model.embedding import TfidfEmbedder


@pytest.fixture
def mock_category_data():
    """Create small synthetic CategoryData for testing the classifier pipeline."""
    rng = np.random.RandomState(42)

    large_categories = ["한식", "양식", "일식", "중식"]
    hierarchy = {
        "한식": ["고기", "찌개", "국밥"],
        "양식": ["이탈리안", "스테이크"],
        "일식": ["초밥", "라멘"],
        "중식": ["중화요리"],
    }
    n_train = 80
    n_val = 20
    n_missing = 10

    def make_rows(n, include_middle=True):
        rows = []
        for _ in range(n):
            large = rng.choice(large_categories)
            middle = rng.choice(hierarchy[large]) if include_middle else np.nan
            diner_name = rng.choice(["맛집", "식당", "레스토랑", "카페"])
            menu = rng.choice(
                ["삼겹살 목살", "파스타 피자", "초밥 사시미", "짜장면 짬뽕"]
            )
            rows.append(
                {
                    "diner_idx": rng.randint(1, 100000),
                    "diner_category_large": large,
                    "diner_category_middle": middle,
                    "diner_name": diner_name,
                    "combined_text": f"{diner_name} {menu}",
                }
            )
        return pd.DataFrame(rows)

    df_train = make_rows(n_train)
    df_val = make_rows(n_val)
    df_missing = make_rows(n_missing, include_middle=False)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train["diner_category_middle"])
    y_val = label_encoder.transform(df_val["diner_category_middle"])

    # Simple space-split tokenization (no Kiwi dependency needed)
    train_tokenized = df_train["combined_text"].tolist()
    val_tokenized = df_val["combined_text"].tolist()
    missing_tokenized = df_missing["combined_text"].tolist()

    # Build category_df as union of all rows for save_results
    category_df = pd.concat([df_train, df_val, df_missing], ignore_index=True)

    return CategoryData(
        category_df=category_df,
        diner_df=pd.DataFrame(),
        menu_df=pd.DataFrame(),
        df_train=df_train,
        df_val=df_val,
        df_missing=df_missing,
        y_train=y_train,
        y_val=y_val,
        label_encoder=label_encoder,
        train_tokenized=train_tokenized,
        val_tokenized=val_tokenized,
        missing_tokenized=missing_tokenized,
        hierarchy=hierarchy,
    )


class TestCategoryClassifier:
    """Integration tests for the category classifier pipeline."""

    def _build_classifier(self, tmp_path, use_large_category=True):
        embedder = TfidfEmbedder(
            max_features=500,
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 1),
            cache_dir=str(tmp_path / "cache"),
        )
        return CatBoostCategoryClassifier(
            embedder=embedder,
            iterations=5,
            learning_rate=0.1,
            depth=3,
            early_stopping_rounds=3,
            verbose=0,
            output_dir=str(tmp_path / "output"),
            use_large_category=use_large_category,
            min_class_samples=1,
        )

    def test_pipeline_with_large_category(self, tmp_path, mock_category_data):
        """Test full pipeline with large category feature enabled."""
        classifier = self._build_classifier(tmp_path, use_large_category=True)

        with patch.object(classifier, "load_data") as mock_load:
            mock_load.return_value = mock_category_data
            classifier.data = mock_category_data

            classifier.prepare_embeddings(force_recompute=True)

            # TF-IDF shape unchanged; cat features stored separately
            tfidf_dim = classifier.embedder.get_embedding_dim()
            assert classifier.X_train.shape[1] == tfidf_dim
            assert classifier.cat_feature_indices == [tfidf_dim]
            assert classifier._cat_train is not None

            classifier.build_model()
            classifier.train_model()

            result = classifier.evaluate(
                classifier.X_val,
                mock_category_data.y_val,
                "Validation",
                df=mock_category_data.df_val,
            )
            assert 0.0 <= result.accuracy <= 1.0
            assert 0.0 <= result.f1_weighted <= 1.0

            predictions = classifier.predict_missing()
            assert len(predictions) == len(mock_category_data.df_missing)
            assert "predicted_middle_category" in predictions.columns
            assert "prediction_confidence" in predictions.columns

            classifier.save_results(result, predictions)
            assert (tmp_path / "output" / "model.pkl").exists()

    def test_pipeline_without_large_category(self, tmp_path, mock_category_data):
        """Test full pipeline with large category feature disabled."""
        classifier = self._build_classifier(tmp_path, use_large_category=False)

        with patch.object(classifier, "load_data") as mock_load:
            mock_load.return_value = mock_category_data
            classifier.data = mock_category_data

            classifier.prepare_embeddings(force_recompute=True)

            tfidf_dim = classifier.embedder.get_embedding_dim()
            assert classifier.X_train.shape[1] == tfidf_dim
            assert classifier.cat_feature_indices is None

            classifier.build_model()
            classifier.train_model()

            result = classifier.evaluate(
                classifier.X_val,
                mock_category_data.y_val,
                "Validation",
                df=mock_category_data.df_val,
            )
            assert 0.0 <= result.accuracy <= 1.0

            classifier.save_results(result)
            assert (tmp_path / "output" / "model.pkl").exists()
