"""CatBoost classifier for category classification."""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from yamyam_lab.model.classification.base_classifier import BaseClassifier
from yamyam_lab.model.embedding.base_embedder import BaseEmbedder


class CatBoostCategoryClassifier(BaseClassifier):
    """
    CatBoost-based category classifier.

    Uses gradient boosting on text embeddings for multi-class classification.
    Supports native categorical features via CatBoost's cat_features parameter.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        iterations: int = 300,
        learning_rate: float = 0.1,
        depth: int = 8,
        l2_leaf_reg: float = 3.0,
        early_stopping_rounds: int = 30,
        verbose: int = 50,
        **kwargs,
    ):
        super().__init__(embedder=embedder, **kwargs)

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

    @property
    def name(self) -> str:
        return f"catboost_{self.embedder.name}"

    def _make_pool(
        self,
        X_sparse,
        y=None,
        cat_values: np.ndarray = None,
    ) -> Pool:
        """Build CatBoost Pool from sparse embeddings + categorical values."""
        df = pd.DataFrame(X_sparse.toarray())
        if cat_values is not None:
            df[X_sparse.shape[1]] = cat_values
            return Pool(df, label=y, cat_features=[X_sparse.shape[1]])
        return Pool(df, label=y)

    def build_model(self) -> None:
        """Build CatBoost classifier."""
        print("\nBuilding CatBoost model...")

        self.model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_state,
            verbose=self.verbose,
            early_stopping_rounds=self.early_stopping_rounds,
            task_type="CPU",
            thread_count=-1,
        )

        print(f"  Iterations: {self.iterations}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Depth: {self.depth}")

    def train_model(self) -> None:
        """Train CatBoost model."""
        print("\nTraining CatBoost model...")
        print(f"  X_train: {self.X_train.shape}")
        print(f"  X_val: {self.X_val.shape}")

        if self.cat_feature_indices:
            print(f"  cat_features: {self.cat_feature_indices}")
            train_pool = self._make_pool(
                self.X_train, self.data.y_train, self._cat_train
            )
            val_pool = self._make_pool(self.X_val, self.data.y_val, self._cat_val)
            self.model.fit(train_pool, eval_set=val_pool, verbose=self.verbose)
        else:
            self.model.fit(
                self.X_train,
                self.data.y_train,
                eval_set=(self.X_val, self.data.y_val),
                verbose=self.verbose,
            )

        print(f"  Best iteration: {self.model.best_iteration_}")

    def _predict_pool(self, X_sparse, cat_values=None):
        """Create Pool for prediction and return predict_proba result."""
        pool = self._make_pool(X_sparse, cat_values=cat_values)
        return self.model.predict_proba(pool)

    def evaluate(self, X, y_true, set_name="Validation", df=None):
        """Evaluate model, using Pool with cat_features for prediction."""
        print(f"\nEvaluating on {set_name} set...")

        cat_values = None
        if self.cat_feature_indices and df is not None:
            cat_values = self._get_large_categories(df)

        if df is not None and hasattr(self.model, "predict_proba"):
            y_proba = self._predict_pool(X, cat_values)
            y_pred, _ = self._constrained_predict(
                y_proba, large_categories=df["diner_category_large"].values
            )
        else:
            if cat_values is not None:
                pool = self._make_pool(X, cat_values=cat_values)
                y_pred = self.model.predict(pool)
            else:
                y_pred = self.model.predict(X)

        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
        )

        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")

        report = classification_report(
            y_true,
            y_pred,
            target_names=self.data.label_encoder.classes_,
            output_dict=True,
        )

        print(f"\n{set_name} Report (top 15 classes by support):")
        class_metrics = [
            (cls, m)
            for cls, m in report.items()
            if isinstance(m, dict) and "support" in m
        ]
        class_metrics.sort(key=lambda x: x[1]["support"], reverse=True)

        print(f"{'Category':<30} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>10}")
        print("-" * 66)
        for cls, m in class_metrics[:15]:
            print(
                f"{cls:<30} {m['precision']:>8.3f} {m['recall']:>8.3f} "
                f"{m['f1-score']:>8.3f} {int(m['support']):>10}"
            )

        from yamyam_lab.model.classification.base_classifier import (
            ClassificationResult,
        )

        return ClassificationResult(
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            report=report,
            predictions=y_pred,
        )

    def predict_missing(self):
        """Predict missing category values using Pool with cat_features."""
        if self.X_missing is None or len(self.data.df_missing) == 0:
            print("No missing data to predict.")
            return pd.DataFrame()

        print(f"\nPredicting {len(self.data.df_missing):,} missing values...")

        if hasattr(self.model, "predict_proba"):
            y_proba = self._predict_pool(self.X_missing, self._cat_missing)
            y_pred, y_conf = self._constrained_predict(y_proba)
            y_pred_labels = self.data.label_encoder.inverse_transform(y_pred)
        else:
            pool = self._make_pool(self.X_missing, cat_values=self._cat_missing)
            y_pred = self.model.predict(pool)
            y_pred_labels = self.data.label_encoder.inverse_transform(y_pred)
            y_conf = None

        result = self.data.df_missing[["diner_idx"]].copy()
        result["predicted_middle_category"] = y_pred_labels

        if y_conf is not None:
            result["prediction_confidence"] = y_conf

        print("\nPrediction distribution (top 10):")
        pred_dist = result["predicted_middle_category"].value_counts().head(10)
        for cat, count in pred_dist.items():
            print(f"  {cat}: {count:,}")

        return result
