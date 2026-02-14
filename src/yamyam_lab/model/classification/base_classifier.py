"""Base classifier for category classification."""

import pickle
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from yamyam_lab.data.category_loader import CategoryData, CategoryDataLoader
from yamyam_lab.model.embedding.base_embedder import BaseEmbedder


@dataclass
class ClassificationResult:
    """Container for classification results."""

    accuracy: float
    f1_macro: float
    f1_weighted: float
    report: dict
    predictions: np.ndarray = None
    probabilities: np.ndarray = None


class BaseClassifier(ABC):
    """
    Abstract base class for category classifiers.

    Provides shared functionality:
    - Data loading and preprocessing
    - Embedding management
    - Evaluation metrics
    - Prediction and saving
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        output_dir: str = "result/category_classifier",
        data_dir: str = "data",
        cache_dir: str = "cache/classification",
        use_menu: bool = True,
        use_large_category: bool = True,
        val_ratio: float = 0.1,
        min_class_samples: int = 10,
        random_state: int = 42,
        sample_size: int = None,
    ):
        self.embedder = embedder
        self.output_dir = Path(output_dir)
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.use_menu = use_menu
        self.use_large_category = use_large_category
        self.val_ratio = val_ratio
        self.min_class_samples = min_class_samples
        self.random_state = random_state
        self.sample_size = sample_size

        self.data: CategoryData = None
        self.X_train = None
        self.X_val = None
        self.X_missing = None
        self.model = None
        self.cat_feature_indices: list[int] = None
        self._cat_train: np.ndarray = None
        self._cat_val: np.ndarray = None
        self._cat_missing: np.ndarray = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Return classifier name."""
        raise NotImplementedError

    def load_data(self, force_reload: bool = False) -> CategoryData:
        """Load and preprocess data."""
        loader = CategoryDataLoader(
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            use_menu=self.use_menu,
            val_ratio=self.val_ratio,
            min_class_samples=self.min_class_samples,
            random_state=self.random_state,
            sample_size=self.sample_size,
        )
        self.data = loader.load(force_reload=force_reload)
        return self.data

    def prepare_embeddings(self, force_recompute: bool = False) -> None:
        """
        Prepare embeddings for train, val, and missing data.

        Uses caching to avoid recomputation.
        """
        if self.data is None:
            raise RuntimeError("Data must be loaded first. Call load_data().")

        print(f"\nPreparing embeddings with {self.embedder.name}...")

        # Try to load cached embeddings
        if not force_recompute:
            self.X_train = self.embedder.load_embeddings("train")
            self.X_val = self.embedder.load_embeddings("val")
            self.X_missing = self.embedder.load_embeddings("missing")

            if self.X_train is not None and self.X_val is not None:
                # Validate cached dimensions match current data
                n_train = len(self.data.train_tokenized)
                n_val = len(self.data.val_tokenized)
                if self.X_train.shape[0] != n_train or self.X_val.shape[0] != n_val:
                    print(
                        f"  Cached embeddings dimension mismatch "
                        f"(cached train={self.X_train.shape[0]}, "
                        f"expected={n_train}). Recomputing..."
                    )
                    self.X_train = None
                    self.X_val = None
                    self.X_missing = None
                # Also need to load the fitted embedder
                elif (
                    embedder_path := self.embedder.cache_dir
                    / f"{self.embedder.name}.pkl"
                ).exists():
                    self.embedder = BaseEmbedder.load(embedder_path)
                    print("Using cached embeddings.")

                    # Still need to append large category features
                    if self.use_large_category:
                        self._prepare_large_category_features()

                    print(f"  Train shape: {self.X_train.shape}")
                    print(f"  Val shape: {self.X_val.shape}")
                    if self.X_missing is not None:
                        print(f"  Missing shape: {self.X_missing.shape}")
                    return

        # Fit embedder on training data
        self.embedder.fit(self.data.train_tokenized)
        self.embedder.save()

        # Transform all sets
        print("Transforming training data...")
        self.X_train = self.embedder.transform(self.data.train_tokenized)
        self.embedder.save_embeddings(self.X_train, "train")

        print("Transforming validation data...")
        self.X_val = self.embedder.transform(self.data.val_tokenized)
        self.embedder.save_embeddings(self.X_val, "val")

        if len(self.data.missing_tokenized) > 0:
            print("Transforming missing data...")
            self.X_missing = self.embedder.transform(self.data.missing_tokenized)
            self.embedder.save_embeddings(self.X_missing, "missing")

        # Prepare large category feature (stored separately for CatBoost cat_features)
        if self.use_large_category:
            self._prepare_large_category_features()

        print(f"  Embedding shape: {self.X_train.shape}")
        print(f"  Val shape: {self.X_val.shape}")
        if self.X_missing is not None:
            print(f"  Missing shape: {self.X_missing.shape}")
        if self.cat_feature_indices:
            print(f"  Cat features: {self.cat_feature_indices}")

    def _get_large_categories(self, df: pd.DataFrame) -> np.ndarray:
        """Extract large category column, filling NaN with 'unknown'."""
        return df["diner_category_large"].fillna("unknown").values

    def _prepare_large_category_features(self) -> None:
        """Store large category arrays for use as native categorical features."""
        print("Preparing large category feature...")

        self._cat_train = self._get_large_categories(self.data.df_train)
        self._cat_val = self._get_large_categories(self.data.df_val)

        if self.X_missing is not None and len(self.data.df_missing) > 0:
            self._cat_missing = self._get_large_categories(self.data.df_missing)

        self.cat_feature_indices = [self.X_train.shape[1]]
        unique_cats = sorted(set(self._cat_train))
        print(
            f"  Large category feature at index {self.cat_feature_indices[0]}"
            f" ({len(unique_cats)} classes: {unique_cats})"
        )

    @abstractmethod
    def build_model(self) -> None:
        """Build the classification model. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def train_model(self) -> None:
        """Train the model. Must be implemented by subclasses."""
        raise NotImplementedError

    def evaluate(
        self, X, y_true, set_name: str = "Validation", df: pd.DataFrame = None
    ) -> ClassificationResult:
        """Evaluate model on given data with optional hierarchy constraint."""
        print(f"\nEvaluating on {set_name} set...")

        if df is not None and hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X)
            y_pred, _ = self._constrained_predict(
                y_proba, large_categories=df["diner_category_large"].values
            )
        else:
            y_pred = self.model.predict(X)

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

        # Print top classes
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

        return ClassificationResult(
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            report=report,
            predictions=y_pred,
        )

    def predict_missing(self) -> pd.DataFrame:
        """Predict missing category values with hierarchy constraint."""
        if self.X_missing is None or len(self.data.df_missing) == 0:
            print("No missing data to predict.")
            return pd.DataFrame()

        print(f"\nPredicting {len(self.data.df_missing):,} missing values...")

        # Get probabilities for constrained prediction
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(self.X_missing)
            y_pred, y_conf = self._constrained_predict(y_proba)
            y_pred_labels = self.data.label_encoder.inverse_transform(y_pred)
        else:
            # Fallback to unconstrained prediction
            y_pred = self.model.predict(self.X_missing)
            y_pred_labels = self.data.label_encoder.inverse_transform(y_pred)
            y_conf = None

        result = self.data.df_missing[["diner_idx"]].copy()
        result["predicted_middle_category"] = y_pred_labels

        if y_conf is not None:
            result["prediction_confidence"] = y_conf

        # Print distribution
        print("\nPrediction distribution (top 10):")
        pred_dist = result["predicted_middle_category"].value_counts().head(10)
        for cat, count in pred_dist.items():
            print(f"  {cat}: {count:,}")

        return result

    def _constrained_predict(
        self, y_proba: np.ndarray, large_categories: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply hierarchy constraint to predictions.

        For each sample, mask out middle categories that don't belong to
        the sample's large category, then select the highest probability
        valid category.

        Args:
            y_proba: Probability matrix from model.
            large_categories: Array of large category values. If None,
                uses df_missing large categories.
        """
        print("Applying hierarchy constraint...")

        # Build mapping: middle_category_idx -> valid for which large_categories
        classes = self.data.label_encoder.classes_
        hierarchy = self.data.hierarchy

        # Reverse mapping: middle -> large
        middle_to_large = {}
        for large, middles in hierarchy.items():
            for middle in middles:
                middle_to_large[middle] = large

        if large_categories is None:
            large_categories = self.data.df_missing["diner_category_large"].values

        y_pred = np.zeros(len(y_proba), dtype=int)
        y_conf = np.zeros(len(y_proba))

        constrained_count = 0
        unconstrained_count = 0

        for i, (proba, large_cat) in enumerate(zip(y_proba, large_categories)):
            if pd.isna(large_cat) or large_cat not in hierarchy:
                # No constraint - use max probability
                y_pred[i] = np.argmax(proba)
                y_conf[i] = proba[y_pred[i]]
                unconstrained_count += 1
            else:
                # Apply constraint - mask invalid categories
                valid_middles = hierarchy[large_cat]
                valid_indices = [
                    idx for idx, cls in enumerate(classes) if cls in valid_middles
                ]

                if valid_indices:
                    # Select best among valid categories
                    masked_proba = np.full_like(proba, -np.inf)
                    masked_proba[valid_indices] = proba[valid_indices]
                    y_pred[i] = np.argmax(masked_proba)
                    y_conf[i] = proba[y_pred[i]]
                    constrained_count += 1
                else:
                    # No valid categories found - fallback to unconstrained
                    y_pred[i] = np.argmax(proba)
                    y_conf[i] = proba[y_pred[i]]
                    unconstrained_count += 1

        print(f"  Constrained predictions: {constrained_count:,}")
        print(f"  Unconstrained predictions: {unconstrained_count:,}")

        return y_pred, y_conf

    def save_results(
        self,
        val_result: ClassificationResult,
        predictions: pd.DataFrame = None,
    ) -> None:
        """Save model, metrics, and predictions."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = self.output_dir

        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved: {model_path}")

        # Save label encoder
        le_path = model_dir / "label_encoder.pkl"
        with open(le_path, "wb") as f:
            pickle.dump(self.data.label_encoder, f)

        # Save hierarchy mapping
        hierarchy_path = model_dir / "hierarchy.pkl"
        with open(hierarchy_path, "wb") as f:
            pickle.dump(self.data.hierarchy, f)
        print(f"Hierarchy saved: {hierarchy_path}")

        # Save metrics
        metrics = {
            "accuracy": val_result.accuracy,
            "f1_macro": val_result.f1_macro,
            "f1_weighted": val_result.f1_weighted,
            "num_classes": len(self.data.label_encoder.classes_),
            "embedder": self.embedder.name,
            "classifier": self.name,
        }
        metrics_path = model_dir / "metrics.pkl"
        with open(metrics_path, "wb") as f:
            pickle.dump(metrics, f)

        # Save predictions
        if predictions is not None and len(predictions) > 0:
            pred_path = model_dir / "predictions.csv"
            predictions.to_csv(pred_path, index=False)
            print(f"Predictions saved: {pred_path}")

            # Create filled category file
            filled_path = model_dir / "diner_category_filled.csv"
            category_filled = self.data.category_df.copy()
            pred_dict = dict(
                zip(predictions["diner_idx"], predictions["predicted_middle_category"])
            )
            mask = category_filled["diner_category_middle"].isna()
            category_filled.loc[mask, "diner_category_middle"] = category_filled.loc[
                mask, "diner_idx"
            ].map(pred_dict)
            category_filled.to_csv(filled_path, index=False)
            print(f"Filled categories saved: {filled_path}")

    def _setup_logging(self) -> None:
        """Setup logging to both console and file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / "train.log"

        # Create a tee-like stdout that writes to both console and file
        class TeeOutput:
            def __init__(self, file_path):
                self.terminal = sys.stdout
                self.log = open(file_path, "w")

            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
                self.log.flush()

            def flush(self):
                self.terminal.flush()
                self.log.flush()

            def close(self):
                self.log.close()

        self._tee = TeeOutput(log_path)
        sys.stdout = self._tee
        print(f"Logging to: {log_path}")

    def _cleanup_logging(self) -> None:
        """Restore stdout and close log file."""
        if hasattr(self, "_tee"):
            sys.stdout = self._tee.terminal
            self._tee.close()

    def run(
        self,
        force_reload_data: bool = False,
        force_recompute_embeddings: bool = False,
    ) -> ClassificationResult:
        """
        Run the full training pipeline.

        Args:
            force_reload_data: Reload data from scratch.
            force_recompute_embeddings: Recompute embeddings from scratch.

        Returns:
            Validation results.
        """
        # Setup logging to file
        self._setup_logging()

        print(f"\n{'=' * 60}")
        print(f"Running {self.name} classifier")
        print(f"{'=' * 60}")

        # Step 1: Load data
        self.load_data(force_reload=force_reload_data)

        # Step 2: Prepare embeddings
        self.prepare_embeddings(force_recompute=force_recompute_embeddings)

        # Step 3: Build model
        self.build_model()

        # Step 4: Train model
        self.train_model()

        # Step 5: Evaluate on validation
        val_result = self.evaluate(
            self.X_val, self.data.y_val, "Validation", df=self.data.df_val
        )

        # Step 6: Predict missing
        predictions = self.predict_missing()

        # Step 7: Save results
        self.save_results(val_result, predictions)

        print(f"\n{'=' * 60}")
        print(f"{self.name} training complete!")
        print(f"{'=' * 60}")

        # Cleanup logging
        self._cleanup_logging()

        return val_result
