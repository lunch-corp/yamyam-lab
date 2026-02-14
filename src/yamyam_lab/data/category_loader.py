"""Data loader for category classification with caching support."""

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


@dataclass
class CategoryData:
    """Container for category classification data."""

    # Raw dataframes
    category_df: pd.DataFrame
    diner_df: pd.DataFrame
    menu_df: pd.DataFrame

    # Processed data
    df_train: pd.DataFrame = None
    df_val: pd.DataFrame = None
    df_missing: pd.DataFrame = None

    # Labels
    y_train: np.ndarray = None
    y_val: np.ndarray = None
    label_encoder: LabelEncoder = None

    # Tokenized text (cached)
    train_tokenized: list = None
    val_tokenized: list = None
    missing_tokenized: list = None

    # Hierarchy mapping: large_category -> list of valid middle_categories
    hierarchy: dict = None


class CategoryDataLoader:
    """
    Data loader with preprocessing and caching for category classification.

    Supports caching of:
    - Tokenized text (expensive Korean morphological analysis)
    - Aggregated menu data
    """

    def __init__(
        self,
        data_dir: str = "data",
        cache_dir: str = "data/processed_category",
        use_menu: bool = True,
        val_ratio: float = 0.1,
        min_class_samples: int = 10,
        random_state: int = 42,
        sample_size: int = None,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.use_menu = use_menu
        self.val_ratio = val_ratio
        self.min_class_samples = min_class_samples
        self.random_state = random_state
        self.sample_size = sample_size

        self._kiwi = None

    @property
    def kiwi(self) -> Kiwi:
        """Lazy initialization of Kiwi tokenizer."""
        if self._kiwi is None:
            self._kiwi = Kiwi()
        return self._kiwi

    def _get_cache_key(self) -> str:
        """Generate cache key based on configuration."""
        config_str = f"{self.use_menu}_{self.val_ratio}_{self.min_class_samples}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _get_cache_path(self, name: str) -> Path:
        """Get cache file path."""
        cache_key = self._get_cache_key()
        return self.cache_dir / f"{name}_{cache_key}.pkl"

    def _save_cache(self, obj: object, name: str) -> None:
        """Save object to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._get_cache_path(name)
        with open(cache_path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Cached: {cache_path}")

    def _load_cache(self, name: str) -> object:
        """Load object from cache if exists."""
        cache_path = self._get_cache_path(name)
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                print(f"  Loaded from cache: {cache_path}")
                return pickle.load(f)
        return None

    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw CSV files."""
        print("Loading raw data...")

        category_df = pd.read_csv(self.data_dir / "diner_category_processed.csv")
        diner_df = pd.read_csv(self.data_dir / "diner.csv")
        menu_df = pd.read_csv(self.data_dir / "menu_df.csv")

        print(f"  Category: {len(category_df):,} rows")
        print(f"  Diner: {len(diner_df):,} rows")
        print(f"  Menu: {len(menu_df):,} rows")

        return category_df, diner_df, menu_df

    def aggregate_menu_names(self, menu_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate menu names by diner_idx with caching."""
        cached = self._load_cache("menu_agg")
        if cached is not None:
            return cached

        print("Aggregating menu names...")
        menu_agg = (
            menu_df.groupby("diner_idx")["name"]
            .apply(lambda x: " ".join(x.dropna().astype(str)))
            .reset_index()
        )
        menu_agg.columns = ["diner_idx", "menu_text"]

        self._save_cache(menu_agg, "menu_agg")
        return menu_agg

    def prepare_features(
        self,
        category_df: pd.DataFrame,
        diner_df: pd.DataFrame,
        menu_agg: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare combined text features."""
        print("Preparing features...")

        df = category_df.merge(
            diner_df[["diner_idx", "diner_name"]], on="diner_idx", how="left"
        )

        if self.use_menu:
            df = df.merge(menu_agg, on="diner_idx", how="left")
            df["menu_text"] = df["menu_text"].fillna("")
            df["combined_text"] = df["diner_name"].fillna("") + " " + df["menu_text"]
        else:
            df["combined_text"] = df["diner_name"].fillna("")

        return df

    def build_hierarchy(self, df: pd.DataFrame) -> dict:
        """Build hierarchy mapping from large to middle categories."""
        print("Building category hierarchy...")

        filled = df[
            df["diner_category_middle"].notna() & df["diner_category_large"].notna()
        ]

        hierarchy = (
            filled.groupby("diner_category_large")["diner_category_middle"]
            .apply(lambda x: sorted(x.unique().tolist()))
            .to_dict()
        )

        total_middles = sum(len(v) for v in hierarchy.values())
        print(
            f"  {len(hierarchy)} large categories -> {total_middles} middle categories"
        )

        return hierarchy

    def tokenize_korean(self, text: str) -> str:
        """Tokenize Korean text using Kiwi."""
        if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
            return ""
        try:
            tokens = self.kiwi.tokenize(text)
            meaningful_tags = {"NNG", "NNP", "VV", "VA", "MAG", "SL", "SN"}
            words = [token.form for token in tokens if token.tag in meaningful_tags]
            return " ".join(words)
        except Exception:
            return text

    def tokenize_texts(
        self, texts: pd.Series, cache_name: str = None, desc: str = "Tokenizing"
    ) -> list[str]:
        """Tokenize texts with optional caching."""
        if cache_name:
            cached = self._load_cache(cache_name)
            if cached is not None:
                return cached

        tokenized = []
        for text in tqdm(texts, desc=desc):
            tokenized.append(self.tokenize_korean(text))

        if cache_name:
            self._save_cache(tokenized, cache_name)

        return tokenized

    def split_data(
        self, df: pd.DataFrame
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder
    ]:
        """Split data into train, validation, and missing sets."""
        print("Splitting data...")

        df_filled = df[df["diner_category_middle"].notna()].copy()
        df_missing = df[df["diner_category_middle"].isna()].copy()

        print(f"  Filled: {len(df_filled):,}")
        print(f"  Missing: {len(df_missing):,}")

        # Sample data if sample_size is set (for test mode)
        if self.sample_size is not None and len(df_filled) > self.sample_size:
            print(f"  Sampling {self.sample_size:,} rows for test mode...")
            df_filled = df_filled.sample(
                n=self.sample_size, random_state=self.random_state
            )
            df_missing = df_missing.sample(
                n=min(self.sample_size // 5, len(df_missing)),
                random_state=self.random_state,
            )

        # Filter rare classes
        class_counts = df_filled["diner_category_middle"].value_counts()
        valid_classes = class_counts[class_counts >= self.min_class_samples].index
        rare_classes = class_counts[class_counts < self.min_class_samples].index

        if len(rare_classes) > 0:
            print(
                f"  Filtering {len(rare_classes)} rare classes (< {self.min_class_samples} samples)"
            )
            df_filled = df_filled[
                df_filled["diner_category_middle"].isin(valid_classes)
            ].copy()

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df_filled["diner_category_middle"])
        num_classes = len(label_encoder.classes_)
        print(f"  Classes: {num_classes}")

        # Train/val split
        train_idx, val_idx = train_test_split(
            np.arange(len(df_filled)),
            test_size=self.val_ratio,
            random_state=self.random_state,
            stratify=y,
        )

        df_train = df_filled.iloc[train_idx].copy()
        df_val = df_filled.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]

        print(f"  Train: {len(df_train):,}")
        print(f"  Val: {len(df_val):,}")

        return df_train, df_val, df_missing, y_train, y_val, label_encoder

    def load(self, force_reload: bool = False) -> CategoryData:
        """
        Load and preprocess all data.

        Args:
            force_reload: If True, ignore cache and reload everything.

        Returns:
            CategoryData with all preprocessed data.
        """
        if force_reload:
            import shutil

            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)

        # Load raw data
        category_df, diner_df, menu_df = self.load_raw_data()

        # Aggregate menu
        menu_agg = (
            self.aggregate_menu_names(menu_df) if self.use_menu else pd.DataFrame()
        )

        # Prepare features
        df = self.prepare_features(category_df, diner_df, menu_agg)

        # Build hierarchy mapping
        hierarchy = self.build_hierarchy(df)

        # Split data
        df_train, df_val, df_missing, y_train, y_val, label_encoder = self.split_data(
            df
        )

        # Tokenize with caching
        print("Tokenizing texts...")
        train_tokenized = self.tokenize_texts(
            df_train["combined_text"], cache_name="train_tokens", desc="Train"
        )
        val_tokenized = self.tokenize_texts(
            df_val["combined_text"], cache_name="val_tokens", desc="Val"
        )
        missing_tokenized = (
            self.tokenize_texts(
                df_missing["combined_text"], cache_name="missing_tokens", desc="Missing"
            )
            if len(df_missing) > 0
            else []
        )

        return CategoryData(
            category_df=category_df,
            diner_df=diner_df,
            menu_df=menu_df,
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
