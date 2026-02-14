"""TF-IDF embedder for text classification."""

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from yamyam_lab.model.embedding.base_embedder import BaseEmbedder


class TfidfEmbedder(BaseEmbedder):
    """
    TF-IDF based text embedder.

    Converts tokenized text to sparse TF-IDF vectors.
    """

    def __init__(
        self,
        max_features: int = 20000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: tuple = (1, 2),
        sublinear_tf: bool = True,
        cache_dir: str = "data/processed_category",
    ):
        super().__init__(cache_dir=cache_dir)

        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
        )

    @property
    def name(self) -> str:
        """Return unique name based on configuration."""
        return (
            f"tfidf_f{self.max_features}_ng{self.ngram_range[0]}{self.ngram_range[1]}"
        )

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        """Fit TF-IDF vectorizer on texts."""
        print(f"Fitting TF-IDF (max_features={self.max_features})...")
        self.vectorizer.fit(texts)
        self._is_fitted = True
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        return self

    def transform(self, texts: list[str]) -> sp.csr_matrix:
        """Transform texts to TF-IDF vectors."""
        if not self._is_fitted:
            raise RuntimeError("Embedder must be fitted before transform.")
        return self.vectorizer.transform(texts)

    def get_embedding_dim(self) -> int:
        """Return TF-IDF feature dimension."""
        if not self._is_fitted:
            return self.max_features
        return len(self.vectorizer.vocabulary_)

    def get_feature_names(self) -> list[str]:
        """Return feature names (vocabulary)."""
        return self.vectorizer.get_feature_names_out().tolist()
