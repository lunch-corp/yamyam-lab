"""Base embedder interface for text embeddings."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import scipy.sparse as sp


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedders.

    Embedders convert tokenized text into numerical representations.
    Supports caching of fitted embedders and precomputed embeddings.
    """

    def __init__(self, cache_dir: str = "data/processed_category"):
        self.cache_dir = Path(cache_dir)
        self._is_fitted = False

    @property
    def name(self) -> str:
        """Return embedder name for caching."""
        return self.__class__.__name__

    @abstractmethod
    def fit(self, texts: list[str]) -> "BaseEmbedder":
        """
        Fit the embedder on training texts.

        Args:
            texts: List of tokenized texts.

        Returns:
            self for chaining.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, texts: list[str]) -> Union[np.ndarray, sp.spmatrix]:
        """
        Transform texts to embeddings.

        Args:
            texts: List of tokenized texts.

        Returns:
            Embedding matrix (dense or sparse).
        """
        raise NotImplementedError

    def fit_transform(self, texts: list[str]) -> Union[np.ndarray, sp.spmatrix]:
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        raise NotImplementedError

    def save(self, path: Union[str, Path] = None) -> Path:
        """
        Save fitted embedder to disk.

        Args:
            path: Save path. If None, uses cache_dir/name.pkl

        Returns:
            Path where embedder was saved.
        """
        if path is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / f"{self.name}.pkl"
        else:
            path = Path(path)

        with open(path, "wb") as f:
            pickle.dump(self, f)

        print(f"Embedder saved: {path}")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseEmbedder":
        """
        Load embedder from disk.

        Args:
            path: Path to saved embedder.

        Returns:
            Loaded embedder instance.
        """
        with open(path, "rb") as f:
            embedder = pickle.load(f)

        print(f"Embedder loaded: {path}")
        return embedder

    def save_embeddings(
        self,
        embeddings: Union[np.ndarray, sp.spmatrix],
        name: str,
    ) -> Path:
        """
        Save precomputed embeddings to disk.

        Args:
            embeddings: Embedding matrix to save.
            name: Name for the cache file.

        Returns:
            Path where embeddings were saved.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{self.name}_{name}.pkl"

        with open(path, "wb") as f:
            pickle.dump(embeddings, f)

        print(f"Embeddings saved: {path}")
        return path

    def load_embeddings(self, name: str) -> Union[np.ndarray, sp.spmatrix, None]:
        """
        Load precomputed embeddings from disk.

        Args:
            name: Name of the cache file.

        Returns:
            Embedding matrix or None if not found.
        """
        path = self.cache_dir / f"{self.name}_{name}.pkl"

        if not path.exists():
            return None

        with open(path, "rb") as f:
            embeddings = pickle.load(f)

        print(f"Embeddings loaded: {path}")
        return embeddings
