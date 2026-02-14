"""Text embedding models."""

from yamyam_lab.model.embedding.base_embedder import BaseEmbedder
from yamyam_lab.model.embedding.tfidf_embedder import TfidfEmbedder

__all__ = ["BaseEmbedder", "TfidfEmbedder"]
