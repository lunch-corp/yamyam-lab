from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd


class BaseReranker(ABC):
    """Abstract base class for reranking strategies."""

    def __init__(
        self,
        lambda_div: float = 0.55,
        w_cat: float = 0.5,
        w_geo: float = 0.5,
        geo_tau_km: float = 2.0,
        coverage_min: dict[str, int] | None = None,
        coverage_max: dict[str, int] | None = None,
        region_of: dict[int, str] | None = None,
        prefix_freeze: int = 0,
        coverage_step: float = 0.05,
    ) -> None:
        self.lambda_div = float(np.clip(lambda_div, 0.0, 1.0))
        self.w_cat = w_cat
        self.w_geo = w_geo
        self.geo_tau_km = geo_tau_km
        self.coverage_min = coverage_min or {}
        self.coverage_max = coverage_max or {}
        self.region_of = region_of
        self.prefix_freeze = prefix_freeze
        self.coverage_step = coverage_step

    @abstractmethod
    def rerank(
        self,
        item_ids: np.ndarray,
        base_scores: np.ndarray,
        item_meta: pd.DataFrame,
        k: int,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
