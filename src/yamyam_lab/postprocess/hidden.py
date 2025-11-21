import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from yamyam_lab.inference import haversine
from yamyam_lab.postprocess.base import BaseReranker
from yamyam_lab.tools.rerank import (
    apply_coverage_max_candonly,
    build_coverage_labels,
    coverage_min_bonus_candonly,
    encode_categories,
    extract_region_label,
    filter_candidates_by_meta_and_topm,
    geo_precompute,
    geo_similarity_haversine,
    minmax,
    prepare_meta,
    validate_and_clip_k,
)

logger = logging.getLogger(__name__)

# Constants
EMPTY_RESULT = (np.array([], dtype=int), np.array([], dtype=float))
EMPTY_DF = pd.DataFrame()
DEBUG_STEPS = (0, 1)


class Reranker(BaseReranker):
    """Reranker using MMR for diversity."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def rerank(
        self,
        item_ids: np.ndarray,
        base_scores: np.ndarray,
        item_meta: pd.DataFrame,
        k: int,
        popularity_scores: np.ndarray | None = None,
        popularity_weight: float = 0.0,
        normalize_rel: bool = True,
        top_m: int | None = None,
        debug: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        k = validate_and_clip_k(item_ids, base_scores, k)
        if k == 0:
            return EMPTY_RESULT

        # Relevance blending
        rel = np.asarray(base_scores, dtype=np.float32)
        if normalize_rel:
            rel = minmax(rel)
        if popularity_scores is not None:
            pop = np.asarray(popularity_scores, dtype=np.float32)
            if normalize_rel:
                pop = minmax(pop)
            rel = (1 - popularity_weight) * rel + popularity_weight * pop

        normalized_relevance = minmax(rel)

        # Prepare metadata
        meta, id2row = prepare_meta(item_meta)
        item_ids_f, rel_f, rows = filter_candidates_by_meta_and_topm(
            item_ids, normalized_relevance, id2row, top_m
        )
        if item_ids_f.size == 0:
            return EMPTY_RESULT

        meta_sorted = meta.iloc[rows]
        cat_codes, cats = encode_categories(meta_sorted)
        lat_rad, lon_rad, _ = geo_precompute(meta_sorted)
        labels_by_idx, label_to_indices = build_coverage_labels(
            item_ids_f, cats, self.region_of
        )

        # MMR reranking
        return self._perform_mmr_rerank(
            item_ids_f,
            normalized_relevance,
            cat_codes,
            lat_rad,
            lon_rad,
            labels_by_idx,
            label_to_indices,
            k,
            debug,
        )

    def _perform_mmr_rerank(
        self,
        item_ids_f: np.ndarray,
        normalized_relevance: np.ndarray,
        cat_codes: np.ndarray,
        lat_rad: np.ndarray,
        lon_rad: np.ndarray,
        labels_by_idx: list[list[str]],
        label_to_indices: dict[str, np.ndarray],
        k: int,
        debug: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Prefix freeze
        T = min(max(0, self.prefix_freeze), k, item_ids_f.size)
        frozen_ids = item_ids_f[:T].astype(int, copy=False)

        size = item_ids_f.size
        alive = np.ones(size, dtype=bool)
        alive[:T] = False

        # Coverage counts
        cov_counts: dict[str, int] = {}
        for i in range(T):
            for lab in labels_by_idx[i]:
                cov_counts[lab] = cov_counts.get(lab, 0) + 1

        # Initialize similarities
        current_max_sim = self._initialize_similarities(
            T, alive, size, cat_codes, lat_rad, lon_rad
        )

        # Greedy MMR loop
        chosen_ids = self._greedy_mmr_loop(
            T,
            frozen_ids,
            k,
            alive,
            item_ids_f,
            cov_counts,
            label_to_indices,
            normalized_relevance,
            current_max_sim,
            cat_codes,
            lat_rad,
            lon_rad,
            labels_by_idx,
            debug,
        )

        return np.array(chosen_ids[:k], dtype=int), np.array([], dtype=float)

    def _initialize_similarities(
        self,
        T: int,
        alive: np.ndarray,
        size: int,
        cat_codes: np.ndarray,
        lat_rad: np.ndarray,
        lon_rad: np.ndarray,
    ) -> np.ndarray:
        current_max_sim = np.zeros(size, dtype=np.float32)
        if T > 0 and alive.any():
            aidx = np.flatnonzero(alive)
            for sel in range(T):
                sel_code, sel_lat, sel_lon = (
                    cat_codes[sel],
                    float(lat_rad[sel]),
                    float(lon_rad[sel]),
                )
                sim_cat = (cat_codes[aidx] == sel_code).astype(np.float32)
                sim_geo = geo_similarity_haversine(
                    lat_rad[aidx], lon_rad[aidx], sel_lat, sel_lon, self.geo_tau_km
                )
                np.maximum(
                    current_max_sim[aidx],
                    self.w_cat * sim_cat + self.w_geo * sim_geo,
                    out=current_max_sim[aidx],
                )
        return current_max_sim

    def _greedy_mmr_loop(
        self,
        T: int,
        frozen_ids: np.ndarray,
        k: int,
        alive: np.ndarray,
        item_ids_f: np.ndarray,
        cov_counts: dict[str, int],
        label_to_indices: dict[str, np.ndarray],
        normalized_relevance: np.ndarray,
        current_max_sim: np.ndarray,
        cat_codes: np.ndarray,
        lat_rad: np.ndarray,
        lon_rad: np.ndarray,
        labels_by_idx: list[list[str]],
        debug: bool,
    ) -> list[int]:
        chosen_ids: list[int] = [] if T == 0 else list(frozen_ids)
        step = 0
        while len(chosen_ids) < k and alive.any():
            cand_idx = np.flatnonzero(alive)
            if cand_idx.size == 0:
                break

            cand_idx = apply_coverage_max_candonly(
                cand_idx, cov_counts, self.coverage_max, label_to_indices
            )
            if cand_idx.size == 0:
                break

            bonus_c = coverage_min_bonus_candonly(
                cand_idx,
                cov_counts,
                self.coverage_min,
                label_to_indices,
                step=self.coverage_step,
            )

            # Normalize similarity
            sim_c = current_max_sim[cand_idx]
            rng = sim_c.max() - sim_c.min()
            normalized_similarity = (
                (sim_c - sim_c.min()) / (rng + 1e-8)
                if rng > 0
                else np.zeros_like(sim_c)
            )

            mmr = (
                self.lambda_div * normalized_relevance[cand_idx]
                - (1.0 - self.lambda_div) * normalized_similarity
                + bonus_c
            )
            best_idx = int(cand_idx[np.argmax(mmr)])

            chosen_ids.append(int(item_ids_f[best_idx]))
            alive[best_idx] = False

            # Update coverage
            for lab in labels_by_idx[best_idx]:
                cov_counts[lab] = cov_counts.get(lab, 0) + 1

            # Update similarities
            self._update_similarities(
                best_idx, alive, current_max_sim, cat_codes, lat_rad, lon_rad
            )

            if debug and step in DEBUG_STEPS:
                ci = np.flatnonzero(alive)
                if ci.size:
                    logger.debug(
                        f"[step {step}] penalty.std={current_max_sim[ci].std():.5f}"
                    )
            step += 1

        return chosen_ids


class HiddenReranker(Reranker):
    """Rerank to find hidden gem restaurants by favoring periphery locations."""

    def __init__(
        self,
        hotspot_coords: Iterable[tuple[float, float]] | None = None,
        n_auto_hotspots: int = 10,
        periphery_strength: float = 0.5,
        periphery_cap: float = 0.5,
        rating_weight: float = 0.2,
        recent_weight: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hotspot_coords = hotspot_coords
        self.n_auto_hotspots = n_auto_hotspots
        self.periphery_strength = periphery_strength
        self.periphery_cap = periphery_cap
        self.rating_weight = rating_weight
        self.recent_weight = recent_weight

    def rerank(
        self,
        df: pd.DataFrame,
        k: int,
        **kwargs,
    ) -> pd.DataFrame:
        # Extract region
        df = df.assign(region=df["diner_road_address"].map(extract_region_label))

        # Remove duplicates
        df_filtered = df.drop_duplicates(subset="diner_idx")
        df_filtered = df_filtered.reset_index(drop=True)

        item_ids = df_filtered["diner_idx"].to_numpy(dtype=np.int64)
        base_scores = df_filtered["bayesian_score"].to_numpy(dtype=np.float32)
        meta = df_filtered

        if item_ids.size == 0 or k <= 0:
            return EMPTY_DF

        # Hotspot centers by region
        centers = self._compute_hotspots(df_filtered)

        # Bonuses
        boosted_scores, periphery_bonus = self._apply_bonuses(
            base_scores, df_filtered, centers, meta
        )

        # Delegate to MMR reranker
        reranked_ids, _ = super().rerank(
            item_ids=item_ids, base_scores=boosted_scores, item_meta=meta, k=k
        )

        # Return DataFrame
        df_filtered = df_filtered.assign(
            periphery_bonus=periphery_bonus, hidden_score=boosted_scores
        )
        df_reranked = (
            df_filtered.set_index("diner_idx").reindex(reranked_ids).reset_index()
        )
        return df_reranked

    def _compute_hotspots(self, df_filtered: pd.DataFrame) -> dict[str, np.ndarray]:
        centers = {}
        for region, group in df_filtered.groupby("region"):
            latlon = group[["diner_lat", "diner_lon"]].to_numpy()
            valid = np.isfinite(latlon).all(axis=1)
            latlon_valid = latlon[valid]
            if latlon_valid.shape[0] >= 2:
                n_clusters = min(self.n_auto_hotspots, latlon_valid.shape[0])
                km = KMeans(
                    n_clusters=n_clusters, n_init=self.n_auto_hotspots, random_state=42
                )
                centers[region] = km.fit(latlon_valid).cluster_centers_
        return centers

    def _apply_bonuses(
        self,
        base_scores: np.ndarray,
        df_filtered: pd.DataFrame,
        centers: dict[str, np.ndarray],
        meta: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        boosted_scores = base_scores.copy()
        periphery_bonus = np.zeros_like(base_scores, dtype=np.float32)

        # Periphery bonus (vectorized by region)
        for region, group in df_filtered.groupby("region"):
            if region in centers:
                idx = group.index
                latlon = group[["diner_lat", "diner_lon"]].values
                centers_arr = centers[region]
                # Vectorized distance calculation
                dists = np.array(
                    [
                        [haversine(c[0], c[1], lat, lon) for c in centers_arr]
                        for lat, lon in latlon
                    ]
                )
                dmin = np.min(dists, axis=1)
                normalized_dmin = minmax(dmin)
                periphery_bonus[idx] = np.clip(
                    self.periphery_strength * normalized_dmin, 0.0, self.periphery_cap
                )
        boosted_scores += periphery_bonus

        # Rating bonus
        rating_bonus = np.zeros_like(base_scores, dtype=np.float32)
        if "avg_rating" in meta.columns and self.rating_weight > 0:
            ratings = meta["avg_rating"].to_numpy(np.float32)
            rating_bonus = self.rating_weight * minmax(ratings)
        boosted_scores += rating_bonus

        # Recent review bonus
        recent_bonus = np.zeros_like(base_scores, dtype=np.float32)
        if "recent_review_count" in meta.columns and self.recent_weight > 0:
            recent_counts = meta["recent_review_count"].to_numpy(np.float32)
            recent_bonus = self.recent_weight * minmax(recent_counts)
        boosted_scores += recent_bonus

        return boosted_scores, periphery_bonus

    def _update_similarities(
        self,
        best_idx: int,
        alive: np.ndarray,
        current_max_sim: np.ndarray,
        cat_codes: np.ndarray,
        lat_rad: np.ndarray,
        lon_rad: np.ndarray,
    ) -> None:
        """Update maximum similarities for alive candidates after selecting best_idx."""
        if alive.any():
            sel_code, sel_lat, sel_lon = (
                cat_codes[best_idx],
                float(lat_rad[best_idx]),
                float(lon_rad[best_idx]),
            )
            aidx = np.flatnonzero(alive)
            sim_cat = (cat_codes[aidx] == sel_code).astype(np.float32)
            sim_geo = geo_similarity_haversine(
                lat_rad[aidx], lon_rad[aidx], sel_lat, sel_lon, self.geo_tau_km
            )
            np.maximum(
                current_max_sim[aidx],
                self.w_cat * sim_cat + self.w_geo * sim_geo,
                out=current_max_sim[aidx],
            )
