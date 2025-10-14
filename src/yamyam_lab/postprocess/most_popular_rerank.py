from abc import abstractmethod
from typing import Dict, Iterable, List, Tuple

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


class MostPopularReranker(BaseReranker):
    """Rerank most popular items with diversity (category + geo) using MMR."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        k = validate_and_clip_k(item_ids, base_scores, k)
        if k == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        # --- 1. relevance blending ---
        rel = np.asarray(base_scores, dtype=np.float32)
        if normalize_rel:
            rel = minmax(rel)
        if popularity_scores is not None:
            pop = np.asarray(popularity_scores, dtype=np.float32)
            if normalize_rel:
                pop = minmax(pop)
            rel = (1 - popularity_weight) * rel + popularity_weight * pop

        rel_n = minmax(rel)

        # --- 2. prepare metadata ---
        meta, id2row = prepare_meta(item_meta)
        item_ids_f, rel_f, rows = filter_candidates_by_meta_and_topm(
            item_ids, rel_n, id2row, top_m
        )
        if item_ids_f.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        meta_sorted = meta.iloc[rows]
        cat_codes, cats = encode_categories(meta_sorted)
        lat_rad, lon_rad, _ = geo_precompute(meta_sorted)
        labels_by_idx, label_to_indices = build_coverage_labels(
            item_ids_f, cats, self.region_of
        )

        # --- 3. prefix freeze ---
        T = min(max(0, self.prefix_freeze), k, item_ids_f.size)
        frozen_ids = item_ids_f[:T].astype(int, copy=False)

        size = item_ids_f.size
        alive = np.ones(size, dtype=bool)
        alive[:T] = False

        # --- 4. coverage counts ---
        cov_counts: Dict[str, int] = {}
        for i in range(T):
            for lab in labels_by_idx[i]:
                cov_counts[lab] = cov_counts.get(lab, 0) + 1

        # --- 5. initialize similarities ---
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

        # --- 6. greedy MMR loop ---
        chosen_ids: List[int] = [] if T == 0 else list(frozen_ids)
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

            # normalize similarity
            sim_c = current_max_sim[cand_idx]
            rng = sim_c.max() - sim_c.min()
            sim_c_n = (
                (sim_c - sim_c.min()) / (rng + 1e-8)
                if rng > 0
                else np.zeros_like(sim_c)
            )

            mmr = (
                self.lambda_div * rel_n[cand_idx]
                - (1.0 - self.lambda_div) * sim_c_n
                + bonus_c
            )
            best_idx = int(cand_idx[np.argmax(mmr)])

            chosen_ids.append(int(item_ids_f[best_idx]))
            alive[best_idx] = False

            # update coverage
            for lab in labels_by_idx[best_idx]:
                cov_counts[lab] = cov_counts.get(lab, 0) + 1

            # update similarities
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

            if debug and step in (0, 1):
                ci = np.flatnonzero(alive)
                if ci.size:
                    print(f"[step {step}] penalty.std={current_max_sim[ci].std():.5f}")
            step += 1

        return np.array(chosen_ids[:k], dtype=int), np.array([], dtype=float)


class RegionPeripheryReranker(MostPopularReranker):
    """Rerank items within a region by adding periphery bonus, then MMR reranking."""

    def __init__(
        self,
        region_label: str = "서울 강남구",
        hotspot_coords: Iterable[tuple[float, float]] | None = None,
        n_auto_hotspots: int = 5,
        periphery_strength: float = 0.5,
        periphery_cap: float = 0.15,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.region_label = region_label
        self.hotspot_coords = hotspot_coords
        self.n_auto_hotspots = n_auto_hotspots
        self.periphery_strength = periphery_strength
        self.periphery_cap = periphery_cap

    def rerank(
        self,
        item_ids: np.ndarray,
        base_scores: np.ndarray,
        item_meta: pd.DataFrame,
        k: int,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        item_ids = np.asarray(item_ids, dtype=np.int64)
        base_scores = np.asarray(base_scores, dtype=np.float32)
        meta = item_meta

        # --- 1. region filter ---
        target_region = extract_region_label(self.region_label)
        region_idx = (
            meta.loc[
                meta["diner_road_address"].map(extract_region_label) == target_region,
                "diner_idx",
            ].to_numpy(np.int64, copy=False)
            if "diner_road_address" in meta.columns
            else np.empty(0, dtype=np.int64)
        )

        if region_idx.size > 0:
            mask = np.isin(item_ids, region_idx)
            item_ids_g, base_scores_g = item_ids[mask], base_scores[mask]

        else:
            item_ids_g, base_scores_g = item_ids, base_scores

        if item_ids_g.size == 0 or k <= 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        # --- 2. coordinates ---
        meta_idx = meta.set_index("diner_idx", drop=False)
        latlon = meta_idx.reindex(item_ids_g)[["diner_lat", "diner_lon"]].to_numpy(
            np.float32
        )
        valid = np.isfinite(latlon).all(axis=1)
        latlon_valid = latlon[valid]

        # --- 3. hotspot centers ---
        if self.hotspot_coords is not None:
            centers = np.asarray(list(self.hotspot_coords), dtype=np.float32)
        elif self.n_auto_hotspots > 0 and latlon_valid.shape[0] >= 2:
            n_clusters = min(self.n_auto_hotspots, latlon_valid.shape[0])
            km = KMeans(n_clusters=n_clusters, n_init=1, random_state=42)
            centers = km.fit(latlon_valid).cluster_centers_.astype(
                np.float32, copy=False
            )
        else:
            centers = np.empty((0, 2), dtype=np.float32)

        # --- 4. periphery bonus ---
        periphery_bonus = np.zeros_like(base_scores_g, dtype=np.float32)
        if (
            centers.size
            and latlon_valid.shape[0]
            and self.periphery_strength > 0
            and self.periphery_cap > 0
        ):
            diner_lat, diner_lon = (
                pd.Series(latlon_valid[:, 0]),
                pd.Series(latlon_valid[:, 1]),
            )
            dists = [
                np.asarray(
                    haversine(
                        reviewer_lat=float(c_lat),
                        reviewer_lon=float(c_lon),
                        diner_lat=diner_lat,
                        diner_lon=diner_lon,
                    ),
                    dtype=np.float64,
                )
                for c_lat, c_lon in centers
            ]
            if dists:
                dmin = np.min(np.vstack(dists).T, axis=1).astype(np.float32)
                bonus_valid = np.clip(
                    self.periphery_strength * minmax(dmin),
                    0.0,
                    self.periphery_cap,
                )
                periphery_bonus[valid] = bonus_valid

        boosted_scores = base_scores_g + periphery_bonus

        # --- 5. delegate to MostPopularReranker ---
        return super().rerank(
            item_ids=item_ids_g, base_scores=boosted_scores, item_meta=meta, k=k
        )
