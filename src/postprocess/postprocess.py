import re
import numpy as np
import pandas as pd

from tools.utils import haversine

from typing import Dict, Iterable, List, Optional, Tuple


class ReRankerUtils:
    @staticmethod
    def extract_region_label(addr: str) -> str:
        """
        Extract a region label (e.g., '서울시 강남구') from a road address string.

        Args:
            addr (str): Address string.

        Returns:
            str: Region label extracted from the address.
                - If the second token ends with '구', returns first two tokens joined.
                - If matches '군', '구', '시', returns the matched group.
                - Otherwise returns the first token.
                - Returns 'unknown' if input is invalid or empty.
        """
        if not isinstance(addr, str) or not addr:
            return "unknown"

        parts = addr.split()
        if len(parts) >= 2 and parts[1].endswith("구"):
            return " ".join(parts[:2])

        m = re.match(r"^(\S+)\s+(\S+구|\S+군|\S+시)", addr)
        return m.group(0) if m else parts[0]
    

    @staticmethod
    def minmax(x: np.ndarray) -> np.ndarray:
        """
        Apply min-max normalization to an array.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Normalized array scaled to [0, 1].
                        Returns zeros if input has no variation.
        """
        x = x.astype(np.float32, copy=False)
        mn, mx = float(np.min(x)), float(np.max(x))
        rng = mx - mn
        return (x - mn) / (rng + 1e-8) if rng > 0 else np.zeros_like(x, dtype=np.float32)

    @staticmethod
    def validate_and_clip_k(item_ids: np.ndarray, base_scores: np.ndarray, k: int) -> int:
        """
        Validate and clip k to be within the valid range of candidate items.

        Args:
            item_ids (np.ndarray): Candidate item IDs.
            base_scores (np.ndarray): Base scores corresponding to item_ids.
            k (int): Requested number of items.

        Returns:
            int: Clipped value of k (0 if no items or invalid input).
        """
        if len(item_ids) != len(base_scores):
            raise ValueError("item_ids/base_scores length mismatch")

        L = len(item_ids)
        if L == 0 or k <= 0:
            return 0
        return min(k, L)


class BaseReranker:
    """Abstract base class for reranking strategies."""

    def __init__(
        self,
        lambda_div: float = 0.55,
        w_cat: float = 0.5,
        w_geo: float = 0.5,
        geo_tau_km: float = 2.0,
        coverage_min: Optional[Dict[str, int]] = None,
        coverage_max: Optional[Dict[str, int]] = None,
        region_of: Optional[Dict[int, str]] = None,
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

    def rerank(
        self,
        item_ids: np.ndarray,
        base_scores: np.ndarray,
        item_meta: pd.DataFrame,
        k: int,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class MostPopularReranker(BaseReranker):
    """Rerank most popular items with diversity (category + geo) using MMR."""
    @staticmethod
    def prepare_meta(item_meta: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int]]:
        """
        Prepare metadata for items and build an index mapping.

        Args:
            item_meta (pd.DataFrame): DataFrame containing diner metadata.
                                    Required columns: diner_idx, diner_category_large,
                                    diner_lat, diner_lon.

        Returns:
            Tuple[pd.DataFrame, Dict[int, int]]:
                - Cleaned metadata DataFrame (unique diner_idx).
                - Mapping from diner_idx to row index.
        """
        required = {"diner_idx", "diner_category_large", "diner_lat", "diner_lon"}
        if not required.issubset(item_meta.columns):
            missing = required - set(item_meta.columns)
            raise ValueError(f"item_meta missing columns: {missing}")

        meta = (
            item_meta[list(required)]
            .drop_duplicates("diner_idx")
            .reset_index(drop=True)
        )
        id2row = pd.Series(meta.index, index=meta["diner_idx"].astype(int)).to_dict()
        return meta, id2row

    @staticmethod
    def filter_candidates_by_meta_and_topm(
        item_ids: np.ndarray,
        rel: np.ndarray,
        id2row: Dict[int, int],
        top_m: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter candidate items by available metadata and optionally keep only top-M by relevance.

        Args:
            item_ids (np.ndarray): Candidate item IDs.
            rel (np.ndarray): Relevance scores for items.
            id2row (Dict[int, int]): Mapping from diner_idx to row index.
            top_m (Optional[int]): If set, keep only top-M items by relevance.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Filtered item IDs.
                - Corresponding relevance scores.
                - Row indices in metadata DataFrame.
        """
        mask = np.isin(item_ids, list(id2row.keys()))
        item_ids, rel = item_ids[mask], rel[mask]

        if item_ids.size == 0:
            return item_ids, rel, np.array([], dtype=int)

        if top_m and top_m < item_ids.size:
            top_idx = np.argpartition(-rel, kth=top_m - 1)[:top_m]
            item_ids, rel = item_ids[top_idx], rel[top_idx]

        rows = np.fromiter((id2row[int(cid)] for cid in item_ids), dtype=int)
        return item_ids, rel, rows

    @staticmethod
    def encode_categories(meta_sorted: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
        """
        Encode categorical labels of diners into numeric codes.

        Args:
            meta_sorted (pd.DataFrame): Metadata DataFrame sorted to match candidates.

        Returns:
            Tuple[np.ndarray, pd.Series]:
                - Category codes as integers.
                - Original category labels as pandas Series.
        """
        cats = meta_sorted["diner_category_large"].astype("category")
        return cats.cat.codes.to_numpy(np.int32), cats

    @staticmethod
    def build_coverage_labels(
        item_ids: np.ndarray,
        cats: pd.Series,
        region_of: Optional[Dict[int, str]] = None,
    ) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
        """
        Build coverage labels for items (category + region) and index mapping.

        Args:
            item_ids (np.ndarray): Candidate item IDs.
            cats (pd.Series): Category labels for items.
            region_of (Optional[Dict[int, str]]): Mapping from item_id to region label.

        Returns:
            Tuple[List[List[str]], Dict[str, np.ndarray]]:
                - List of labels per item (category + region).
                - Dictionary mapping label -> array of indices having that label.
        """
        region_of = region_of or {}
        categories_str = cats.astype(str).to_numpy()

        lab_cat = np.array([f"diner_category_large:{c}" for c in categories_str], dtype=object)
        lab_reg = np.array(
            [f"diner_road_address:{region_of.get(int(cid), 'unknown')}" for cid in item_ids],
            dtype=object,
        )

        labels_by_idx = [[lab_cat[i], lab_reg[i]] for i in range(item_ids.size)]

        label_to_indices: Dict[str, np.ndarray] = {}
        for labs in (lab_cat, lab_reg):
            for lab in np.unique(labs):
                label_to_indices[lab] = np.flatnonzero(labs == lab)

        return labels_by_idx, label_to_indices

    @staticmethod
    def apply_coverage_max_candonly(
        cand_idx: np.ndarray,
        cov_counts: Dict[str, int],
        coverage_max: Dict[str, int],
        label_to_indices: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Apply coverage_max constraint to filter out candidates exceeding maximum quota.

        Args:
            cand_idx (np.ndarray): Indices of candidate items.
            cov_counts (Dict[str, int]): Current coverage counts per label.
            coverage_max (Dict[str, int]): Maximum allowed counts per label.
            label_to_indices (Dict[str, np.ndarray]): Label-to-indices mapping.

        Returns:
            np.ndarray: Filtered candidate indices.
        """
        if not coverage_max or cand_idx.size == 0:
            return cand_idx

        cand_mask = np.ones(cand_idx.size, dtype=bool)
        idx_in_cand = {int(i): pos for pos, i in enumerate(cand_idx)}

        for lab, mx in coverage_max.items():
            if mx is None or cov_counts.get(lab, 0) < mx:
                continue
            idxs = label_to_indices.get(lab)
            if idxs is None or idxs.size == 0:
                continue
            for i in idxs:
                pos = idx_in_cand.get(int(i))
                if pos is not None:
                    cand_mask[pos] = False

        return cand_idx[cand_mask]


    @staticmethod
    def coverage_min_bonus_candonly(
        cand_idx: np.ndarray,
        cov_counts: Dict[str, int],
        coverage_min: Dict[str, int],
        label_to_indices: Dict[str, np.ndarray],
        step: float = 0.05,
    ) -> np.ndarray:
        """
        Compute bonus scores for candidates based on coverage_min deficits.

        Args:
            cand_idx (np.ndarray): Indices of candidate items.
            cov_counts (Dict[str, int]): Current coverage counts per label.
            coverage_min (Dict[str, int]): Minimum desired counts per label.
            label_to_indices (Dict[str, np.ndarray]): Label-to-indices mapping.
            step (float): Bonus increment per deficit unit.

        Returns:
            np.ndarray: Bonus values for candidates.
        """
        if not coverage_min or cand_idx.size == 0:
            return np.zeros(cand_idx.size, dtype=np.float32)

        bonus = np.zeros(cand_idx.size, dtype=np.float32)
        idx_in_cand = {int(i): pos for pos, i in enumerate(cand_idx)}

        for lab, mn in coverage_min.items():
            deficit = mn - cov_counts.get(lab, 0)
            if deficit <= 0:
                continue
            idxs = label_to_indices.get(lab)
            if idxs is None or idxs.size == 0:
                continue
            for i in idxs:
                pos = idx_in_cand.get(int(i))
                if pos is not None:
                    bonus[pos] += deficit * step

        return bonus

    @staticmethod
    def geo_similarity_haversine(
        lat_vec_rad: np.ndarray,
        lon_vec_rad: np.ndarray,
        sel_lat_rad: float,
        sel_lon_rad: float,
        tau_km: float,
    ) -> np.ndarray:
        """
        Compute geographic similarity exp(-d/tau) using haversine distance.

        Args:
            lat_vec_rad (np.ndarray): Latitudes of candidates in radians.
            lon_vec_rad (np.ndarray): Longitudes of candidates in radians.
            sel_lat_rad (float): Selected latitude in radians.
            sel_lon_rad (float): Selected longitude in radians.
            tau_km (float): Decay parameter in kilometers.

        Returns:
            np.ndarray: Geographic similarity values in [0, 1].
        """
        lat_deg = np.degrees(lat_vec_rad)
        lon_deg = np.degrees(lon_vec_rad)
        sel_lat_deg, sel_lon_deg = map(np.degrees, [sel_lat_rad, sel_lon_rad])

        d_km = haversine(
            reviewer_lat=float(sel_lat_deg),
            reviewer_lon=float(sel_lon_deg),
            diner_lat=pd.Series(lat_deg),
            diner_lon=pd.Series(lon_deg),
        )

        inv_tau = 1.0 / max(float(tau_km), 1e-6)
        return np.exp(-np.asarray(d_km, dtype=np.float64) * inv_tau).astype(np.float32)

    @staticmethod
    def geo_precompute(meta_sorted: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Precompute geographic values for efficiency.

        Args:
            meta_sorted (pd.DataFrame): Metadata DataFrame containing diner_lat, diner_lon.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Latitudes in radians.
                - Longitudes in radians.
                - Cosine of latitudes.
        """
        lat_rad = np.deg2rad(meta_sorted["diner_lat"].to_numpy(dtype=np.float32))
        lon_rad = np.deg2rad(meta_sorted["diner_lon"].to_numpy(dtype=np.float32))
        cos_lat = np.cos(lat_rad)

        return (
            np.ascontiguousarray(lat_rad, dtype=np.float32),
            np.ascontiguousarray(lon_rad, dtype=np.float32),
            np.ascontiguousarray(cos_lat, dtype=np.float32),
        )

    def rerank(
        self,
        item_ids: np.ndarray,
        base_scores: np.ndarray,
        item_meta: pd.DataFrame,
        k: int,
        popularity_scores: Optional[np.ndarray] = None,
        popularity_weight: float = 0.0,
        normalize_rel: bool = True,
        top_m: Optional[int] = None,
        debug: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:

        k = ReRankerUtils.validate_and_clip_k(item_ids, base_scores, k)
        if k == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        # --- 1. relevance blending ---
        rel = np.asarray(base_scores, dtype=np.float32)
        if normalize_rel:
            rel = ReRankerUtils.minmax(rel)
        if popularity_scores is not None:
            pop = np.asarray(popularity_scores, dtype=np.float32)
            if normalize_rel:
                pop = ReRankerUtils.minmax(pop)
            rel = (1 - popularity_weight) * rel + popularity_weight * pop

        rel_n = ReRankerUtils.minmax(rel)

        # --- 2. prepare metadata ---
        meta, id2row = MostPopularReranker.prepare_meta(item_meta)
        item_ids_f, rel_f, rows = MostPopularReranker.filter_candidates_by_meta_and_topm(
            item_ids, rel_n, id2row, top_m
        )
        if item_ids_f.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        meta_sorted = meta.iloc[rows]
        cat_codes, cats = MostPopularReranker.encode_categories(meta_sorted)
        lat_rad, lon_rad, _ = MostPopularReranker.geo_precompute(meta_sorted)
        labels_by_idx, label_to_indices = MostPopularReranker.build_coverage_labels(item_ids_f, cats, self.region_of)

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
                sim_geo = MostPopularReranker.geo_similarity_haversine(
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

            cand_idx = MostPopularReranker.apply_coverage_max_candonly(
                cand_idx, cov_counts, self.coverage_max, label_to_indices
            )
            if cand_idx.size == 0:
                break

            bonus_c = MostPopularReranker.coverage_min_bonus_candonly(
                cand_idx, cov_counts, self.coverage_min, label_to_indices, step=self.coverage_step
            )

            # normalize similarity
            sim_c = current_max_sim[cand_idx]
            rng = sim_c.max() - sim_c.min()
            sim_c_n = (sim_c - sim_c.min()) / (rng + 1e-8) if rng > 0 else np.zeros_like(sim_c)

            mmr = self.lambda_div * rel_n[cand_idx] - (1.0 - self.lambda_div) * sim_c_n + bonus_c
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
                sim_geo = MostPopularReranker.geo_similarity_haversine(
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
        hotspot_coords: Optional[Iterable[Tuple[float, float]]] = None,
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
        target_region = ReRankerUtils.extract_region_label(self.region_label)
        region_idx = (
            meta.loc[
                meta["diner_road_address"].map(ReRankerUtils.extract_region_label) == target_region,
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
        latlon = meta_idx.reindex(item_ids_g)[["diner_lat", "diner_lon"]].to_numpy(np.float32)
        valid = np.isfinite(latlon).all(axis=1)
        latlon_valid = latlon[valid]

        # --- 3. hotspot centers ---
        if self.hotspot_coords is not None:
            centers = np.asarray(list(self.hotspot_coords), dtype=np.float32)
        elif self.n_auto_hotspots > 0 and latlon_valid.shape[0] >= 2:
            from sklearn.cluster import KMeans

            n_clusters = min(self.n_auto_hotspots, latlon_valid.shape[0])
            km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
            centers = km.fit(latlon_valid).cluster_centers_.astype(np.float32, copy=False)
        else:
            centers = np.empty((0, 2), dtype=np.float32)

        # --- 4. periphery bonus ---
        periphery_bonus = np.zeros_like(base_scores_g, dtype=np.float32)
        if centers.size and latlon_valid.shape[0] and self.periphery_strength > 0 and self.periphery_cap > 0:
            diner_lat, diner_lon = pd.Series(latlon_valid[:, 0]), pd.Series(latlon_valid[:, 1])
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
                bonus_valid = np.clip(self.periphery_strength * ReRankerUtils.minmax(dmin), 0.0, self.periphery_cap)
                periphery_bonus[valid] = bonus_valid

        boosted_scores = base_scores_g + periphery_bonus

        # --- 5. delegate to MostPopularReranker ---
        return super().rerank(item_ids=item_ids_g, base_scores=boosted_scores, item_meta=meta, k=k)
