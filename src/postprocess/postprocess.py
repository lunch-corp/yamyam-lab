from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from tools.utils import *


def rerank_most_popular_with_diversity(
    item_ids: np.ndarray,
    base_scores: np.ndarray,
    item_meta: pd.DataFrame,
    k: int,
    lambda_div: float = 0.55,
    w_cat: float = 0.5,
    w_geo: float = 0.5,
    geo_tau_km: float = 2.0,
    coverage_min: Optional[Dict[str, int]] = None,
    coverage_max: Optional[Dict[str, int]] = None,
    region_of: Optional[Dict[int, str]] = None,
    popularity_weight: float = 0.0,
    popularity_scores: Optional[np.ndarray] = None,
    normalize_rel: bool = True,
    top_m: Optional[int] = None,
    debug: bool = False,
    prefix_freeze: int = 0,
    coverage_step: float = 0.05,
):
    """
    Re-rank the Most Popular recommendation list with category and geographic diversity
    using a Maximal Marginal Relevance (MMR) approach.

    The final score of a candidate item i is computed as:
        score(i) = λ · rel(i) - (1-λ) · sim_max(i) + bonus(i)
    where:
        - rel(i): normalized relevance (optionally blended with popularity)
        - sim_max(i): maximum similarity to already selected items
          sim(i,j) = w_cat * [cat(i) = cat(j)] + w_geo * exp(-d(i,j)/τ)
          with d(i,j) the haversine distance in km and τ = geo_tau_km
        - bonus(i): additional bonus to encourage coverage_min constraints

    Args:
        item_ids (np.ndarray): Array of candidate item IDs.
        base_scores (np.ndarray): Base relevance scores for each item.
        item_meta (pd.DataFrame): Metadata with required columns:
            ['diner_idx', 'diner_category_large', 'diner_lat', 'diner_lon'].
        k (int): Number of items to select.
        lambda_div (float, optional): Trade-off parameter (λ↑ → accuracy↑, diversity↓).
        w_cat (float, optional): Weight for category similarity.
        w_geo (float, optional): Weight for geographic similarity.
        geo_tau_km (float, optional): Decay parameter (km) for geographic similarity kernel.
        coverage_min (Optional[Dict[str, int]], optional): Minimum coverage requirements per label.
        coverage_max (Optional[Dict[str, int]], optional): Maximum coverage limits per label.
        region_of (Optional[Dict[int, str]], optional): Mapping from item_id to region label.
        popularity_weight (float, optional): Weight for blending relevance with popularity.
        popularity_scores (Optional[np.ndarray], optional): Popularity scores for items.
        normalize_rel (bool, optional): Whether to min-max normalize relevance.
        top_m (Optional[int], optional): Pre-select top-M items by relevance before re-ranking.
        debug (bool, optional): If True, print simple debug info during first steps.
        prefix_freeze (int, optional): Keep top-T original items fixed in the final ranking.
        use_geo_fast (bool, optional): Legacy flag (ignored).
        coverage_step (float, optional): Step size for coverage_min bonus increments.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Selected item IDs (length k).
            - (Currently empty) array for scores, placeholder for extension.
    """

    k = validate_and_clip_k(item_ids, base_scores, k)
    if k == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    base_scores = np.asarray(item_ids, dtype=np.int64) * 0 + np.asarray(
        base_scores, dtype=np.float32
    )  # ensure same length
    rel = base_scores.astype(np.float32, copy=True)
    if normalize_rel:
        rel = minmax(rel)
    if popularity_scores is not None:
        pop = np.asarray(popularity_scores, dtype=np.float32)
        if normalize_rel:
            pop = minmax(pop)
        rel = (1 - popularity_weight) * rel + popularity_weight * pop

    lam = float(np.clip(lambda_div, 0.0, 1.0))
    rel_n = minmax(rel)  # 루프 밖 1회

    meta, id2row = prepare_meta(item_meta)
    item_ids_f, rel_f, rows = filter_candidates_by_meta_and_topm(
        item_ids, rel_n, id2row, top_m
    )
    if item_ids_f.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    meta_sorted = meta.iloc[rows]
    cat_codes, cats = encode_categories(meta_sorted)
    lat_rad, lon_rad, _cos_lat_unused = geo_precompute(meta_sorted)
    labels_by_idx, label_to_indices = build_coverage_labels(item_ids_f, cats, region_of)

    T = int(max(0, min(prefix_freeze, k, item_ids_f.size)))
    frozen_ids = item_ids_f[:T].astype(int, copy=False)
    size = item_ids_f.size
    alive = np.ones(size, dtype=bool)
    alive[:T] = False

    coverage_min = coverage_min or {}
    coverage_max = coverage_max or {}
    cov_counts: Dict[str, int] = {}
    for i in range(T):
        for lab in labels_by_idx[i]:
            cov_counts[lab] = cov_counts.get(lab, 0) + 1

    current_max_sim = np.zeros(size, dtype=np.float32)
    if T > 0 and alive.any():
        aidx = np.flatnonzero(alive)
        for sel in range(T):
            sel_code = cat_codes[sel]
            sel_lat = float(lat_rad[sel])
            sel_lon = float(lon_rad[sel])

            sim_cat = (cat_codes[aidx] == sel_code).astype(np.float32)
            sim_geo = geo_similarity_haversine(
                lat_vec_rad=lat_rad[aidx],
                lon_vec_rad=lon_rad[aidx],
                sel_lat_rad=sel_lat,
                sel_lon_rad=sel_lon,
                tau_km=geo_tau_km,
            )
            combined = (w_cat * sim_cat + w_geo * sim_geo).astype(np.float32)
            np.maximum(current_max_sim[aidx], combined, out=current_max_sim[aidx])

    chosen_ids: List[int] = [] if T == 0 else list(frozen_ids)
    chosen_scores: List[float] = []  # MMR 점수도 기록용

    step = 0
    while len(chosen_ids) < k and alive.any():
        cand_idx = np.flatnonzero(alive)
        if cand_idx.size == 0:
            break

        cand_idx = apply_coverage_max_candonly(
            cand_idx=cand_idx,
            cov_counts=cov_counts,
            coverage_max=coverage_max,
            label_to_indices=label_to_indices,
        )
        if cand_idx.size == 0:
            break

        bonus_c = coverage_min_bonus_candonly(
            cand_idx=cand_idx,
            cov_counts=cov_counts,
            coverage_min=coverage_min,
            label_to_indices=label_to_indices,
            step=coverage_step,
        )

        sim_c = current_max_sim[cand_idx]
        mn, mx = float(sim_c.min()), float(sim_c.max())
        sim_c_n = (
            (sim_c - mn) / (mx - mn + 1e-8)
            if mx > mn
            else np.zeros_like(sim_c, dtype=np.float32)
        )

        mmr = lam * rel_n[cand_idx] - (1.0 - lam) * sim_c_n + bonus_c
        best_local = int(np.argmax(mmr))
        best_idx = int(cand_idx[best_local])

        chosen_ids.append(int(item_ids_f[best_idx]))
        alive[best_idx] = False

        for lab in labels_by_idx[best_idx]:
            cov_counts[lab] = max(0, cov_counts.get(lab, 0)) + 1

        if alive.any():
            sel_code = cat_codes[best_idx]
            sel_lat = float(lat_rad[best_idx])
            sel_lon = float(lon_rad[best_idx])
            aidx = np.flatnonzero(alive)

            sim_cat = (cat_codes[aidx] == sel_code).astype(np.float32)
            sim_geo = geo_similarity_haversine(
                lat_vec_rad=lat_rad[aidx],
                lon_vec_rad=lon_rad[aidx],
                sel_lat_rad=sel_lat,
                sel_lon_rad=sel_lon,
                tau_km=geo_tau_km,
            )
            combined = (w_cat * sim_cat + w_geo * sim_geo).astype(np.float32)
            np.maximum(current_max_sim[aidx], combined, out=current_max_sim[aidx])

        if debug and step in (0, 1):
            ci = np.flatnonzero(alive)
            if ci.size:
                print(f"[step {step}] penalty.std={current_max_sim[ci].std():.5f}")
        step += 1

    return np.array(chosen_ids[:k], dtype=int), np.array([], dtype=float)


def rerank_region_periphery(
    item_ids: np.ndarray,
    base_scores: np.ndarray,
    item_meta_std: pd.DataFrame,
    k: int,
    region_label: str = "서울 강남구",
    hotspot_coords: Optional[Iterable[Tuple[float, float]]] = None,
    n_auto_hotspots: int = 5,
    periphery_strength: float = 0.5,
    periphery_cap: float = 0.15,
    lambda_div: float = 0.55,
    w_cat: float = 0.5,
    w_geo: float = 0.5,
    geo_tau_km: float = 2.0,
    coverage_min: Optional[Dict[str, int]] = None,
    coverage_max: Optional[Dict[str, int]] = None,
    region_of: Optional[Dict[int, str]] = None,
):
    """
    Re-rank items within a target region by adding a periphery bonus (farther from hotspots)
    and then applying MMR-based re-ranking with category/geography diversity.

    The periphery bonus is computed by:
      1) Determining hotspot centers (given or auto via KMeans).
      2) Computing each candidate's minimum haversine distance to the centers.
      3) Min-max normalizing that minimum distance and scaling/clipping to [0, periphery_cap].
      4) Adding the bonus to base_scores before calling MMR re-ranking.

    Args:
        item_ids (np.ndarray): Array of candidate item IDs (ordered by base ranking).
        base_scores (np.ndarray): Base relevance scores aligned with item_ids.
        item_meta_std (pd.DataFrame): Metadata containing at least:
            ['diner_idx', 'diner_lat', 'diner_lon', 'diner_road_address'].
        k (int): Number of items to select.
        region_label (str, optional): Target region label (e.g., "서울 강남구").
        hotspot_coords (Optional[Iterable[Tuple[float, float]]], optional):
            Iterable of (lat, lon) hotspot coordinates in degrees. If None, auto-detected.
        n_auto_hotspots (int, optional): Number of clusters (hotspots) to auto-detect via KMeans when not provided.
        periphery_strength (float, optional): Scale for the periphery bonus before capping.
        periphery_cap (float, optional): Upper cap applied to the periphery bonus.
        lambda_div (float, optional): MMR trade-off between relevance and diversity [0, 1].
        w_cat (float, optional): Weight for category similarity in MMR.
        w_geo (float, optional): Weight for geographic similarity in MMR.
        geo_tau_km (float, optional): Length scale (km) for geographic similarity kernel.
        coverage_min (Optional[Dict[str, int]], optional): Minimum coverage constraints per label.
        coverage_max (Optional[Dict[str, int]], optional): Maximum coverage constraints per label.
        region_of (Optional[Dict[int, str]], optional): Mapping from item_id to region label (used by MMR coverage).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Selected item IDs (length ≤ k).
            - Placeholder scores array (currently empty).
    """
    # 0) 기본 변환
    item_ids = np.asarray(item_ids, dtype=np.int64)
    base_scores = np.asarray(base_scores, dtype=np.float32)
    meta = item_meta_std

    # 1) 지역 필터
    target_region = extract_region_label(region_label)
    if "diner_road_address" in meta.columns:
        region_norm = meta["diner_road_address"].map(extract_region_label)
        region_idx = meta.loc[region_norm == target_region, "diner_idx"].to_numpy(
            dtype=np.int64, copy=False
        )
    else:
        region_idx = np.empty(0, dtype=np.int64)

    if region_idx.size > 0:
        mask = np.isin(item_ids, region_idx, assume_unique=False)
        item_ids_g = item_ids[mask]
        base_scores_g = base_scores[mask]
    else:
        # 필터 결과가 0이면 원본 전체 후보 사용
        item_ids_g = item_ids
        base_scores_g = base_scores

    if item_ids_g.size == 0 or k <= 0:
        return item_ids[:0], base_scores[:0]

    # 2) 좌표 정리
    meta_idx = meta.set_index("diner_idx", drop=False)
    latlon = meta_idx.reindex(item_ids_g)[["diner_lat", "diner_lon"]].to_numpy(
        dtype=np.float32
    )
    valid = np.isfinite(latlon).all(axis=1)
    latlon_valid = latlon[valid]

    # 3) 핫스팟 결정 (주어지지 않으면 KMeans로 자동 산출)
    if hotspot_coords is None and n_auto_hotspots > 0 and latlon_valid.shape[0] >= 2:
        from sklearn.cluster import KMeans

        n_clusters = int(min(n_auto_hotspots, latlon_valid.shape[0]))
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        km.fit(latlon_valid)
        centers = km.cluster_centers_.astype(np.float32, copy=False)  # degrees
    elif hotspot_coords is not None:
        centers = np.asarray(list(hotspot_coords), dtype=np.float32)  # degrees
    else:
        centers = np.empty((0, 2), dtype=np.float32)

    # 4) 변두리 보너스 (정확한 거리: haversine 사용)
    periphery_bonus = np.zeros_like(base_scores_g, dtype=np.float32)
    if (
        centers.size
        and latlon_valid.shape[0] > 0
        and periphery_strength > 0
        and periphery_cap > 0
    ):
        # centers: shape (Nc, 2) in degrees; latlon_valid: (Nv, 2) in degrees
        # 각 center에 대해 haversine(center, 모든 valid 후보) → (Nv, Nc) 거리 행렬 후 min
        dists_stack = []
        diner_lat_series = pd.Series(latlon_valid[:, 0], copy=False)
        diner_lon_series = pd.Series(latlon_valid[:, 1], copy=False)
        for c_lat, c_lon in centers:
            d_km = haversine(
                reviewer_lat=float(c_lat),
                reviewer_lon=float(c_lon),
                diner_lat=diner_lat_series,
                diner_lon=diner_lon_series,
            )  # (Nv,)
            dists_stack.append(np.asarray(d_km, dtype=np.float64))
        if dists_stack:
            D = np.vstack(dists_stack).T  # (Nv, Nc)
            dmin = D.min(axis=1).astype(np.float32, copy=False)
            # 0–1 정규화 후 가점
            dmin_n = minmax(dmin)  # float32, [0,1]
            bonus_valid = np.clip(
                periphery_strength * dmin_n, 0.0, periphery_cap
            ).astype(np.float32, copy=False)
            periphery_bonus[valid] = bonus_valid  # invalid 좌표는 0 유지

    base_scores_boosted = (base_scores_g + periphery_bonus).astype(
        np.float32, copy=False
    )

    # 5) 최종 MMR 재랭크 호출
    use_geo = float(np.isfinite(latlon).sum()) >= 2.0
    final_ids, final_scores = rerank_most_popular_with_diversity(
        item_ids=item_ids_g,
        base_scores=base_scores_boosted,
        item_meta=meta,
        k=k,
        lambda_div=lambda_div,
        w_cat=w_cat,
        w_geo=w_geo if use_geo else 0.0,
        geo_tau_km=geo_tau_km,
        coverage_min=coverage_min,
        coverage_max=coverage_max,
        region_of=region_of,
        popularity_weight=0.0,
        popularity_scores=None,
        normalize_rel=True,
        top_m=None,
        debug=False,
    )
    return final_ids, final_scores
