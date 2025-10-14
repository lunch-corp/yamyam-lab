import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from inference import haversine


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

    meta = item_meta[list(required)].drop_duplicates("diner_idx").reset_index(drop=True)
    id2row = pd.Series(meta.index, index=meta["diner_idx"].astype(int)).to_dict()
    return meta, id2row


def filter_candidates_by_meta_and_topm(
    item_ids: np.ndarray,
    rel: np.ndarray,
    id2row: Dict[int, int],
    top_m: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter candidate items by available metadata and optionally keep only top-M by relevance.

    Args:
        item_ids (np.ndarray): Candidate item IDs.
        rel (np.ndarray): Relevance scores for items.
        id2row (Dict[int, int]): Mapping from diner_idx to row index.
        top_m (int | None): If set, keep only top-M items by relevance.

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


def build_coverage_labels(
    item_ids: np.ndarray,
    cats: pd.Series,
    region_of: dict[int, str] | None = None,
) -> Tuple[List[List[str]], Dict[str, np.ndarray]]:
    """
    Build coverage labels for items (category + region) and index mapping.

    Args:
        item_ids (np.ndarray): Candidate item IDs.
        cats (pd.Series): Category labels for items.
        region_of (dict[int, str] | None): Mapping from item_id to region label.

    Returns:
        Tuple[List[List[str]], Dict[str, np.ndarray]]:
            - List of labels per item (category + region).
            - Dictionary mapping label -> array of indices having that label.
    """
    region_of = region_of or {}
    categories_str = cats.astype(str).to_numpy()

    lab_cat = np.array(
        [f"diner_category_large:{c}" for c in categories_str], dtype=object
    )
    lab_reg = np.array(
        [
            f"diner_road_address:{region_of.get(int(cid), 'unknown')}"
            for cid in item_ids
        ],
        dtype=object,
    )

    labels_by_idx = [[lab_cat[i], lab_reg[i]] for i in range(item_ids.size)]

    label_to_indices: Dict[str, np.ndarray] = {}
    for labs in (lab_cat, lab_reg):
        for lab in np.unique(labs):
            label_to_indices[lab] = np.flatnonzero(labs == lab)

    return labels_by_idx, label_to_indices


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


def geo_precompute(
    meta_sorted: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
