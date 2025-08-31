import json
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import re
import numpy as np
import pandas as pd
import requests
from torch import Tensor
from tqdm import tqdm

warnings.filterwarnings("ignore")


def haversine(
    reviewer_lat: float, reviewer_lon: float, diner_lat: pd.Series, diner_lon: pd.Series
) -> np.ndarray:
    """
    Compute the great-circle distance between a single point (lat1, lon1) and multiple points (lat2, lon2)
    using the Haversine formula in a vectorized way.

    Args:
        reviewer_lat (float): Latitude of the reviewer.
        reviewer_lon (float): Longitude of the reviewer.
        diner_lat (pd.Series): Latitude of the diners.
        diner_lon (pd.Series): Longitude of the diners.

    Returns:
        np.ndarray: Array of distances.
    """
    # Convert degrees to radians
    reviewer_lat, reviewer_lon = np.radians(reviewer_lat), np.radians(reviewer_lon)
    diner_lat, diner_lon = np.radians(diner_lat), np.radians(diner_lon)

    # Haversine formula
    dlat = diner_lat - reviewer_lat
    dlon = diner_lon - reviewer_lon
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(reviewer_lat) * np.cos(diner_lat) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth's radius in kilometers
    radius = 6371.0

    return radius * c


# 위도, 경도 반환하는 함수
def get_kakao_lat_lng(address: str) -> dict[str, str]:
    try:
        url = f"https://dapi.kakao.com/v2/local/search/keyword.json?query={address}"
        headers = {"Authorization": f"KakaoAK {os.getenv('KAKAO_REST_API_KEY')}"}

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            raise Exception(
                f"Geocoding API request failed with status code: {response.status_code}"
            )

        api_json = json.loads(response.text)
        address = api_json["documents"][0]
        crd = {"lat": float(address["y"]), "lng": float(address["x"])}
        return crd

    except Exception as e:
        raise Exception(f"Geocoding failed: {str(e)}")


def convert_tensor(
    ts: Tensor, structure: Union[dict, list]
) -> Dict[int, Union[List[int], Dict[int, int]]]:
    """
    Convert 2 dimensional tensor to dict or list.
    Original tensor includes interaction between reviewer and diner.

    Args:
        ts (Tensor): n x 2 dimension tensors whose columns are matched with (diner_id, reviewer_id).
            Should be careful of column ordering.
        structure (Union[dict, list]): Data type of value corresponding key in return object.

    Returns (Dict[int, Union[List[int], Dict[int, int]]]):
        Key is reviewer id and values are diner_id interacted by reviewer id.
        Data types of values are dictionary or list.
        In case dictionary, res[reviewer_id][diner_id] is 1 if interacted else 0.
        In case list, res[reviewer_id] is a list of diner_id interacted by reviewer id.
    """
    assert ts.shape[1] == 2
    assert structure in [dict, list]
    res = defaultdict(structure)
    for diner_id, reviewer_id in ts:
        reviewer_id = reviewer_id.item()
        diner_id = diner_id.item()
        if structure is dict:
            res[reviewer_id][diner_id] = 1
        else:
            res[reviewer_id].append(diner_id)
    return res


def get_num_workers() -> int:
    """
    Get number of workers for data loader in pytorch.

    Returns (int)
        Number of workers for data loader in pytorch. Note that even if there are
        lots of cpus, it may not be a good idea to use many of them because
        context switching overhead could interrupt training.
        It could be best to determine optimal num_workers with minimal experiments.
    """
    num_cores = os.cpu_count()
    return min(4, num_cores // 2)


def safe_divide(numerator, denominator):
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return 0


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in tqdm(df.columns, leave=False):
        df[col] = _reduce_column_memory(df[col], numerics)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print(
            f"Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )

    return df


def _reduce_column_memory(col: pd.Series, numerics: list) -> pd.Series:
    col_type = col.dtypes
    if col_type in numerics:
        c_min = col.min()
        c_max = col.max()
        if str(col_type)[:3] == "int":
            col = _reduce_int_memory(col, c_min, c_max)
        else:
            col = _reduce_float_memory(col, c_min, c_max)

    return col


def _reduce_int_memory(col: pd.Series, c_min: int, c_max: int) -> pd.Series:
    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
        return col.astype(np.int8)

    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
        return col.astype(np.int16)

    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
        return col.astype(np.int32)

    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
        return col.astype(np.int64)

    return col


def _reduce_float_memory(col: pd.Series, c_min: float, c_max: float) -> pd.Series:
    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
        return col.astype(np.float16)

    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
        return col.astype(np.float32)

    return col.astype(np.float64)


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
    return m.group(0) if m else parts[0] if parts else "unknown"


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
    mn, mx = float(x.min()), float(x.max())
    return (
        (x - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(x, dtype=np.float32)
    )


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
    assert len(item_ids) == len(base_scores), "item_ids/base_scores length mismatch"
    L = len(item_ids)
    return 0 if (L == 0 or k <= 0) else min(k, L)


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
    missing = required - set(item_meta.columns)
    if missing:
        raise ValueError(f"item_meta missing columns: {missing}")
    meta = (
        item_meta[["diner_idx", "diner_category_large", "diner_lat", "diner_lon"]]
        .drop_duplicates("diner_idx")
        .reset_index(drop=True)
    )
    id2row = {int(r.diner_idx): i for i, r in meta.iterrows()}
    return meta, id2row


def filter_candidates_by_meta_and_topm(
    item_ids: np.ndarray,
    rel: np.ndarray,
    id2row: Dict[int, int],
    top_m: Optional[int],
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
    has_meta = np.fromiter(
        (int(x) in id2row for x in item_ids), dtype=bool, count=len(item_ids)
    )
    item_ids = item_ids[has_meta]
    rel = rel[has_meta]
    if item_ids.size == 0:
        return item_ids, rel, np.array([], dtype=int)

    if top_m is not None and top_m < item_ids.size:
        top_idx = np.argpartition(-rel, kth=top_m - 1)[:top_m]
        item_ids = item_ids[top_idx]
        rel = rel[top_idx]

    rows = np.fromiter(
        (id2row[int(cid)] for cid in item_ids), dtype=int, count=item_ids.size
    )
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
    cat_codes = cats.cat.codes.to_numpy(dtype=np.int32)
    return cat_codes, cats


def build_coverage_labels(
    item_ids: np.ndarray,
    cats: pd.Series,
    region_of: Optional[Dict[int, str]],
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
    regions = np.array(
        [region_of.get(int(cid), "unknown") for cid in item_ids], dtype=object
    )

    lab_cat = np.array(
        [f"diner_category_large:{v}" for v in categories_str], dtype=object
    )
    lab_reg = np.array([f"diner_road_address:{r}" for r in regions], dtype=object)
    labels_by_idx = [[lab_cat[i], lab_reg[i]] for i in range(item_ids.size)]

    label_to_indices: Dict[str, np.ndarray] = {}
    for lab in np.unique(lab_cat):
        label_to_indices[lab] = np.flatnonzero(lab_cat == lab)
    for lab in np.unique(lab_reg):
        label_to_indices[lab] = np.flatnonzero(lab_reg == lab)
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
        if mx is None:
            continue
        if cov_counts.get(lab, 0) >= mx:
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
    sel_lat_deg = float(np.degrees(sel_lat_rad))
    sel_lon_deg = float(np.degrees(sel_lon_rad))

    d_km = haversine(
        reviewer_lat=sel_lat_deg,
        reviewer_lon=sel_lon_deg,
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
    lat_rad = np.deg2rad(meta_sorted["diner_lat"].to_numpy(dtype=np.float32)).astype(
        np.float32
    )
    lon_rad = np.deg2rad(meta_sorted["diner_lon"].to_numpy(dtype=np.float32)).astype(
        np.float32
    )
    cos_lat = np.cos(lat_rad).astype(np.float32)
    return (
        np.ascontiguousarray(lat_rad),
        np.ascontiguousarray(lon_rad),
        np.ascontiguousarray(cos_lat),
    )
