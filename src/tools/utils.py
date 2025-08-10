import json
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import requests
from torch import Tensor
from tqdm import tqdm

warnings.filterwarnings("ignore")


# 위도, 경도 반환하는 함수
def get_kakao_lat_lng(address: str) -> dict[str, str]:
    try:
        url = f"https://dapi.kakao.com/v2/local/search/address.json?query={address}"
        headers = {"Authorization": f"KakaoAK {os.getenv('KAKAO_REST_API_KEY')}"}

        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            raise Exception(
                f"Geocoding API request failed with status code: {response.status_code}"
            )

        api_json = json.loads(response.text)
        address = api_json["documents"][0]["address"]
        crd = {"lat": str(address["y"]), "lng": str(address["x"])}
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
