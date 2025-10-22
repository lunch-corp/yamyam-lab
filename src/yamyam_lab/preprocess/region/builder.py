# -*- coding: utf-8 -*-
"""
🗺️ 서울시 음식점 추천용 도보 권역 생성 도구

이 모듈은 서울시 전체를 H3 해상도 10으로 빈틈없이 커버한 후,
도보 거리 기반 클러스터링을 통해 음식점 추천에 적합한 권역을 생성합니다.

주요 기능:
- 🏙️ OSMnx로 서울시 행정경계 자동 획득
- 📍 H3 해상도 10 기반 서울시 전체 커버리지
- 🚶 OSRM 도보 거리 계산 (Haversine 백업)
- 🔗 그래프 기반 연결 요소 분석
- 📁 CSV/GeoJSON 결과 출력
- 📈 실시간 진행상황 추적

작성자: yamyam-lab
버전: 3.0 (음식점 추천용 도보 권역 생성 특화)
"""

import hashlib
import json
import logging
import math
import pickle
from collections import deque
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Set, Tuple

import h3  # pip install h3
import networkx as nx
import osmnx as ox  # pip install osmnx
import pandas as pd
import requests
from shapely.geometry import Polygon
from tqdm import tqdm

from yamyam_lab.data.base import BaseDatasetLoader
from yamyam_lab.data.config import DataConfig

# 로깅 설정
logger = logging.getLogger(__name__)

# ---------------------------
# H3 v3/v4 호환성 어댑터
# ---------------------------
_HAS_V4 = hasattr(h3, "latlng_to_cell")  # v4이면 True


def h3_geo_to_cell(lat: float, lon: float, res: int) -> str:
    """H3 v3/v4 호환 좌표→셀 변환"""
    return h3.latlng_to_cell(lat, lon, res) if _HAS_V4 else h3.geo_to_h3(lat, lon, res)


def h3_cell_to_boundary_latlon(cell_id: str) -> List[Tuple[float, float]]:
    """H3 v3/v4 호환 셀→경계좌표 변환 (lat, lon) 순서"""
    if _HAS_V4:
        return h3.cell_to_boundary(cell_id)
    else:
        return h3.h3_to_geo_boundary(cell_id, geo_json=False)


def h3_neighbors(cell_id: str, k: int = 1) -> Set[str]:
    """H3 v3/v4 호환 k-ring 이웃 셀"""
    if _HAS_V4:
        s = set(h3.grid_disk(cell_id, k))
    else:
        s = set(h3.k_ring(cell_id, k))
    s.discard(cell_id)
    return s


# ---------------------------
# OSRM Distance Cache Manager
# ---------------------------
class OSRMDistanceCache:
    """OSRM 거리 캐시를 파일로 저장/로드하는 관리 클래스"""

    def __init__(self, cache_dir: str = "cache", region_name: str = "default"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.region_name = region_name
        self.cache: Dict[Tuple[str, str], float] = {}
        self.cache_file = self.cache_dir / f"osrm_distance_cache_{region_name}.pkl"
        self.metadata_file = (
            self.cache_dir / f"osrm_distance_cache_{region_name}_metadata.json"
        )
        self.api_calls_count = 0
        self.cache_hits_count = 0

    def _generate_cache_key(
        self, lat1: float, lon1: float, lat2: float, lon2: float, profile: str = "foot"
    ) -> str:
        """좌표와 프로필을 기반으로 캐시 키 생성"""
        # 소수점 6자리로 반올림하여 키 생성 (약 10cm 정밀도)
        coords = f"{lat1:.6f},{lon1:.6f},{lat2:.6f},{lon2:.6f},{profile}"
        return hashlib.md5(coords.encode()).hexdigest()

    def load_cache(self) -> bool:
        """캐시 파일에서 데이터 로드"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "rb") as f:
                    self.cache = pickle.load(f)

                # 메타데이터 로드
                metadata = {}
                if self.metadata_file.exists():
                    with open(self.metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                logger.info(f"OSRM 거리 캐시 로드 완료: {len(self.cache):,}개 항목")
                if metadata:
                    logger.info(f"캐시 생성일: {metadata.get('created_at', 'Unknown')}")
                    logger.info(
                        f"마지막 업데이트: {metadata.get('last_updated', 'Unknown')}"
                    )

                return True
            else:
                logger.info("기존 OSRM 거리 캐시 파일이 없습니다. 새로 생성합니다.")
                return False

        except Exception as e:
            logger.warning(f"캐시 로드 실패: {e}. 새로운 캐시를 생성합니다.")
            self.cache = {}
            return False

    def save_cache(self) -> bool:
        """캐시를 파일로 저장"""
        try:
            # 캐시 데이터 저장
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)

            # 메타데이터 저장
            metadata = {
                "region_name": self.region_name,
                "cache_size": len(self.cache),
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "api_calls_this_session": self.api_calls_count,
                "cache_hits_this_session": self.cache_hits_count,
                "cache_hit_rate": f"{self.cache_hits_count / max(1, self.cache_hits_count + self.api_calls_count) * 100:.1f}%",
            }

            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"OSRM 거리 캐시 저장 완료: {len(self.cache):,}개 항목")
            logger.info(
                f"이번 세션 API 호출: {self.api_calls_count}회, 캐시 히트: {self.cache_hits_count}회"
            )

            return True

        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
            return False

    def get_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        base_url: str = "https://router.project-osrm.org",
        profile: str = "foot",
        timeout: float = 5.0,
    ) -> Optional[float]:
        """캐시를 활용한 거리 조회"""
        cache_key = self._generate_cache_key(lat1, lon1, lat2, lon2, profile)

        # 캐시에서 조회
        if cache_key in self.cache:
            self.cache_hits_count += 1
            return self.cache[cache_key]

        # 캐시에 없으면 API 호출
        self.api_calls_count += 1
        distance = osrm_distance_m(lat1, lon1, lat2, lon2, base_url, profile, timeout)

        # 성공한 경우 캐시에 저장
        if distance is not None:
            self.cache[cache_key] = distance

        return distance

    def get_stats(self) -> Dict:
        """캐시 통계 반환"""
        total_requests = self.api_calls_count + self.cache_hits_count
        hit_rate = (self.cache_hits_count / max(1, total_requests)) * 100

        return {
            "cache_size": len(self.cache),
            "api_calls": self.api_calls_count,
            "cache_hits": self.cache_hits_count,
            "hit_rate_percent": hit_rate,
            "cache_file": str(self.cache_file),
            "file_exists": self.cache_file.exists(),
        }


# ---------------------------
# Distance utilities
# ---------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine 직선거리 (meters)."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def osrm_route_distance(
    coords: List[Tuple[float, float]],
    base_url: str = "https://router.project-osrm.org",
    profile: str = "foot",
    overview: str = "false",
    timeout: float = 5.0,
) -> Optional[Dict]:
    """
    OSRM /route API로 여러 좌표를 잇는 경로를 요청.
    coords: [(lon, lat), (lon, lat), ...]  # OSRM은 경도,위도 순서!
    반환: {
        "distance": float(총거리 m),
        "duration": float(총시간 s),
        "legs": List[{"distance": m, "duration": s}, ...],
        "waypoints": List[...]
    }  또는 실패 시 None
    """
    if len(coords) < 2:
        raise ValueError("coords must contain at least 2 points")

    try:
        coord_str = ";".join([f"{lon:.6f},{lat:.6f}" for lon, lat in coords])
        url = f"{base_url}/route/v1/{profile}/{coord_str}"
        params = {
            "overview": overview,  # "false"|"simplified"|"full"
            "alternatives": "false",
            "steps": "false",
            "annotations": "false",
        }
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None

        route = data["routes"][0]
        total_distance = float(route["distance"])  # m
        total_duration = float(route["duration"])  # s
        legs_summary = []
        for leg in route.get("legs", []):
            legs_summary.append(
                {
                    "distance": float(leg["distance"]),
                    "duration": float(leg["duration"]),
                }
            )

        return {
            "distance": total_distance,
            "duration": total_duration,
            "legs": legs_summary,
            "waypoints": data.get("waypoints", []),
        }
    except Exception:
        return None


def osrm_distance_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    base_url: str = "https://router.project-osrm.org",
    profile: str = "foot",
    timeout: float = 5.0,
) -> Optional[float]:
    """
    2점 전용 OSRM 래퍼. (driving/foot/cycling 프로필 지원)

    Args:
        lat1, lon1: 첫 번째 점의 위도, 경도
        lat2, lon2: 두 번째 점의 위도, 경도
        base_url: OSRM 서버 URL
        profile: 경로 프로필 (foot, driving, cycling)
        timeout: 요청 타임아웃 (초)

    Returns:
        거리(미터) 또는 실패 시 None
    """
    try:
        res = osrm_route_distance(
            coords=[(lon1, lat1), (lon2, lat2)],
            base_url=base_url,
            profile=profile,
            timeout=timeout,
            overview="false",
        )
        return None if res is None else res["distance"]
    except Exception as e:
        logger.warning(f"OSRM distance calculation failed: {e}")
        return None


# ---------------------------
# Seoul boundary & H3 coverage
# ---------------------------
def get_region_boundary(region_name: str = "서울특별시") -> Polygon:
    """OSMnx로 지정된 지역의 행정경계 가져오기"""
    logger.info(f"{region_name} 행정경계를 OSMnx로 가져오는 중...")

    # 다양한 검색어 시도
    search_terms = []
    if region_name == "서울특별시":
        search_terms = ["Seoul, South Korea", "서울특별시, 대한민국"]
    else:
        # 구/동 단위는 한글명 우선
        search_terms = [
            f"{region_name}, 대한민국",
            f"{region_name}, 서울특별시, 대한민국",
        ]

    gdf = None
    for search_term in search_terms:
        try:
            logger.info(f"검색어 시도: {search_term}")
            gdf = ox.geocode_to_gdf(search_term)
            break
        except Exception as e:
            logger.warning(f"'{search_term}' 검색 실패: {e}")
            continue

    if gdf is None:
        raise ValueError(
            f"'{region_name}' 지역을 찾을 수 없습니다. 정확한 지역명을 확인해주세요."
        )

    # WGS84로 변환 및 단일 폴리곤으로 통합
    gdf = gdf.to_crs(epsg=4326)
    region_geom = gdf.unary_union
    logger.info(f"{region_name} 경계 획득 완료: {region_geom.geom_type}")

    # 경계 정보 출력
    bounds = gdf.bounds
    logger.info(f"경계 좌표: {bounds.iloc[0].to_dict()}")

    return region_geom


def cell_polygon(cell_id: str) -> Polygon:
    """H3 셀 ID → Shapely Polygon (lon, lat 순서)"""
    boundary_latlon = h3_cell_to_boundary_latlon(cell_id)
    boundary_lonlat = [(lon, lat) for (lat, lon) in boundary_latlon]
    return Polygon(boundary_lonlat)


def flood_fill_region_cells(
    region_boundary: Polygon, resolution: int, region_name: str = "지역"
) -> Set[str]:
    """
    지역 경계 내를 H3 셀로 완전히 커버하는 flood-fill 알고리즘
    - 지역 중심에서 시작해 이웃으로 확장
    - 셀과 경계가 교차하는 모든 셀을 포함
    """
    logger.info(f"{region_name}을 H3 해상도 {resolution}로 완전 커버 중...")
    start_time = time()

    # 시작점: 지역 중심
    center = region_boundary.representative_point()
    start_cell = h3_geo_to_cell(center.y, center.x, resolution)

    result = set()
    visited = set([start_cell])
    queue = deque([start_cell])

    while queue:
        current_cell = queue.popleft()
        cell_poly = cell_polygon(current_cell)

        # 경계와 교차하지 않으면 제외
        if not cell_poly.intersects(region_boundary):
            continue

        # 교차하는 셀은 포함
        result.add(current_cell)

        # 이웃 셀들을 큐에 추가
        for neighbor in h3_neighbors(current_cell, 1):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    elapsed = time() - start_time
    logger.info(f"{region_name} H3 커버리지 완료: {len(result)}개 셀 ({elapsed:.2f}s)")
    return result


def load_restaurant_data_with_dataloader(
    data_config: DataConfig, region_name: str = "서울특별시"
) -> pd.DataFrame:
    """기존 DataLoader를 사용하여 음식점 데이터 로드 및 지역 필터링"""
    logger.info("DataLoader를 사용하여 음식점 데이터 로드 중...")
    start_time = time()

    try:
        # BaseDatasetLoader를 사용하여 데이터 로드
        loader = BaseDatasetLoader(data_config)
        review, diner, diner_with_raw_category = loader.load_dataset()

        initial_count = len(diner)

        # 서울 지역 필터링 (region_name에 따라 조정)
        if region_name == "서울특별시":
            filtered_df = diner[
                diner["diner_road_address"].str.contains("서울", na=False)
            ].copy()
        else:
            # 특정 구/동 필터링
            filtered_df = diner[
                diner["diner_road_address"].str.contains(region_name, na=False)
            ].copy()

        # 필요한 컬럼만 선택 및 타입 변환
        required_columns = [
            "diner_idx",
            "diner_name",
            "diner_lat",
            "diner_lon",
            "diner_review_cnt",
            "diner_review_avg",
            "diner_road_address",
        ]

        # bayesian_score가 있으면 포함, 없으면 0으로 생성
        if "bayesian_score" in filtered_df.columns:
            required_columns.append("bayesian_score")
        else:
            filtered_df["bayesian_score"] = 0.0
            required_columns.append("bayesian_score")

        filtered_df = filtered_df[required_columns].copy()

        # 숫자 컬럼들을 안전하게 변환
        filtered_df["diner_lat"] = pd.to_numeric(
            filtered_df["diner_lat"], errors="coerce"
        )
        filtered_df["diner_lon"] = pd.to_numeric(
            filtered_df["diner_lon"], errors="coerce"
        )
        filtered_df["diner_review_cnt"] = (
            pd.to_numeric(filtered_df["diner_review_cnt"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        filtered_df["diner_review_avg"] = (
            pd.to_numeric(filtered_df["diner_review_avg"], errors="coerce")
            .fillna(0)
            .astype(float)
        )
        filtered_df["bayesian_score"] = (
            pd.to_numeric(filtered_df["bayesian_score"], errors="coerce")
            .fillna(0)
            .astype(float)
        )

        # NaN이 있는 행 제거
        filtered_df = filtered_df.dropna(subset=["diner_lat", "diner_lon"])

        # 좌표 범위 검증
        valid_coords = filtered_df[
            (filtered_df["diner_lat"] >= 37.0)
            & (filtered_df["diner_lat"] <= 38.0)
            & (filtered_df["diner_lon"] >= 126.0)
            & (filtered_df["diner_lon"] <= 128.0)
        ]

        elapsed = time() - start_time
        logger.info(f"음식점 데이터 로드 완료 ({elapsed:.2f}s)")
        logger.info(
            f"전체: {initial_count:,}개 → {region_name}: {len(valid_coords):,}개 음식점"
        )

        return valid_coords

    except Exception as e:
        logger.error(f"음식점 데이터 로드 실패: {e}")
        return pd.DataFrame()


def map_restaurants_to_h3_cells(
    restaurants_df: pd.DataFrame, resolution: int
) -> pd.DataFrame:
    """음식점을 H3 셀에 매핑하고 셀별 음식점 통계 계산"""
    logger.info("음식점을 H3 셀에 매핑 중...")
    start_time = time()

    if restaurants_df.empty:
        logger.warning("음식점 데이터가 비어있습니다.")
        return pd.DataFrame()

    # 음식점을 H3 셀에 매핑
    restaurants_df = restaurants_df.copy()
    restaurants_df["cell_id"] = restaurants_df.apply(
        lambda row: h3_geo_to_cell(row["diner_lat"], row["diner_lon"], resolution),
        axis=1,
    )

    # 셀별 음식점 통계 계산
    cell_stats = (
        restaurants_df.groupby("cell_id")
        .agg(
            {
                "diner_idx": "count",  # 음식점 수
                "diner_review_cnt": "sum",  # 총 리뷰 수
                "diner_review_avg": "mean",  # 평균 평점
                "bayesian_score": "mean",  # 평균 베이지안 점수
            }
        )
        .round(2)
    )

    cell_stats.columns = [
        "restaurant_count",
        "total_reviews",
        "avg_rating",
        "avg_bayesian_score",
    ]

    # 데이터 타입 안전하게 변환
    cell_stats["restaurant_count"] = cell_stats["restaurant_count"].astype(int)
    cell_stats["total_reviews"] = cell_stats["total_reviews"].fillna(0).astype(int)
    cell_stats["avg_rating"] = cell_stats["avg_rating"].fillna(0).astype(float).round(2)
    cell_stats["avg_bayesian_score"] = (
        cell_stats["avg_bayesian_score"].fillna(0).astype(float).round(3)
    )

    # 셀 중심 좌표 추가
    def get_cell_center(cell_id):
        if _HAS_V4:
            return h3.cell_to_latlng(cell_id)
        else:
            return h3.h3_to_geo(cell_id)

    cell_centers = pd.DataFrame(
        [
            {
                "cell_id": cell_id,
                "cell_lat": get_cell_center(cell_id)[0],
                "cell_lon": get_cell_center(cell_id)[1],
            }
            for cell_id in cell_stats.index
        ]
    ).set_index("cell_id")

    # 통계와 좌표 결합
    result_df = cell_stats.join(cell_centers).reset_index()

    elapsed = time() - start_time
    logger.info(f"H3 매핑 완료 ({elapsed:.2f}s): {len(result_df)}개 셀에 음식점 분포")
    logger.info(f"음식점이 있는 셀: {len(result_df):,}개")
    logger.info(f"셀당 평균 음식점 수: {result_df['restaurant_count'].mean():.1f}개")

    return result_df


def region_cells_to_dataframe(
    cell_ids: Set[str], restaurants_df: pd.DataFrame = None, resolution: int = 10
) -> pd.DataFrame:
    """지역 H3 셀 집합을 DataFrame으로 변환 (음식점 정보 포함)"""
    logger.info("H3 셀 정보를 DataFrame으로 변환 중...")
    start_time = time()

    # 기본 셀 정보 생성
    cells_data = []
    for cell_id in tqdm(cell_ids, desc="셀 정보 변환"):
        if _HAS_V4:
            lat, lon = h3.cell_to_latlng(cell_id)
        else:
            lat, lon = h3.h3_to_geo(cell_id)
        cells_data.append(
            {
                "cell_id": cell_id,
                "cell_lat": lat,
                "cell_lon": lon,
                "point_count": 1,  # 기본 가중치
            }
        )

    result_df = pd.DataFrame(cells_data)

    # 음식점 데이터가 있으면 매핑
    if restaurants_df is not None and not restaurants_df.empty:
        logger.info("음식점 정보를 셀에 매핑 중...")
        restaurants_df.dropna(subset=["diner_lat", "diner_lon"], inplace=True)
        restaurant_cells = map_restaurants_to_h3_cells(restaurants_df, resolution)

        # 음식점 정보 조인 (left join으로 모든 셀 유지)
        result_df = result_df.merge(
            restaurant_cells[
                [
                    "cell_id",
                    "restaurant_count",
                    "total_reviews",
                    "avg_rating",
                    "avg_bayesian_score",
                ]
            ],
            on="cell_id",
            how="left",
        )

        # 음식점이 없는 셀은 0으로 채우기
        result_df["restaurant_count"] = (
            result_df["restaurant_count"].fillna(0).astype(int)
        )
        result_df["total_reviews"] = result_df["total_reviews"].fillna(0).astype(int)
        result_df["avg_rating"] = (
            result_df["avg_rating"].fillna(0).astype(float).round(2)
        )
        result_df["avg_bayesian_score"] = (
            result_df["avg_bayesian_score"].fillna(0).astype(float).round(3)
        )

        # 음식점 수를 point_count에 반영 (가중치)
        result_df["point_count"] = (
            result_df["restaurant_count"].astype(int) + 1
        )  # 최소 1개는 보장

        logger.info(
            f"음식점이 있는 셀: {(result_df['restaurant_count'] > 0).sum():,}개"
        )
        logger.info(
            f"음식점이 없는 셀: {(result_df['restaurant_count'] == 0).sum():,}개"
        )

    elapsed = time() - start_time
    logger.info(f"DataFrame 변환 완료 ({elapsed:.2f}s)")
    return result_df


# ---------------------------
# H3 helpers (legacy support)
# ---------------------------
def to_h3_cells(
    df: pd.DataFrame, lat_col: str, lon_col: str, resolution: int
) -> pd.DataFrame:
    """포인트 → H3 셀 집계 및 셀 중심 좌표 (기존 호환성 유지)"""
    logger.info(f"Converting {len(df)} points to H3 cells (resolution={resolution})")
    start_time = time()

    tmp = df[[lat_col, lon_col]].dropna().copy()
    logger.info(f"Processing {len(tmp)} valid coordinates after removing NaN")

    tmp["cell_id"] = tmp.apply(
        lambda r: h3_geo_to_cell(r[lat_col], r[lon_col], resolution), axis=1
    )
    agg = tmp.groupby("cell_id").size().reset_index(name="point_count")

    # H3 v3/v4 호환 셀 중심 좌표
    def get_cell_center(cell_id):
        if _HAS_V4:
            return pd.Series(h3.cell_to_latlng(cell_id))
        else:
            return pd.Series(h3.h3_to_geo(cell_id))

    agg[["cell_lat", "cell_lon"]] = agg["cell_id"].apply(get_cell_center)

    elapsed = time() - start_time
    logger.info(f"Generated {len(agg)} H3 cells in {elapsed:.2f}s")
    return agg[["cell_id", "cell_lat", "cell_lon", "point_count"]]


def neighbor_candidates(cell_id: str, k: int) -> Set[str]:
    """H3 k-ring 이웃 후보 (기존 호환성 유지)"""
    return h3_neighbors(cell_id, k)


# ---------------------------
# Graph construction
# ---------------------------
def build_cell_graph(
    cells_df: pd.DataFrame,
    distance_threshold_m: float,
    distance_metric: str = "osrm_then_haversine",
    osrm_base_url: str = "https://router.project-osrm.org",
    osrm_profile: str = "foot",
    osrm_timeout: float = 5.0,
    kring: int = 1,
    osrm_cache: Optional[OSRMDistanceCache] = None,
) -> Tuple[nx.Graph, Dict[Tuple[str, str], float]]:
    """
    셀 중심점 기반 그래프 구성.
    - kring > 0: H3 k-ring 이웃 셀만 거리 계산 → 고속
    - kring == 0: 모든 쌍 비교 (데이터 적을 때만 권장)
    distance_metric: "osrm" | "haversine" | "osrm_then_haversine"
    """
    logger.info(
        f"Building graph for {len(cells_df)} cells (distance_metric={distance_metric}, threshold={distance_threshold_m}m)"
    )
    start_time = time()

    G = nx.Graph()
    # 노드 추가
    logger.info("Adding nodes to graph...")
    for _, row in tqdm(cells_df.iterrows(), total=len(cells_df), desc="Adding nodes"):
        G.add_node(
            row["cell_id"],
            lat=row["cell_lat"],
            lon=row["cell_lon"],
            point_count=int(row["point_count"]),
        )

    # 빠른 조회용 dict
    latlon = {
        r["cell_id"]: (float(r["cell_lat"]), float(r["cell_lon"]))
        for _, r in cells_df.iterrows()
    }
    present = set(latlon.keys())

    # 거리 캐시 (로컬 세션용, OSRM 캐시와 별도)
    dcache: Dict[Tuple[str, str], float] = {}

    def cell_distance(a: str, b: str) -> Optional[float]:
        key = (a, b) if a < b else (b, a)
        if key in dcache:
            return dcache[key]

        lat_a, lon_a = latlon[a]
        lat_b, lon_b = latlon[b]

        if distance_metric == "osrm":
            if osrm_cache:
                d = osrm_cache.get_distance(
                    lat_a,
                    lon_a,
                    lat_b,
                    lon_b,
                    base_url=osrm_base_url,
                    profile=osrm_profile,
                    timeout=osrm_timeout,
                )
            else:
                d = osrm_distance_m(
                    lat_a,
                    lon_a,
                    lat_b,
                    lon_b,
                    base_url=osrm_base_url,
                    profile=osrm_profile,
                    timeout=osrm_timeout,
                )
        elif distance_metric == "haversine":
            d = haversine_m(lat_a, lon_a, lat_b, lon_b)
        else:  # osrm_then_haversine
            if osrm_cache:
                d = osrm_cache.get_distance(
                    lat_a,
                    lon_a,
                    lat_b,
                    lon_b,
                    base_url=osrm_base_url,
                    profile=osrm_profile,
                    timeout=osrm_timeout,
                )
            else:
                d = osrm_distance_m(
                    lat_a,
                    lon_a,
                    lat_b,
                    lon_b,
                    base_url=osrm_base_url,
                    profile=osrm_profile,
                    timeout=osrm_timeout,
                )
            if d is None:
                d = haversine_m(lat_a, lon_a, lat_b, lon_b)

        if d is not None:
            dcache[key] = d
        return d

    # 엣지 추가
    logger.info("Computing distances and adding edges...")
    edges_added = 0
    distance_calculations = 0

    if kring > 0:
        # 각 셀의 k-ring 이웃만 비교
        logger.info(f"Using k-ring strategy with k={kring}")
        for cid in tqdm(present, desc="Processing cells"):
            for nb in neighbor_candidates(cid, kring):
                if nb in present and cid < nb:  # 중복 계산 방지
                    distance_calculations += 1
                    d = cell_distance(cid, nb)
                    if d is not None and d <= distance_threshold_m:
                        G.add_edge(cid, nb, distance_m=d)
                        edges_added += 1
    else:
        # 모든 쌍 비교 (N^2) — 데이터가 매우 적을 때만
        logger.info("Using all-pairs strategy (O(n²))")
        ids = sorted(present)
        total_pairs = len(ids) * (len(ids) - 1) // 2
        with tqdm(total=total_pairs, desc="Computing distances") as pbar:
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    distance_calculations += 1
                    d = cell_distance(a, b)
                    if d is not None and d <= distance_threshold_m:
                        G.add_edge(a, b, distance_m=d)
                        edges_added += 1
                    pbar.update(1)

    elapsed = time() - start_time
    logger.info(f"Graph construction completed in {elapsed:.2f}s")
    logger.info(
        f"Nodes: {G.number_of_nodes()}, Edges: {edges_added}, Distance calculations: {distance_calculations}"
    )
    logger.info(
        f"Local cache hit rate: {(distance_calculations - len(dcache)) / distance_calculations * 100:.1f}%"
    )

    # OSRM 캐시 통계 출력
    if osrm_cache:
        cache_stats = osrm_cache.get_stats()
        logger.info(
            f"OSRM cache stats: {cache_stats['cache_hits']} hits, {cache_stats['api_calls']} API calls"
        )
        logger.info(f"OSRM cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")

    return G, dcache


def calculate_region_diameter(G: nx.Graph, nodes: List[str]) -> float:
    """권역 내 최대 거리(지름) 계산"""
    if len(nodes) < 2:
        return 0.0

    max_distance = 0.0
    # 서브그래프 생성
    subgraph = G.subgraph(nodes)

    # 모든 쌍 간의 최단 경로 중 최대값 찾기
    try:
        # 연결된 그래프에서만 계산
        if nx.is_connected(subgraph):
            # 모든 쌍 간 최단 경로 계산 (edge weight 사용)
            path_lengths = dict(
                nx.all_pairs_dijkstra_path_length(subgraph, weight="distance_m")
            )
            for source in path_lengths:
                for target in path_lengths[source]:
                    if source != target:
                        max_distance = max(max_distance, path_lengths[source][target])
    except:
        # 계산 실패 시 0 반환
        pass

    return max_distance


def split_large_region(
    G: nx.Graph,
    nodes: List[str],
    max_distance_m: float,
    max_cells_per_region: Optional[int] = None,
) -> List[List[str]]:
    """
    큰 권역을 작은 권역들로 분할.
    - 제약 1: 권역 지름 <= max_distance_m
    - 제약 2: 권역 셀 수 <= max_cells_per_region (옵션)
    두 제약을 모두 만족할 때까지 Girvan-Newman으로 재귀 분할.
    """
    if len(nodes) < 2:
        return [nodes]

    # 서브그래프 생성
    subgraph = G.subgraph(nodes)

    # 현재 권역의 지름 계산
    diameter = calculate_region_diameter(G, nodes)

    meets_size = (
        True if max_cells_per_region is None else (len(nodes) <= max_cells_per_region)
    )
    meets_diameter = diameter <= max_distance_m

    if meets_size and meets_diameter:
        return [nodes]  # 분할 불필요

    # Girvan-Newman 알고리즘으로 커뮤니티 분할
    try:
        communities = list(nx.community.girvan_newman(subgraph))
        if len(communities) > 0:
            # 첫 번째 분할 결과 사용
            split_communities = communities[0]
            result = []

            for community in split_communities:
                community_nodes = list(community)
                # 재귀적으로 더 분할이 필요한지 확인
                sub_regions = split_large_region(
                    G,
                    community_nodes,
                    max_distance_m,
                    max_cells_per_region,
                )
                result.extend(sub_regions)

            return result
        else:
            return [nodes]
    except:
        # 분할 실패 시 원본 반환
        return [nodes]


def label_regions_from_components(
    G: nx.Graph,
    min_cells_per_region: int = 1,
    max_region_distance_m: float = 2000.0,
    max_cells_per_region: Optional[int] = None,
) -> Dict[str, int]:
    """
    연결요소를 region_id로 라벨링.
    - 작은 조각(<min_cells)은 -1(노이즈)
    - 큰 권역은 (지름, 최대 셀 수) 제약을 만족하도록 분할
    """
    logger.info(
        f"Analyzing connected components (min_cells={min_cells_per_region}, max_distance={max_region_distance_m}m, max_cells={max_cells_per_region})"
    )
    start_time = time()

    regions: Dict[str, int] = {}
    comp_id = 0
    components = list(nx.connected_components(G))

    large_components = 0
    small_components = 0
    split_regions = 0

    for comp in tqdm(components, desc="Processing components"):
        comp_nodes = list(comp)

        if len(comp_nodes) < min_cells_per_region:
            # 작은 조각은 노이즈로 처리
            for cid in comp_nodes:
                regions[cid] = -1
            small_components += 1
            continue

        # 큰 권역 분할 시도 (지름/셀 수 제약 동시 만족)
        sub_regions = split_large_region(
            G,
            comp_nodes,
            max_region_distance_m,
            max_cells_per_region=max_cells_per_region,
        )

        for sub_region in sub_regions:
            if len(sub_region) >= min_cells_per_region:
                for cid in sub_region:
                    regions[cid] = comp_id
                comp_id += 1
                large_components += 1

                if len(sub_regions) > 1:
                    split_regions += 1
            else:
                # 분할 후에도 작은 조각은 노이즈
                for cid in sub_region:
                    regions[cid] = -1
                small_components += 1

    elapsed = time() - start_time
    logger.info(f"Region labeling completed in {elapsed:.2f}s")
    logger.info(
        f"Found {large_components} valid regions, {small_components} small components (noise)"
    )
    if split_regions > 0:
        logger.info(f"Split {split_regions} large regions due to distance constraint")

    return regions


def reassign_orphan_cells(
    G: nx.Graph,
    cells_df: pd.DataFrame,
    region_labels: Dict[str, int],
    max_reassign_distance_m: float = 1500.0,
    osrm_cache: Optional[OSRMDistanceCache] = None,
) -> Dict[str, int]:
    """
    고아 셀(region_id = -1)을 가장 가까운 이웃 권역에 재할당.

    Args:
        G: 셀 그래프
        cells_df: 셀 정보 DataFrame
        region_labels: 기존 권역 라벨 딕셔너리
        max_reassign_distance_m: 재할당 최대 거리 (미터)

    Returns:
        업데이트된 권역 라벨 딕셔너리
    """
    logger.info("고아 셀(노이즈)을 이웃 권역에 재할당 중...")
    start_time = time()

    # 고아 셀과 유효 권역 셀 분리
    orphan_cells = [cid for cid, rid in region_labels.items() if rid == -1]
    valid_cells = [cid for cid, rid in region_labels.items() if rid >= 0]

    if not orphan_cells:
        logger.info("재할당할 고아 셀이 없습니다.")
        return region_labels

    logger.info(f"재할당 대상: {len(orphan_cells)}개 고아 셀")

    # 셀 좌표 매핑
    cell_coords = {
        row["cell_id"]: (row["cell_lat"], row["cell_lon"])
        for _, row in cells_df.iterrows()
    }

    updated_labels = region_labels.copy()
    reassigned_count = 0

    for orphan_cell in tqdm(orphan_cells, desc="고아 셀 재할당"):
        if orphan_cell not in cell_coords:
            continue

        orphan_lat, orphan_lon = cell_coords[orphan_cell]
        best_region_id = -1
        best_distance = float("inf")
        best_neighbor_cell = None

        # k-ring 이웃들 중에서 유효 권역에 속한 셀 찾기
        for k in range(1, 4):  # 1, 2, 3 ring까지 확장 탐색
            neighbors = h3_neighbors(orphan_cell, k)

            for neighbor_cell in neighbors:
                if neighbor_cell in valid_cells:
                    neighbor_region = region_labels[neighbor_cell]
                    if neighbor_region >= 0:  # 유효 권역
                        neighbor_lat, neighbor_lon = cell_coords.get(
                            neighbor_cell, (None, None)
                        )
                        if neighbor_lat is None:
                            continue

                        # 도보 거리 계산
                        distance = haversine_m(
                            orphan_lat, orphan_lon, neighbor_lat, neighbor_lon
                        )

                        # OSRM 도보 거리로 더 정확하게 계산 (선택적)
                        if osrm_cache:
                            osrm_dist = osrm_cache.get_distance(
                                orphan_lat, orphan_lon, neighbor_lat, neighbor_lon
                            )
                        else:
                            osrm_dist = osrm_distance_m(
                                orphan_lat, orphan_lon, neighbor_lat, neighbor_lon
                            )
                        if osrm_dist is not None:
                            distance = osrm_dist

                        if (
                            distance < best_distance
                            and distance <= max_reassign_distance_m
                        ):
                            best_distance = distance
                            best_region_id = neighbor_region
                            best_neighbor_cell = neighbor_cell

            # 가까운 이웃을 찾았으면 더 멀리 탐색하지 않음
            if best_region_id >= 0:
                break

        # 재할당 수행
        if best_region_id >= 0:
            updated_labels[orphan_cell] = best_region_id
            reassigned_count += 1
            logger.debug(
                f"셀 {orphan_cell} → 권역 {best_region_id} (거리: {best_distance:.0f}m, 이웃: {best_neighbor_cell})"
            )

    elapsed = time() - start_time
    logger.info(f"고아 셀 재할당 완료 ({elapsed:.2f}s)")
    logger.info(f"재할당 성공: {reassigned_count}/{len(orphan_cells)}개 셀")
    logger.info(f"남은 고아 셀: {len(orphan_cells) - reassigned_count}개")

    return updated_labels


def save_graph_analysis(
    G: nx.Graph,
    cells_df: pd.DataFrame,
    region_labels: Dict[str, int],
    distance_cache: Dict[Tuple[str, str], float],
    out_dir: str,
    filename_prefix: str,
) -> Dict[str, str]:
    """
    그래프와 분석 결과를 중간 저장.

    Returns:
        저장된 파일 경로들
    """
    logger.info("그래프와 분석 결과 중간 저장 중...")
    start_time = time()

    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    try:
        # 1. NetworkX 그래프 저장 (pickle)
        graph_path = output_path / f"{filename_prefix}_graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        saved_files["graph"] = str(graph_path)
        logger.info(f"그래프 저장: {graph_path.name}")

        # 2. 거리 캐시 저장 (pickle)
        cache_path = output_path / f"{filename_prefix}_distance_cache.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump(distance_cache, f)
        saved_files["distance_cache"] = str(cache_path)
        logger.info(f"거리 캐시 저장: {cache_path.name}")

        # 3. 엣지 리스트 CSV 저장 (분석용)
        edges_data = []
        for u, v, data in G.edges(data=True):
            u_coords = (G.nodes[u]["lat"], G.nodes[u]["lon"])
            v_coords = (G.nodes[v]["lat"], G.nodes[v]["lon"])
            edges_data.append(
                {
                    "cell_a": u,
                    "cell_b": v,
                    "cell_a_lat": u_coords[0],
                    "cell_a_lon": u_coords[1],
                    "cell_b_lat": v_coords[0],
                    "cell_b_lon": v_coords[1],
                    "distance_m": data.get("distance_m", 0),
                    "region_a": region_labels.get(u, -1),
                    "region_b": region_labels.get(v, -1),
                    "is_intra_region": region_labels.get(u, -1)
                    == region_labels.get(v, -1)
                    and region_labels.get(u, -1) >= 0,
                }
            )

        edges_df = pd.DataFrame(edges_data)
        edges_path = output_path / f"{filename_prefix}_edges.csv"
        edges_df.to_csv(edges_path, index=False, encoding="utf-8")
        saved_files["edges"] = str(edges_path)
        logger.info(f"엣지 리스트 저장: {edges_path.name} ({len(edges_df)}개 엣지)")

        # 4. 노드 통계 CSV 저장
        nodes_data = []
        for node_id, data in G.nodes(data=True):
            neighbors = list(G.neighbors(node_id))
            region_id = region_labels.get(node_id, -1)
            nodes_data.append(
                {
                    "cell_id": node_id,
                    "cell_lat": data["lat"],
                    "cell_lon": data["lon"],
                    "point_count": data.get("point_count", 1),
                    "region_id": region_id,
                    "degree": G.degree(node_id),
                    "neighbors_count": len(neighbors),
                    "is_isolated": len(neighbors) == 0,
                    "is_orphan": region_id == -1,
                }
            )

        nodes_df = pd.DataFrame(nodes_data)
        nodes_path = output_path / f"{filename_prefix}_nodes.csv"
        nodes_df.to_csv(nodes_path, index=False, encoding="utf-8")
        saved_files["nodes"] = str(nodes_path)
        logger.info(f"노드 통계 저장: {nodes_path.name} ({len(nodes_df)}개 노드)")

        # 5. 권역 통계 JSON 저장
        region_stats = {}
        for region_id in set(region_labels.values()):
            region_cells = [
                cid for cid, rid in region_labels.items() if rid == region_id
            ]
            if region_cells:
                region_nodes = [cid for cid in region_cells if cid in G.nodes]
                subgraph = G.subgraph(region_nodes)

                region_stats[str(region_id)] = {
                    "cell_count": len(region_cells),
                    "connected_cells": len(region_nodes),
                    "edges_count": subgraph.number_of_edges(),
                    "is_connected": nx.is_connected(subgraph)
                    if len(region_nodes) > 1
                    else True,
                    "diameter_m": calculate_region_diameter(G, region_nodes)
                    if len(region_nodes) > 1
                    else 0,
                    "is_orphan": region_id == -1,
                }

        stats_path = output_path / f"{filename_prefix}_region_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(region_stats, f, ensure_ascii=False, indent=2)
        saved_files["region_stats"] = str(stats_path)
        logger.info(f"권역 통계 저장: {stats_path.name}")

        elapsed = time() - start_time
        logger.info(f"중간 저장 완료 ({elapsed:.2f}s): {len(saved_files)}개 파일")

        return saved_files

    except Exception as e:
        logger.error(f"중간 저장 실패: {e}")
        return {}


def attach_anchor_labels(
    cells_df: pd.DataFrame,
    region_labels: Dict[str, int],
    anchors_df: Optional[pd.DataFrame] = None,
    anchor_lat_col: str = "anchor_lat",
    anchor_lon_col: str = "anchor_lon",
) -> pd.DataFrame:
    """(선택) 가장 가까운 앵커 id를 각 셀에 부여."""
    logger.info("앵커 라벨 부여 중...")
    start_time = time()

    cells_df = cells_df.copy()
    cells_df["region_id"] = cells_df["cell_id"].map(region_labels)

    if anchors_df is None or anchors_df.empty:
        cells_df["anchor_id"] = None
        logger.info("앵커 데이터가 없어 건너뜀")
        return cells_df

    logger.info(f"{len(anchors_df)}개 앵커를 사용하여 최근접 앵커 계산 중...")
    coords = list(anchors_df[[anchor_lat_col, anchor_lon_col]].to_records(index=False))

    def nearest_anchor(lat: float, lon: float) -> int:
        best_idx, best_d = -1, float("inf")
        for idx, (alat, alon) in enumerate(coords):
            d = haversine_m(lat, lon, float(alat), float(alon))
            if d < best_d:
                best_d, best_idx = d, idx
        return best_idx

    tqdm.pandas(desc="앵커 라벨 부여")
    cells_df["anchor_id"] = cells_df.progress_apply(
        lambda r: nearest_anchor(r["cell_lat"], r["cell_lon"]), axis=1
    )

    elapsed = time() - start_time
    logger.info(f"앵커 라벨 부여 완료 ({elapsed:.2f}s)")
    return cells_df


def cells_to_geojson(cells_with_regions: pd.DataFrame, out_path: str) -> str:
    """각 H3 셀을 Polygon Feature로 내보내기 (region_id 및 음식점 정보 포함)."""
    features = []
    for _, r in cells_with_regions.iterrows():
        # H3 v3/v4 호환 경계 좌표
        if _HAS_V4:
            boundary_coords = h3.cell_to_boundary(r["cell_id"])
            # GeoJSON 형식으로 변환 (lon, lat 순서)
            boundary = [[lon, lat] for lat, lon in boundary_coords]
        else:
            boundary = h3.h3_to_geo_boundary(r["cell_id"], geo_json=True)

        # 기본 속성
        properties = {
            "cell_id": r["cell_id"],
            "region_id": int(r["region_id"]),
            "point_count": int(r["point_count"]),
            "anchor_id": None
            if pd.isna(r.get("anchor_id", None))
            else int(r["anchor_id"]),
        }

        # 음식점 정보 추가 (있는 경우)
        if "restaurant_count" in r and not pd.isna(r["restaurant_count"]):
            properties.update(
                {
                    "restaurant_count": int(r["restaurant_count"]),
                    "total_reviews": int(r.get("total_reviews", 0)),
                    "avg_rating": float(r.get("avg_rating", 0)),
                    "avg_bayesian_score": float(r.get("avg_bayesian_score", 0)),
                }
            )

        poly = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [boundary]},
            "properties": properties,
        }
        features.append(poly)

    fc = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    return out_path


# ---------------------------
# Seoul Walking Region Builder
# ---------------------------
def build_walking_regions(
    data_config: DataConfig,
    region_name: str = "서울특별시",
    resolution: int = 10,
    walking_threshold_m: float = 1000.0,
    max_region_distance_m: float = 2500.0,
    distance_metric: str = "osrm_then_haversine",
    osrm_base_url: str = "https://router.project-osrm.org",
    osrm_profile: str = "foot",
    osrm_timeout: float = 5.0,
    min_cells_per_region: int = 2,
    max_cells_per_region: Optional[int] = None,
    enable_orphan_reassign: bool = True,
    max_reassign_distance_m: Optional[float] = None,
    use_osrm_cache: bool = True,
    osrm_cache_dir: str = "osrm_cache",
    use_restaurant_data: bool = True,
    out_dir: Optional[str] = None,
    filename: str = None,
    kring: int = 1,
) -> pd.DataFrame:
    """음식점 추천용 도보 권역 생성"""
    logger.info("=" * 60)
    logger.info(f"{region_name} 음식점 추천용 도보 권역 생성 시작")
    logger.info("=" * 60)
    logger.info(f"대상 지역: {region_name}")
    logger.info(f"H3 해상도: {resolution}")
    logger.info(f"도보 거리 임계값: {walking_threshold_m}m")
    logger.info(f"최대 권역 거리: {max_region_distance_m}m")
    logger.info(f"거리 계산 방식: {distance_metric}")
    logger.info(f"최소 셀 수: {min_cells_per_region}")

    total_start = time()

    # OSRM 거리 캐시 초기화
    osrm_cache = None
    if use_osrm_cache and distance_metric in ["osrm", "osrm_then_haversine"]:
        safe_region_name = (
            region_name.replace("특별시", "").replace(",", "").replace(" ", "_")
        )
        osrm_cache = OSRMDistanceCache(
            cache_dir=osrm_cache_dir, region_name=safe_region_name
        )
        osrm_cache.load_cache()
        logger.info(f"OSRM 거리 캐시 활성화: {safe_region_name}")
    else:
        logger.info("OSRM 거리 캐시 비활성화")

    # 기본 파일명 설정
    if filename is None:
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        # 지역명에서 특수문자 제거하여 파일명에 사용
        safe_region_name = (
            region_name.replace("특별시", "").replace(",", "").replace(" ", "_")
        )
        filename = f"{safe_region_name}_walking_regions_{timestamp}"

    # 출력 디렉토리 생성
    if out_dir:
        output_path = Path(out_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"출력 디렉토리: {output_path.absolute()}")

    # 1. 지역 경계 획득
    region_boundary = get_region_boundary(region_name)

    # 2. 음식점 데이터 로드 (선택사항)
    restaurants_df = None
    if use_restaurant_data:
        restaurants_df = load_restaurant_data_with_dataloader(data_config, region_name)
        if not restaurants_df.empty:
            logger.info("🍽️ 음식점 데이터가 권역 생성에 반영됩니다!")
        else:
            logger.warning("음식점 데이터를 로드할 수 없어 기본 모드로 실행합니다.")
    else:
        logger.info("음식점 데이터가 제공되지 않아 기본 모드로 실행합니다.")

    # 3. 지역 전체를 H3로 커버
    region_cell_ids = flood_fill_region_cells(region_boundary, resolution, region_name)

    # 4. H3 셀 정보를 DataFrame으로 변환 (음식점 정보 포함)
    cells_df = region_cells_to_dataframe(region_cell_ids, restaurants_df, resolution)

    # 4. 도보 거리 기반 그래프 구성
    logger.info("도보 거리 기반 권역 그래프 구성 중...")
    G, distance_cache = build_cell_graph(
        cells_df,
        distance_threshold_m=walking_threshold_m,
        distance_metric=distance_metric,
        osrm_base_url=osrm_base_url,
        osrm_profile=osrm_profile,
        osrm_timeout=osrm_timeout,
        kring=kring,
        osrm_cache=osrm_cache,
    )

    # 5. 연결 요소 기반 권역 라벨링 (크기 제한 포함)
    logger.info("연결 요소 분석을 통한 권역 생성 중...")
    region_labels = label_regions_from_components(
        G,
        min_cells_per_region=min_cells_per_region,
        max_region_distance_m=max_region_distance_m,
        max_cells_per_region=max_cells_per_region,
    )

    # 5.5. 그래프 중간 저장 (수정 전 상태)
    if out_dir:
        logger.info("그래프 분석 결과 중간 저장 중...")
        saved_files = save_graph_analysis(
            G=G,
            cells_df=cells_df,
            region_labels=region_labels,
            distance_cache=distance_cache,
            out_dir=str(output_path),
            filename_prefix=f"{filename}_before_reassign",
        )
        logger.info(f"중간 저장 완료: {len(saved_files)}개 파일")

    # 6. 고아 셀 재할당 (새로운 기능)
    if enable_orphan_reassign:
        logger.info("고아 셀 재할당 처리 중...")
        original_orphan_count = len(
            [rid for rid in region_labels.values() if rid == -1]
        )
        if original_orphan_count > 0:
            reassign_distance = max_reassign_distance_m or (walking_threshold_m * 1.5)
            region_labels = reassign_orphan_cells(
                G=G,
                cells_df=cells_df,
                region_labels=region_labels,
                max_reassign_distance_m=reassign_distance,
                osrm_cache=osrm_cache,
            )
            final_orphan_count = len(
                [rid for rid in region_labels.values() if rid == -1]
            )
            logger.info(
                f"고아 셀 감소: {original_orphan_count} → {final_orphan_count}개"
            )
        else:
            logger.info("재할당할 고아 셀이 없습니다.")
    else:
        logger.info("고아 셀 재할당이 비활성화되어 있습니다.")

    # 7. 최종 결과 DataFrame 생성
    logger.info("최종 결과 생성 중...")
    result_df = cells_df.copy()
    result_df["region_id"] = result_df["cell_id"].map(region_labels)

    # 8. 파일 저장 및 최종 그래프 저장
    if out_dir:
        logger.info("결과 파일 저장 중...")
        csv_path = output_path / f"{filename}.csv"
        geojson_path = output_path / f"{filename}.geojson"
        result_df.to_csv(csv_path, index=False, encoding="utf-8")
        cells_to_geojson(result_df, str(geojson_path))
        logger.info(f"저장 완료: {csv_path.name}, {geojson_path.name}")

        # 최종 그래프 상태 저장 (재할당 후)
        final_saved_files = save_graph_analysis(
            G=G,
            cells_df=cells_df,
            region_labels=region_labels,
            distance_cache=distance_cache,
            out_dir=str(output_path),
            filename_prefix=f"{filename}_final",
        )
        logger.info(f"최종 그래프 저장 완료: {len(final_saved_files)}개 파일")

    # 캐시 저장
    if osrm_cache:
        osrm_cache.save_cache()

    # 8. 결과 통계
    n_cells = len(result_df)
    n_regions = result_df["region_id"].nunique()
    if -1 in result_df["region_id"].unique():
        n_regions -= 1  # 노이즈 제외
    noise_cells = len(result_df[result_df["region_id"] == -1])
    valid_regions = result_df[result_df["region_id"] >= 0]
    avg_cells_per_region = len(valid_regions) / n_regions if n_regions > 0 else 0

    # 음식점 통계 (있는 경우)
    restaurant_stats = {}
    if "restaurant_count" in result_df.columns:
        restaurant_stats = {
            "total_restaurants": result_df["restaurant_count"].sum(),
            "cells_with_restaurants": (result_df["restaurant_count"] > 0).sum(),
            "avg_restaurants_per_cell": result_df["restaurant_count"].mean(),
            "avg_rating": result_df[result_df["avg_rating"] > 0]["avg_rating"].mean()
            if (result_df["avg_rating"] > 0).any()
            else 0,
        }

    total_elapsed = time() - total_start
    logger.info("=" * 60)
    logger.info(f"{region_name} 도보 권역 생성 완료!")
    logger.info(f"총 H3 셀: {n_cells}개")
    logger.info(f"생성된 권역: {n_regions}개")
    logger.info(f"노이즈 셀: {noise_cells}개")
    logger.info(f"권역당 평균 셀 수: {avg_cells_per_region:.1f}개")

    # 음식점 통계 출력
    if restaurant_stats:
        logger.info(f"🍽️ 총 음식점: {restaurant_stats['total_restaurants']:,}개")
        logger.info(
            f"🍽️ 음식점이 있는 셀: {restaurant_stats['cells_with_restaurants']:,}개"
        )
        logger.info(
            f"🍽️ 셀당 평균 음식점: {restaurant_stats['avg_restaurants_per_cell']:.1f}개"
        )
        if restaurant_stats["avg_rating"] > 0:
            logger.info(f"⭐ 평균 평점: {restaurant_stats['avg_rating']:.2f}")

    logger.info(f"총 소요시간: {total_elapsed:.2f}s")
    logger.info("=" * 60)

    return result_df
