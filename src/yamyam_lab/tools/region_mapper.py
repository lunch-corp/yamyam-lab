# -*- coding: utf-8 -*-
"""
🗺️ 사용자 좌표를 기반으로 지역 클러스터링 정보를 활용한 음식점 추천 범위 한정 모듈

이 모듈은 사용자의 위치 좌표를 받아서 해당 지역의 음식점들을 추천하기 위한
지역 매핑 기능을 제공합니다.

주요 기능:
- 📍 사용자 좌표를 H3 cell ID로 변환
- 🗺️ 지역 클러스터링 CSV 파일들을 통합 관리
- 🎯 좌표 기반 region_id 검색
- 🍽️ region_id 기반 음식점 목록 반환

작성자: yamyam-lab
버전: 1.0
"""

import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import h3
import pandas as pd

logger = logging.getLogger(__name__)

# H3 v3/v4 호환성 어댑터
_HAS_V4 = hasattr(h3, "latlng_to_cell")
project_root = os.path.join(os.path.dirname(__file__), "..", "..")


def h3_geo_to_cell(lat: float, lon: float, res: int = 10) -> str:
    """H3 v3/v4 호환 좌표→셀 변환

    Args:
        lat: 위도
        lon: 경도
        res: H3 해상도 (기본값: 10)

    Returns:
        H3 cell ID 문자열
    """
    return h3.latlng_to_cell(lat, lon, res) if _HAS_V4 else h3.geo_to_h3(lat, lon, res)


def h3_grid_ring(cell_id: str, k: int):
    """H3 v3/v4 호환 링 셀들 반환"""
    if _HAS_V4:
        return h3.grid_ring(cell_id, k)
    else:
        return h3.k_ring(cell_id, k) - h3.k_ring(cell_id, k - 1) if k > 0 else {cell_id}


def h3_grid_disk(cell_id: str, k: int):
    """H3 v3/v4 호환 디스크 셀들 반환"""
    if _HAS_V4:
        return h3.grid_disk(cell_id, k)
    else:
        return h3.k_ring(cell_id, k)


class RegionMapper:
    """지역 매핑 클래스"""

    def __init__(
        self,
        regions_dir: str = os.path.join(project_root, "data/processed/regions"),
    ):
        """
        Args:
            regions_dir: 지역 클러스터링 CSV 파일들이 있는 디렉토리 경로
        """
        self.regions_dir = Path(regions_dir)
        self.regions_df: Optional[pd.DataFrame] = None
        self.diner_df: Optional[pd.DataFrame] = None
        self.region_diner_mapping: Optional[pd.DataFrame] = None

    def load_regions_data(self) -> pd.DataFrame:
        """모든 지역 CSV 파일들을 로드하고 concat"""
        if self.regions_df is not None:
            return self.regions_df

        logger.info(f"지역 데이터 로딩 시작: {self.regions_dir}")

        # 지역 CSV 파일들 찾기
        csv_files = glob.glob(str(self.regions_dir / "*_walking_regions_*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"지역 CSV 파일을 찾을 수 없습니다: {self.regions_dir}"
            )

        logger.info(f"발견된 지역 파일들: {csv_files}")

        # 모든 CSV 파일 로드 및 concat
        region_dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            region_name = Path(csv_file).stem.split("_walking_regions_")[0]
            df["region_name"] = region_name
            region_dfs.append(df)
            logger.info(f"로드 완료: {region_name} - {len(df)} rows")

        regions_df = pd.concat(region_dfs, ignore_index=True)

        self.regions_df = regions_df.drop_duplicates(subset=["cell_id"])
        logger.info(f"전체 지역 데이터 로드 완료: {len(self.regions_df)} rows")

        return self.regions_df

    def load_diner_data(
        self,
        diner_csv_path: str = os.path.join(project_root, "data/diner.csv"),
    ) -> pd.DataFrame:
        """음식점 데이터 로드"""
        if self.diner_df is not None:
            return self.diner_df

        logger.info(f"음식점 데이터 로딩 시작: {diner_csv_path}")
        self.diner_df = pd.read_csv(diner_csv_path)
        logger.info(f"음식점 데이터 로드 완료: {len(self.diner_df)} rows")

        return self.diner_df

    def find_region_id(self, lat: float, lon: float) -> int:
        """좌표를 받아 해당하는 region_id 반환

        Args:
            lat: 위도
            lon: 경도

        Returns:
            region_id (int), 찾을 수 없으면 -1
        """
        # 지역 데이터가 로드되지 않았다면 로드
        if self.regions_df is None:
            self.load_regions_data()

        # H3 cell ID 계산
        cell_id = h3_geo_to_cell(lat, lon, res=10)

        # region_id 찾기
        region_info = self.regions_df[self.regions_df["cell_id"] == cell_id]

        if region_info.empty:
            logger.warning(
                f"좌표 ({lat}, {lon})에 해당하는 지역을 찾을 수 없습니다. cell_id: {cell_id}"
            )
            return -1

        region_id = region_info["region_id"].iloc[0]
        logger.debug(
            f"좌표 ({lat}, {lon}) -> cell_id: {cell_id} -> region_id: {region_id}"
        )

        return int(region_id)

    def find_nearest_region_id(
        self, lat: float, lon: float, max_distance: int = 5
    ) -> int:
        """좌표를 받아 가장 가까운 region_id 반환 (fallback 메서드)

        Args:
            lat: 위도
            lon: 경도
            max_distance: 최대 검색 거리 (H3 셀 단위, 기본값: 5)

        Returns:
            가장 가까운 region_id (int), 찾을 수 없으면 -1
        """
        # 지역 데이터가 로드되지 않았다면 로드
        if self.regions_df is None:
            self.load_regions_data()

        # 현재 위치의 H3 cell ID
        center_cell_id = h3_geo_to_cell(lat, lon, res=10)

        # 거리별로 점진적으로 확장하며 검색
        for distance in range(max_distance + 1):
            if distance == 0:
                # 중심 셀만 확인
                search_cells = [center_cell_id]
            else:
                # 거리 k의 링 셀들 확인
                try:
                    search_cells = list(h3_grid_ring(center_cell_id, distance))
                except Exception:
                    # 링 생성에 실패하면 (경계 등) 건너뛰기
                    continue

            # 각 셀에 대해 매핑된 region이 있는지 확인
            for cell_id in search_cells:
                region_info = self.regions_df[self.regions_df["cell_id"] == cell_id]
                if not region_info.empty:
                    region_id = int(region_info["region_id"].iloc[0])
                    logger.info(
                        f"가장 가까운 지역 발견: 거리 {distance}, "
                        f"좌표 ({lat}, {lon}) -> region_id: {region_id}"
                    )
                    return region_id

        logger.warning(
            f"최대 거리 {max_distance} 내에서 매핑된 지역을 찾을 수 없습니다. "
            f"좌표: ({lat}, {lon})"
        )
        return -1

    def find_region_id_with_fallback(
        self, lat: float, lon: float, use_fallback: bool = True, max_distance: int = 5
    ) -> int:
        """좌표를 받아 region_id 반환 (fallback 포함)

        Args:
            lat: 위도
            lon: 경도
            use_fallback: fallback 사용 여부
            max_distance: fallback 시 최대 검색 거리

        Returns:
            region_id (int), 찾을 수 없으면 -1
        """
        # 먼저 정확한 매핑 시도
        region_id = self.find_region_id(lat, lon)

        # 매핑이 실패하고 fallback이 활성화된 경우
        if region_id == -1 and use_fallback:
            logger.info(f"정확한 매핑 실패, 가장 가까운 지역 검색 시작: ({lat}, {lon})")
            region_id = self.find_nearest_region_id(lat, lon, max_distance)

        return region_id

    def create_region_diner_mapping(
        self, output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """region_id와 diner_idx 매핑 데이터 생성

        Args:
            output_path: 결과를 저장할 CSV 파일 경로 (옵션)

        Returns:
            region_id와 diner_idx가 매핑된 DataFrame
        """
        # 필요한 데이터 로드
        if self.regions_df is None:
            self.load_regions_data()
        if self.diner_df is None:
            self.load_diner_data()

        logger.info("region_id와 diner_idx 매핑 생성 시작")

        # 음식점들의 H3 cell ID 계산
        diner_cells = []
        for idx, row in self.diner_df.iterrows():
            if pd.notna(row["diner_lat"]) and pd.notna(row["diner_lon"]):
                cell_id = h3_geo_to_cell(row["diner_lat"], row["diner_lon"], res=10)
                diner_cells.append(
                    {
                        "diner_idx": row["diner_idx"],
                        "cell_id": cell_id,
                        "diner_lat": row["diner_lat"],
                        "diner_lon": row["diner_lon"],
                    }
                )

        diner_cells_df = pd.DataFrame(diner_cells)
        logger.info(f"음식점 cell_id 계산 완료: {len(diner_cells_df)} 개")

        # regions_df와 조인하여 region_id 매핑
        mapping_df = diner_cells_df.merge(
            self.regions_df[["cell_id", "region_id", "region_name"]],
            on="cell_id",
            how="left",
        )

        # region_id가 없는 경우 -1로 설정
        mapping_df["region_id"] = mapping_df["region_id"].fillna(-1).astype(int)

        # 결과 정리
        self.region_diner_mapping = mapping_df[
            ["diner_idx", "region_id", "region_name", "cell_id"]
        ]

        logger.info(f"매핑 완료: {len(self.region_diner_mapping)} 개 음식점")
        logger.info(
            f"매핑된 음식점: {len(self.region_diner_mapping[self.region_diner_mapping['region_id'] != -1])} 개"
        )
        logger.info(
            f"매핑되지 않은 음식점: {len(self.region_diner_mapping[self.region_diner_mapping['region_id'] == -1])} 개"
        )

        # 파일로 저장 (옵션)
        if output_path:
            self.region_diner_mapping.to_csv(output_path, index=False)
            logger.info(f"매핑 데이터 저장 완료: {output_path}")

        return self.region_diner_mapping

    def get_diners_by_coordinates(
        self, lat: float, lon: float, use_fallback: bool = True, max_distance: int = 5
    ) -> List[int]:
        """좌표를 받아 해당 지역의 음식점 목록 반환

        Args:
            lat: 위도
            lon: 경도
            use_fallback: fallback 사용 여부 (가장 가까운 지역 찾기)
            max_distance: fallback 시 최대 검색 거리

        Returns:
            해당 지역의 diner_idx 목록
        """
        # region_id 찾기 (fallback 포함)
        region_id = self.find_region_id_with_fallback(
            lat, lon, use_fallback, max_distance
        )

        if region_id == -1:
            logger.warning(f"좌표 ({lat}, {lon})에 해당하는 지역을 찾을 수 없습니다.")
            return []

        return self.get_diners_by_region_id(region_id)

    def get_diners_by_region_id(self, region_id: int) -> List[int]:
        """region_id를 받아 해당 지역의 음식점 목록 반환

        Args:
            region_id: 지역 ID

        Returns:
            해당 지역의 diner_idx 목록
        """
        # 매핑 데이터가 없다면 생성
        if self.region_diner_mapping is None:
            self.create_region_diner_mapping()

        # 해당 지역의 음식점들 찾기
        region_diners = self.region_diner_mapping[
            self.region_diner_mapping["region_id"] == region_id
        ]

        diner_indices = region_diners["diner_idx"].tolist()
        logger.info(f"region_id {region_id}에서 {len(diner_indices)}개 음식점 발견")

        return diner_indices

    def get_region_info(
        self, lat: float, lon: float, use_fallback: bool = True, max_distance: int = 5
    ) -> Dict:
        """좌표에 대한 상세 지역 정보 반환

        Args:
            lat: 위도
            lon: 경도
            use_fallback: fallback 사용 여부 (가장 가까운 지역 찾기)
            max_distance: fallback 시 최대 검색 거리

        Returns:
            지역 정보 딕셔너리
        """
        if self.regions_df is None:
            self.load_regions_data()

        # 먼저 정확한 매핑 시도
        cell_id = h3_geo_to_cell(lat, lon, res=10)
        region_info = self.regions_df[self.regions_df["cell_id"] == cell_id]

        # 정확한 매핑이 없고 fallback이 활성화된 경우
        if region_info.empty and use_fallback:
            region_id = self.find_nearest_region_id(lat, lon, max_distance)
            if region_id != -1:
                # 가장 가까운 지역의 정보 가져오기
                region_info = self.regions_df[self.regions_df["region_id"] == region_id]
                if not region_info.empty:
                    row = region_info.iloc[0]
                    result = {
                        "region_id": int(row["region_id"]),
                        "cell_id": cell_id,  # 원래 좌표의 cell_id 유지
                        "region_name": row.get("region_name", "Unknown"),
                        "restaurant_count": int(row.get("restaurant_count", 0)),
                        "total_reviews": int(row.get("total_reviews", 0)),
                        "avg_rating": float(row.get("avg_rating", 0.0)),
                        "is_fallback": True,  # fallback으로 찾았음을 표시
                        "original_cell_id": cell_id,
                        "matched_cell_id": row["cell_id"],
                    }
                    return result

        # 정확한 매핑이 있는 경우
        if not region_info.empty:
            row = region_info.iloc[0]
            return {
                "region_id": int(row["region_id"]),
                "cell_id": cell_id,
                "region_name": row.get("region_name", "Unknown"),
                "restaurant_count": int(row.get("restaurant_count", 0)),
                "total_reviews": int(row.get("total_reviews", 0)),
                "avg_rating": float(row.get("avg_rating", 0.0)),
                "is_fallback": False,
            }

        # 매핑을 찾을 수 없는 경우
        return {
            "region_id": -1,
            "cell_id": cell_id,
            "region_name": None,
            "restaurant_count": 0,
            "total_reviews": 0,
            "avg_rating": 0.0,
            "is_fallback": False,
        }


# 편의 함수들
def find_region_by_coordinates(
    lat: float,
    lon: float,
    regions_dir: str = os.path.join(project_root, "data/processed/regions"),
    use_fallback: bool = True,
    max_distance: int = 5,
) -> int:
    """좌표로 region_id 찾기 (간단한 버전)

    Args:
        lat: 위도
        lon: 경도
        regions_dir: 지역 데이터 디렉토리
        use_fallback: fallback 사용 여부
        max_distance: fallback 시 최대 검색 거리
    """
    mapper = RegionMapper(regions_dir)
    return mapper.find_region_id_with_fallback(lat, lon, use_fallback, max_distance)


def get_nearby_restaurants(
    lat: float,
    lon: float,
    regions_dir: str = os.path.join(project_root, "data/processed/regions"),
    use_fallback: bool = True,
    max_distance: int = 5,
) -> List[int]:
    """좌표로 근처 음식점 목록 가져오기 (간단한 버전)

    Args:
        lat: 위도
        lon: 경도
        regions_dir: 지역 데이터 디렉토리
        use_fallback: fallback 사용 여부
        max_distance: fallback 시 최대 검색 거리
    """
    mapper = RegionMapper(regions_dir)
    return mapper.get_diners_by_coordinates(lat, lon, use_fallback, max_distance)
