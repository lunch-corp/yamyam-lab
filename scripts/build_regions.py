#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🗺️ 서울시 음식점 추천용 도보 권역 생성 CLI 스크립트

이 스크립트는 yamyam-lab 프로젝트의 region 모듈을 사용하여
음식점 추천에 적합한 도보 권역을 생성합니다.

사용 예시:
  # 🍽️ 음식점 데이터 포함 - 영등포구 테스트 (추천!)
  python scripts/build_regions.py --region "영등포구" --out_dir data/processed/regions

  # 더 작은 권역으로 분할 (1km 제한)
  python scripts/build_regions.py --region "영등포구" --max_region_distance_m 1000

  # 음식점 데이터 없이 실행
  python scripts/build_regions.py --region "영등포구" --no_restaurant_data

  # 서울시 전체 (시간 오래 걸림)
  python scripts/build_regions.py --out_dir data/processed/regions

작성자: yamyam-lab
버전: 3.0 (yamyam-lab 프로젝트 통합)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from yamyam_lab.data.config import DataConfig  # noqa: E402
from yamyam_lab.preprocess.region import build_walking_regions  # noqa: E402
from yamyam_lab.tools.config import load_yaml  # noqa: E402

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / "logs" / "build_regions.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """CLI 메인 함수"""
    ap = argparse.ArgumentParser(
        description="음식점 추천용 도보 권역 생성 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 🍽️ 음식점 데이터 포함 - 영등포구 테스트 (추천!)
  python scripts/build_regions.py --region "영등포구" --out_dir data/processed/regions

  # 더 작은 권역으로 분할 (1km 제한)
  python scripts/build_regions.py --region "영등포구" --max_region_distance_m 1000

  # 음식점 데이터 없이 실행
  python scripts/build_regions.py --region "영등포구" --no_restaurant_data

  # 서울시 전체 (시간 오래 걸림)
  python scripts/build_regions.py --out_dir data/processed/regions

  # 빠른 테스트용 추천 지역들:
  - 영등포구: 작고 테스트하기 좋음 (추천!)
  - 중구: 작은 지역, 빠른 실행
  - 성동구: 중간 크기
  - 강남구: 상대적으로 큰 지역
        """,
    )

    # 기본 설정
    ap.add_argument(
        "--region",
        default="서울특별시",
        help="Target region (default: 서울특별시). Examples: 영등포구, 강남구, 중구",
    )
    ap.add_argument(
        "--config",
        default="config/preprocess/region.yaml",
        help="Configuration file path (default: config/preprocess/region.yaml)",
    )
    ap.add_argument(
        "--data_config",
        default="config/data/dataset.yaml",
        help="Data configuration file path (default: config/data/dataset.yaml)",
    )

    # 음식점 데이터 설정
    ap.add_argument(
        "--no_restaurant_data",
        action="store_true",
        help="Disable restaurant data usage (create regions without restaurant info)",
    )

    # H3 및 거리 설정
    ap.add_argument(
        "--resolution", type=int, default=10, help="H3 resolution (default: 10)"
    )
    ap.add_argument(
        "--threshold_m",
        type=float,
        default=500.0,
        help="Walking distance threshold in meters (default: 500)",
    )
    ap.add_argument(
        "--max_region_distance_m",
        type=float,
        default=2000.0,
        help="Maximum region diameter in meters. Large regions will be split (default: 2000)",
    )

    # 거리 계산 설정
    ap.add_argument(
        "--distance_metric",
        choices=["osrm", "haversine", "osrm_then_haversine"],
        default="osrm_then_haversine",
        help="Distance calculation method",
    )
    ap.add_argument(
        "--osrm_base", default="https://router.project-osrm.org", help="OSRM server URL"
    )
    ap.add_argument(
        "--osrm_profile", default="foot", help="OSRM profile (foot/driving/cycling)"
    )
    ap.add_argument(
        "--osrm_timeout", type=float, default=5.0, help="OSRM request timeout (seconds)"
    )

    # 권역 크기 설정
    ap.add_argument(
        "--min_cells", type=int, default=2, help="Minimum cells per region (default: 2)"
    )
    ap.add_argument(
        "--max_cells",
        type=int,
        default=None,
        help="Maximum cells per region (optional)",
    )

    # 고아 셀 재할당 설정
    ap.add_argument(
        "--reassign_orphans",
        action="store_true",
        default=True,
        help="Enable orphan cells reassignment to nearest regions (default: True)",
    )
    ap.add_argument(
        "--no_reassign_orphans",
        action="store_true",
        help="Disable orphan cells reassignment (keep as noise)",
    )
    ap.add_argument(
        "--max_reassign_distance_m",
        type=float,
        default=None,
        help="Maximum distance for orphan reassignment (default: 1.5 * threshold_m)",
    )

    # 캐시 설정
    ap.add_argument(
        "--use_osrm_cache",
        action="store_true",
        default=True,
        help="Use OSRM distance cache for faster computation (default: True)",
    )
    ap.add_argument(
        "--no_osrm_cache",
        action="store_true",
        help="Disable OSRM distance cache (always use API)",
    )
    ap.add_argument(
        "--osrm_cache_dir",
        default="data/cache/osrm",
        help="Directory to store OSRM distance cache files (default: data/cache/osrm)",
    )

    # 출력 설정
    ap.add_argument(
        "--out_dir",
        default="data/processed/regions",
        help="Output directory to save CSV/GeoJSON files (default: data/processed/regions)",
    )
    ap.add_argument(
        "--filename", default=None, help="Base filename (default: auto-generated)"
    )

    # 성능 설정
    ap.add_argument(
        "--kring",
        type=int,
        default=1,
        help="H3 k-ring neighbor depth (>=1 recommended)",
    )

    # 기타
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = ap.parse_args()

    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # 설정 파일 로드
        config_path = project_root / args.config
        if config_path.exists():
            load_yaml(str(config_path))
            logger.info(f"Region 설정 로드: {config_path}")
        else:
            logger.info("기본 설정 사용 (설정 파일 없음)")

        # 데이터 설정 로드
        data_config_path = project_root / args.data_config
        if data_config_path.exists():
            data_config = DataConfig.from_yaml(str(data_config_path))
            logger.info(f"데이터 설정 로드: {data_config_path}")
        else:
            # Fallback to dataset.yaml if provided path does not exist
            fallback_path = project_root / "config/data/dataset.yaml"
            if fallback_path.exists():
                data_config = DataConfig.from_yaml(str(fallback_path))
                logger.warning(
                    f"지정한 설정이 없어 dataset.yaml로 폴백합니다: {fallback_path}"
                )
            else:
                logger.error(
                    f"데이터 설정 파일을 찾을 수 없습니다: {data_config_path} 또는 {fallback_path}"
                )
                sys.exit(1)

        # 출력 디렉토리 설정 (프로젝트 루트 기준)
        out_dir = project_root / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # 캐시 디렉토리 설정 (프로젝트 루트 기준)
        cache_dir = project_root / args.osrm_cache_dir

        # 지역별 도보 권역 생성
        if args.region == "서울특별시":
            logger.info("서울시 전체 음식점 추천용 도보 권역 생성을 시작합니다...")
        else:
            logger.info(
                f"테스트 모드: {args.region} 음식점 추천용 도보 권역 생성을 시작합니다..."
            )
            logger.info("💡 테스트 모드로 빠른 실행이 가능합니다!")

        # 고아 셀 재할당 설정
        enable_reassign = args.reassign_orphans and not args.no_reassign_orphans
        max_reassign_dist = args.max_reassign_distance_m or (args.threshold_m * 1.5)

        # OSRM 캐시 설정
        use_cache = args.use_osrm_cache and not args.no_osrm_cache

        # 음식점 데이터 사용 설정
        use_restaurant_data = not args.no_restaurant_data

        logger.info(f"고아 셀 재할당: {'활성화' if enable_reassign else '비활성화'}")
        if enable_reassign:
            logger.info(f"재할당 최대 거리: {max_reassign_dist:.0f}m")
        logger.info(f"OSRM 거리 캐시: {'활성화' if use_cache else '비활성화'}")
        if use_cache:
            logger.info(f"캐시 디렉토리: {cache_dir}")
        logger.info(
            f"음식점 데이터 사용: {'활성화' if use_restaurant_data else '비활성화'}"
        )

        # 로그 디렉토리 생성
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)

        result_df = build_walking_regions(
            data_config=data_config,
            region_name=args.region,
            resolution=args.resolution,
            walking_threshold_m=args.threshold_m,
            max_region_distance_m=args.max_region_distance_m,
            distance_metric=args.distance_metric,
            osrm_base_url=args.osrm_base,
            osrm_profile=args.osrm_profile,
            osrm_timeout=args.osrm_timeout,
            min_cells_per_region=args.min_cells,
            max_cells_per_region=args.max_cells,
            enable_orphan_reassign=enable_reassign,
            max_reassign_distance_m=max_reassign_dist,
            use_osrm_cache=use_cache,
            osrm_cache_dir=str(cache_dir),
            use_restaurant_data=use_restaurant_data,
            out_dir=str(out_dir),
            filename=args.filename,
            kring=args.kring,
        )

        # 콘솔 결과 요약
        print("\n" + "=" * 60)
        print(f"🗺️ {args.region} 도보 권역 생성 결과 요약")
        print("=" * 60)

        n_cells = len(result_df)
        n_regions = result_df["region_id"].nunique()
        if -1 in result_df["region_id"].unique():
            n_regions -= 1  # 노이즈 제외
        noise_cells = len(result_df[result_df["region_id"] == -1])
        valid_regions = result_df[result_df["region_id"] >= 0]
        avg_cells_per_region = len(valid_regions) / n_regions if n_regions > 0 else 0

        print(f"🏙️ 대상 지역: {args.region}")
        print(f"📍 H3 해상도: {args.resolution}")
        print(f"🚶 도보 거리 임계값: {args.threshold_m}m")
        print(f"📏 최대 권역 거리: {args.max_region_distance_m}m")
        print(f"🔢 총 H3 셀: {n_cells:,}개")
        print(f"🏘️ 생성된 권역: {n_regions:,}개")
        print(f"📊 권역당 평균 셀 수: {avg_cells_per_region:.1f}개")
        print(f"🗑️ 노이즈 셀: {noise_cells:,}개")
        print(f"✅ 커버리지: {((n_cells - noise_cells) / n_cells * 100):.1f}%")

        # 음식점 통계 출력
        if "restaurant_count" in result_df.columns:
            total_restaurants = result_df["restaurant_count"].sum()
            cells_with_restaurants = (result_df["restaurant_count"] > 0).sum()
            avg_restaurants_per_cell = result_df["restaurant_count"].mean()
            avg_rating = (
                result_df[result_df["avg_rating"] > 0]["avg_rating"].mean()
                if (result_df["avg_rating"] > 0).any()
                else 0
            )

            print(f"🍽️ 총 음식점: {total_restaurants:,}개")
            print(
                f"🍽️ 음식점이 있는 셀: {cells_with_restaurants:,}개 ({cells_with_restaurants / n_cells * 100:.1f}%)"
            )
            print(f"🍽️ 셀당 평균 음식점: {avg_restaurants_per_cell:.1f}개")
            if avg_rating > 0:
                print(f"⭐ 평균 평점: {avg_rating:.2f}")

        # 권역 크기 분포
        if n_regions > 0:
            region_sizes = valid_regions.groupby("region_id").size()
            print(f"📏 권역 크기 범위: {region_sizes.min()}-{region_sizes.max()}셀")
            print(f"📈 권역 크기 중앙값: {region_sizes.median():.0f}셀")

        print("=" * 60)
        print("🎯 음식점 추천 시스템에서 활용 가능한 도보 권역이 생성되었습니다!")

        # 테스트 모드 안내
        if args.region != "서울특별시":
            print(
                "💡 테스트가 완료되면 --region '서울특별시'로 전체 권역을 생성하세요!"
            )

        print(f"📁 결과 파일 위치: {out_dir}")
        print("=" * 60)

        logger.info(f"{args.region} 도보 권역 생성이 성공적으로 완료되었습니다!")

    except Exception as e:
        logger.error(f"권역 생성 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
