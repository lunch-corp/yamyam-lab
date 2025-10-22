#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🗺️ Region-Diner 매핑 데이터 생성 CLI 스크립트

region_id와 diner_idx 매핑 데이터를 생성하여 CSV로 저장합니다.
이 파일은 추후 DB화할 때 사용됩니다.

사용 예시:
  # 기본 실행 (전체 데이터)
  python scripts/create_region_diner_mapping.py

  # 특정 지역만 매핑
  python scripts/create_region_diner_mapping.py --region "영등포구"

  # 출력 파일 지정
  python scripts/create_region_diner_mapping.py --output data/processed/custom_mapping.csv

  # 상세 로그 출력
  python scripts/create_region_diner_mapping.py --verbose

작성자: yamyam-lab
버전: 1.0
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.yamyam_lab.tools.region_mapper import RegionMapper  # noqa: E402

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(project_root / "logs" / "create_region_diner_mapping.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """CLI 메인 함수"""
    ap = argparse.ArgumentParser(
        description="Region-Diner 매핑 데이터 생성 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 실행 (전체 데이터)
  python scripts/create_region_diner_mapping.py

  # 출력 파일 지정
  python scripts/create_region_diner_mapping.py --output data/processed/custom_mapping.csv

  # 다른 지역 데이터 디렉토리 사용
  python scripts/create_region_diner_mapping.py --regions_dir data/processed/regions_test

  # 다른 음식점 데이터 사용
  python scripts/create_region_diner_mapping.py --diner_csv data/diner_test.csv

  # 상세 로그 출력
  python scripts/create_region_diner_mapping.py --verbose

  # 매핑 통계만 출력 (저장하지 않음)
  python scripts/create_region_diner_mapping.py --dry_run
        """,
    )

    # 기본 설정
    ap.add_argument(
        "--output",
        default="data/processed/region_diner_mapping.csv",
        help="Output CSV file path (default: data/processed/region_diner_mapping.csv)",
    )
    ap.add_argument(
        "--regions_dir",
        default="data/processed/regions",
        help="Directory containing region CSV files (default: data/processed/regions)",
    )
    ap.add_argument(
        "--diner_csv",
        default="data/diner.csv",
        help="Path to diner CSV file (default: data/diner.csv)",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Show statistics without saving the mapping file",
    )

    # 기타
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = ap.parse_args()

    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 로그 디렉토리 생성
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    # 출력 파일 경로 설정 (프로젝트 루트 기준)
    if args.dry_run:
        output_path = None
        logger.info("DRY RUN 모드: 파일을 저장하지 않습니다.")
    else:
        output_path = project_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Region-Diner 매핑 데이터 생성 시작")
        logger.info(f"지역 데이터 디렉토리: {args.regions_dir}")
        logger.info(f"음식점 데이터 파일: {args.diner_csv}")

        # RegionMapper 인스턴스 생성
        regions_dir_path = project_root / args.regions_dir
        diner_csv_path = project_root / args.diner_csv
        mapper = RegionMapper(str(regions_dir_path))

        # 음식점 데이터 로드 (사용자 지정 경로 사용)
        logger.info("음식점 데이터 로드 중...")
        mapper.load_diner_data(str(diner_csv_path))

        # 매핑 데이터 생성 및 저장
        logger.info("매핑 데이터 생성 중...")
        if output_path:
            mapping_df = mapper.create_region_diner_mapping(str(output_path))
        else:
            # dry_run 모드에서는 임시 파일로 생성
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as tmp_file:
                mapping_df = mapper.create_region_diner_mapping(tmp_file.name)
                # 임시 파일 삭제
                Path(tmp_file.name).unlink()

        # 결과 요약
        total_diners = len(mapping_df)
        mapped_diners = len(mapping_df[mapping_df["region_id"] != -1])
        unmapped_diners = len(mapping_df[mapping_df["region_id"] == -1])

        # 콘솔 결과 요약
        print("\n" + "=" * 60)
        print("🗺️ Region-Diner 매핑 결과 요약")
        print("=" * 60)

        print(f"🍽️ 총 음식점 수: {total_diners:,}")
        print(
            f"✅ 매핑된 음식점: {mapped_diners:,} ({mapped_diners / total_diners * 100:.1f}%)"
        )
        print(
            f"❌ 매핑되지 않은 음식점: {unmapped_diners:,} ({unmapped_diners / total_diners * 100:.1f}%)"
        )

        if output_path:
            print(f"💾 결과 저장됨: {output_path}")

        # 지역별 통계
        print("\n=== 지역별 음식점 분포 (상위 10개) ===")
        region_stats = (
            mapping_df[mapping_df["region_id"] != -1]
            .groupby(["region_name", "region_id"])
            .size()
            .sort_values(ascending=False)
        )

        for i, ((region_name, region_id), count) in enumerate(
            region_stats.head(10).items(), 1
        ):
            print(f"{i:2d}. {region_name} (region_id: {region_id}): {count:,}개 음식점")

        # 매핑되지 않은 음식점 통계
        if unmapped_diners > 0:
            print(f"\n⚠️  매핑되지 않은 음식점: {unmapped_diners:,}개")
            print("   이는 해당 음식점이 어떤 지역에도 속하지 않음을 의미합니다.")

        print("=" * 60)
        print("🎯 Region-Diner 매핑이 성공적으로 완료되었습니다!")
        print("=" * 60)

        logger.info("매핑 데이터 생성 완료!")

    except Exception as e:
        logger.error(f"오류 발생: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
