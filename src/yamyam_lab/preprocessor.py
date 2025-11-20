"""
중분류 카테고리 간소화 전처리 CLI

원본 중분류를 간소화된 중분류로 변환하는 CLI 스크립트입니다.
"""

import argparse
import logging

import pandas as pd

from yamyam_lab.preprocess.diner_transform import MiddleCategorySimplifier


def main():
    """메인 실행 함수"""

    parser = argparse.ArgumentParser(description="중분류 카테고리 간소화")
    parser.add_argument(
        "--config-path",
        type=str,
        default="config",
        help="Config 파일 경로",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Data 파일 경로",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/diner_category_raw.csv",
        help="입력 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/diner_category_modified.csv",
        help="출력 파일 경로",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    simplifier = MiddleCategorySimplifier(
        config_root_path=args.config_path, data_path=args.data_path, logger=logger
    )

    # 데이터 로드
    category_df = pd.read_csv(args.input)

    # 간소화 처리
    result_df = simplifier.process(category_df)

    # 결과 저장
    result_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
