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
    parser.add_argument(
        "--diner-file",
        type=str,
        default=None,
        help="Diner 데이터 파일 경로 (diner_name, diner_tag 포함). 제공되면 null 채우기에 사용됩니다.",
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

    # diner 파일이 제공되면 merge하여 diner_name과 diner_tag 추가
    if args.diner_file:
        logger.info(f"Loading diner data from {args.diner_file} for null filling")
        diner_df = pd.read_csv(args.diner_file)
        # diner_idx를 기준으로 merge (필요한 컬럼만 선택)
        merge_columns = ["diner_idx"]
        if "diner_name" in diner_df.columns:
            merge_columns.append("diner_name")
        if "diner_tag" in diner_df.columns:
            merge_columns.append("diner_tag")

        category_df = pd.merge(
            category_df,
            diner_df[merge_columns],
            on="diner_idx",
            how="left",
        )
        logger.info(f"Merged diner data. Columns: {category_df.columns.tolist()}")
    else:
        logger.warning(
            "--diner-file not provided. Null filling will be skipped if diner_name/diner_tag columns are missing."
        )

    # 간소화 처리
    result_df = simplifier.process(category_df)

    # null 채우기에 사용된 diner_name, diner_tag 컬럼 제거
    columns_to_drop = []
    if "diner_name" in result_df.columns:
        columns_to_drop.append("diner_name")
    if "diner_tag" in result_df.columns:
        columns_to_drop.append("diner_tag")
    if "diner_menu_name" in result_df.columns:
        columns_to_drop.append("diner_menu_name")

    if columns_to_drop:
        result_df = result_df.drop(columns=columns_to_drop)
        logger.info(f"Dropped columns used for null filling: {columns_to_drop}")

    print(result_df["diner_category_middle"].value_counts())

    # 결과 저장
    result_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
