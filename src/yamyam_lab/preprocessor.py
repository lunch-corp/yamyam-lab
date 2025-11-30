"""
중분류 카테고리 간소화 전처리 CLI

원본 중분류를 간소화된 중분류로 변환하는 CLI 스크립트입니다.
KNN 기반 null 값 imputation도 지원합니다.
"""

import argparse
import logging

import pandas as pd

from yamyam_lab.preprocess.diner_transform import (
    CategoryProcessor,
    MiddleCategoryKNNImputer,
    MiddleCategorySimplifier,
)


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
        "--use-knn-imputation",
        action="store_true",
        help="KNN 기반 중분류 null 값 imputation 사용",
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=5,
        help="KNN에서 사용할 이웃 수 (기본값: 5, model-type=knn일 때만 사용)",
    )
    parser.add_argument(
        "--use-diner-info",
        action="store_true",
        help="diner.csv의 정보(diner_name, diner_tag 등)를 feature로 사용",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="평가용 데이터 비율 (기본값: 0.2)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # 데이터 로드
    category_df = pd.read_csv(args.input)

    # step 1: CategoryProcessor로 대분류 분류 로직 적용
    logger.info("=" * 60)
    logger.info("Step 1: Category Processing")
    logger.info("=" * 60)
    processor = CategoryProcessor(
        df=category_df,
        config_root_path=args.config_path,
    )
    processor.process_all()
    category_df = processor.category_preprocessed_diners
    logger.info("Category processing completed")

    # step 2: 간소화 처리
    logger.info("=" * 60)
    logger.info("Step 2: Middle Category Simplification")
    logger.info("=" * 60)
    simplifier = MiddleCategorySimplifier(
        config_root_path=args.config_path, data_path=args.data_path, logger=logger
    )
    result_df = simplifier.process(category_df)

    # KNN imputation (옵션)
    if args.use_knn_imputation:
        logger.info("=" * 60)
        logger.info("Starting KNN-based imputation")
        logger.info("=" * 60)

        imputer = MiddleCategoryKNNImputer(
            n_neighbors=args.knn_neighbors,
            data_path=args.data_path,
            logger=logger,
        )

        # 학습 및 imputation
        result_df, metrics = imputer.fit_and_impute(
            result_df,
            use_diner_info=args.use_diner_info,
            test_size=args.test_size,
        )

        logger.info("KNN Imputation completed")

    print("\n중분류 분포:")
    print(result_df["diner_category_middle"].value_counts())

    # null 개수 확인
    null_count = result_df["diner_category_middle"].isna().sum()
    logger.info(f"Remaining null middle categories: {null_count:,}")

    # 결과 저장
    result_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
