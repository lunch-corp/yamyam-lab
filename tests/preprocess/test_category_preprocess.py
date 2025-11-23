import logging
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src"))

from yamyam_lab.preprocess.diner_transform import (
    CategoryProcessor,
    MiddleCategorySimplifier,
)
from yamyam_lab.tools.config import load_yaml
from yamyam_lab.tools.google_drive import check_data_and_return_paths

ROOT_PATH = os.path.join(os.path.dirname(__file__), "../..")
CONFIG_DIR = os.path.join(ROOT_PATH, "config")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/data/category_mappings.yaml")


def test_category_preprocess():
    """
    Test if category preprocessing is correctly done or not.
    """
    data_paths = check_data_and_return_paths()
    diner_with_raw_category = pd.read_csv(data_paths["category"])

    processor = CategoryProcessor(
        df=diner_with_raw_category,
        config_root_path=os.path.join(os.path.dirname(__file__), "../../config"),
    )
    processor.process_all()
    diner_with_processd_category = processor.category_preprocessed_diners
    integrated_diner_category_middle = []
    for diner_category_large, config in processor.mappings[
        "integrate_diner_category_middle"
    ].items():
        for asis, tobe in config.items():
            integrated_diner_category_middle += [cat for cat in tobe if cat != asis]

    config = load_yaml(CONFIG_PATH)

    # check lowering_large_categories preprocessing
    # Note: lowering_large_categories는 대분류를 변경하고 카테고리를 한 단계 아래로 이동시킵니다.
    # 따라서 원래 대분류가 "샐러드"였던 경우, 처리 후에는:
    # - large="양식", middle=원래 small 값, small=원래 detail 값
    # 즉, 중분류가 원래 대분류 값과 같을 수 없습니다.
    # 대신 원본 데이터에서 대분류가 before_category였던 행이 처리 후 after_category로 변경되었는지 확인합니다.
    for (
        after_category_large,
        before_category_large,
    ) in config.lowering_large_categories.items():
        for cat in before_category_large:
            if cat in integrated_diner_category_middle:
                continue
            # 원본 데이터에서 대분류가 cat이었던 행 찾기
            original_rows = diner_with_raw_category[
                diner_with_raw_category["diner_category_large"] == cat
            ]
            if original_rows.shape[0] == 0:
                continue  # 원본 데이터에 해당 카테고리가 없으면 스킵

            # 처리 후 해당 행들이 after_category_large로 변경되었는지 확인
            processed_indices = original_rows["diner_idx"].values
            diner_filter = diner_with_processd_category[
                (diner_with_processd_category["diner_idx"].isin(processed_indices))
                & (
                    diner_with_processd_category["diner_category_large"]
                    == after_category_large
                )
            ]
            assert diner_filter.shape[0] > 0, (
                f"대분류가 '{cat}'였던 행들이 '{after_category_large}'로 변경되지 않았습니다. "
                f"원본: {original_rows.shape[0]}개, 처리 후: {diner_filter.shape[0]}개"
            )

    # check chicken category preprocessing
    chicken_small_category = list(config.chicken_category.keys())
    diner_with_chicken = diner_with_processd_category[
        lambda x: (x["diner_category_large"] == "양식")
        & (x["diner_category_middle"] == "치킨")
    ]

    assert diner_with_chicken.shape[0] > 0
    assert (
        diner_with_chicken[
            lambda x: x["diner_category_small"].isin(chicken_small_category)
        ].shape[0]
        > 0
    )


def test_middle_category_simplification():
    """
    Test if middle category simplification is correctly done or not.
    """

    # 테스트 데이터
    test_data = pd.DataFrame(
        {
            "diner_category_large": [
                "한식",
                "한식",
                "양식",
                "카페",
                "치킨",
                "한식",
                "분식",
            ],
            "diner_category_middle": [
                "덮밥90도씨",
                "곰탕",
                "햄버거",
                "아임일리터",
                "BBQ",
                "국수",
                "떡볶이",
            ],
        }
    )

    # 로거 설정 (테스트용)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    # MiddleCategorySimplifier 초기화
    simplifier = MiddleCategorySimplifier(
        config_root_path=CONFIG_DIR,
        data_path=ROOT_PATH,
        logger=logger,
    )

    # 간소화 처리
    result_df = simplifier.process(test_data.copy(), inplace=False)

    # 검증
    # 덮밥90도씨 -> 덮밥
    assert result_df.iloc[0]["diner_category_middle"] == "덮밥"
    assert result_df.iloc[0]["diner_category_large"] == "한식"

    # 곰탕 -> 탕/국밥
    assert result_df.iloc[1]["diner_category_middle"] == "탕/순대/국밥"
    assert result_df.iloc[1]["diner_category_large"] == "한식"

    # 햄버거 -> 햄버거 (변화 없음)
    assert result_df.iloc[2]["diner_category_middle"] == "수제버거"
    assert result_df.iloc[2]["diner_category_large"] == "양식"

    # 아임일리터 -> 커피
    assert result_df.iloc[3]["diner_category_middle"] == "커피"
    assert result_df.iloc[3]["diner_category_large"] == "카페"

    # BBQ -> 치킨전문점
    assert result_df.iloc[4]["diner_category_middle"] == "치킨전문점"
    assert result_df.iloc[4]["diner_category_large"] == "치킨"

    # 국수 -> 국수 (변화 없음)
    assert result_df.iloc[5]["diner_category_middle"] == "국수"
    assert result_df.iloc[5]["diner_category_large"] == "한식"

    # 떡볶이 -> 떡볶이/순대/튀김
    assert result_df.iloc[6]["diner_category_middle"] == "떡볶이/순대/튀김"
    assert result_df.iloc[6]["diner_category_large"] == "분식"

    # 원본 데이터는 변경되지 않았는지 확인 (inplace=False)
    assert test_data.iloc[0]["diner_category_middle"] == "덮밥90도씨"
