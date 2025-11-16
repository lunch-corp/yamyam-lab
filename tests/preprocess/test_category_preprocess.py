import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src"))


from yamyam_lab.preprocess.diner_transform import CategoryProcessor
from yamyam_lab.tools.config import load_yaml
from yamyam_lab.tools.google_drive import check_data_and_return_paths

ROOT_PATH = os.path.join(os.path.dirname(__file__), "../..")
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
    for (
        after_category_large,
        before_category_large,
    ) in config.lowering_large_categories.items():
        for cat in before_category_large:
            if cat in integrated_diner_category_middle:
                continue
            diner_filter = diner_with_processd_category[
                lambda x: (x["diner_category_large"] == after_category_large)
                & (x["diner_category_middle"] == cat)
            ]
            assert diner_filter.shape[0] > 0

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
