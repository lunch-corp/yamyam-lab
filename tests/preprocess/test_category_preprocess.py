import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src"))


from preprocess.diner_transform import CategoryProcessor
from tools.config import load_yaml
from tools.google_drive import ensure_data_files

ROOT_PATH = os.path.join(os.path.dirname(__file__), "../..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/data/category_mappings.yaml")


def test_category_preprocess():
    """
    Test if category preprocessing is correctly done or not.
    """
    data_paths = ensure_data_files()
    diner_with_raw_category = pd.read_csv(data_paths["category"])

    processor = CategoryProcessor(diner_with_raw_category)
    diner_with_processd_category = processor.process_all().df

    config = load_yaml(CONFIG_PATH)

    # check lowering_large_categories preprocessing
    for (
        after_category_large,
        before_category_large,
    ) in config.lowering_large_categories.items():
        for cat in before_category_large:
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
