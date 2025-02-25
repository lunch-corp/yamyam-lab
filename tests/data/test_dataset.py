try:
    import os
    import sys

    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src")
    )

except ModuleNotFoundError:
    raise Exception("Module not found")

import pandas as pd

from constant.evaluation.qualitative import QualitativeReviewerId
from data.dataset import load_test_dataset


def test_load_test_dataset():
    reviewer_id = QualitativeReviewerId.ROCKY.value
    test, already_reviewed = load_test_dataset(
        reviewer_id=reviewer_id,
        diner_engineered_feature_names=["all_review_cnt"]
    )
    assert test is not None
    assert already_reviewed is not None
    assert len(test) > 0
    assert len(already_reviewed) >= 0
    assert isinstance(test, pd.DataFrame)
    assert isinstance(already_reviewed, list)
