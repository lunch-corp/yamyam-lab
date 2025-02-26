try:
    import os
    import sys

    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src")
    )

except ModuleNotFoundError:
    raise Exception("Module not found")


import pytest

from train_ranker import main


@pytest.mark.parametrize(
    "setup_ranker_config",
    [
        (
            "lightgbm",
            {
                "objective": "lambdarank",
                "boosting_type": "gbdt",
                "metric": "ndcg",
                "num_leaves": 16,
                "learning_rate": 0.1,
            },
            1,
        ),
    ],
    indirect=["setup_ranker_config"],
)
def test_run_ranker(setup_ranker_config):
    main(setup_ranker_config)
