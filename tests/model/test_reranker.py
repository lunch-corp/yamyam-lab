import pytest

from yamyam_lab.rerank import main


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
