import pytest

from yamyam_lab.train_torch import main


@pytest.mark.parametrize(
    "setup_config",
    [
        ("svd_bias", False),
    ],
    indirect=["setup_config"],
)
def test_run_svd_bias(setup_config):
    main(setup_config)
