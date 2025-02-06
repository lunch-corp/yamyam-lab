try:
    import os
    import sys

    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src")
    )

except ModuleNotFoundError:
    raise Exception("Module not found")

import pytest

from train_torch import main



@pytest.mark.parametrize(
    "setup_config",
    [
        ("svd_bias", False),
    ],
    indirect=["setup_config"]
)
def test_run_svd_bias(setup_config):
    main(setup_config)
