try:
    import os
    import sys

    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src")
    )

except ModuleNotFoundError:
    raise Exception("Module not found")

import pytest

from train_embedding import main


@pytest.mark.parametrize(
    "setup_config",
    [
        ("node2vec", False),
        ("metapath2vec", True),
    ],
    indirect=["setup_config"],
)
def test_run_node2vec(setup_config):
    main(setup_config)
