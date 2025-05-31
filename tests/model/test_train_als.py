import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src"))


from train_als import main


def test_run_als(setup_als_config):
    main(setup_als_config)
