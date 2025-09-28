try:
    import os
    import sys

    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src")
    )
except ModuleNotFoundError:
    raise Exception("Module not found")


import pytest

from train_most_popular_rerank import main


@pytest.mark.parametrize(
    "setup_mostpopular_config",
    [
        (
            "mostpopular",
            {
                "region_label": "서울 강남구",
                "hotspot_coords": None,
                "n_auto_hotspots": 0,
                "periphery_strength": 0.1,
                "periphery_cap": 0.03,
                "lambda_div": 0.9,
                "w_cat": 0.1,
                "w_geo": 0.1,
                "geo_tau_km": 8.0,
                "prefix_freeze": 15,
            },
            1,
        ),
    ],
    indirect=["setup_mostpopular_config"],
)
def test_run_rerank_most_popular(most_popular_postprocess_config):
    main(most_popular_postprocess_config)
