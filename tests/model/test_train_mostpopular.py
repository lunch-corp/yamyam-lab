
try:
    import os
    import sys
    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src")
    )
except ModuleNotFoundError:
    raise Exception("Module not found")

import pytest
from types import SimpleNamespace

from train_mp_postprocess import main


@pytest.mark.parametrize(
    "args",
    [
        SimpleNamespace(
            # reranker args
            region_label="서울 강남구",
            hotspot_coords=None,
            n_auto_hotspots=0,
            periphery_strength=0.0,
            periphery_cap=0.15,
            lambda_div=0.55,
            w_cat=0.5,
            w_geo=0.5,
            geo_tau_km=2.0,
        ),

        SimpleNamespace(
            region_label="서울 강남구",
            hotspot_coords=None,
            n_auto_hotspots=0,
            periphery_strength=0.8,
            periphery_cap=0.20,
            lambda_div=0.40,
            w_cat=0.3,
            w_geo=0.7,
            geo_tau_km=1.0,
        ),

        SimpleNamespace(
            region_label="서울 강남구",
            hotspot_coords=None,
            n_auto_hotspots=3,
            periphery_strength=0.5,
            periphery_cap=0.10,
            lambda_div=0.20,  
            w_cat=0.7,
            w_geo=0.3,
            geo_tau_km=3.0,
        ),

        SimpleNamespace(
            region_label="서울 강남구",
            hotspot_coords=None,
            n_auto_hotspots=5,
            periphery_strength=0.2,
            periphery_cap=0.05,
            lambda_div=0.90,   # 관련성 쪽
            w_cat=0.6,
            w_geo=0.4,
            geo_tau_km=5.0,
        ),

        SimpleNamespace(
            region_label="서울 강남구",
            hotspot_coords=None,
            n_auto_hotspots=2,
            periphery_strength=0.6,
            periphery_cap=0.25,
            lambda_div=0.50,
            w_cat=0.1,
            w_geo=0.9,  
            geo_tau_km=0.5,
        ),
    ],
)
def test_run_rerank_most_popular(args):

    main(args)
