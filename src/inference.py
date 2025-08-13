import logging

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from prettytable import PrettyTable

from data.dataset import load_test_dataset
from tools.utils import get_kakao_lat_lng


def haversine(
    reviewer_lat: float, reviewer_lon: float, diner_lat: pd.Series, diner_lon: pd.Series
) -> np.ndarray:
    """
    Compute the great-circle distance between a single point (lat1, lon1) and multiple points (lat2, lon2)
    using the Haversine formula in a vectorized way.

    Args:
        reviewer_lat (float): Latitude of the reviewer.
        reviewer_lon (float): Longitude of the reviewer.
        diner_lat (pd.Series): Latitude of the diners.
        diner_lon (pd.Series): Longitude of the diners.

    Returns:
        np.ndarray: Array of distances.
    """
    # Convert degrees to radians
    reviewer_lat, reviewer_lon = np.radians(reviewer_lat), np.radians(reviewer_lon)
    diner_lat, diner_lon = np.radians(diner_lat), np.radians(diner_lon)

    # Haversine formula
    dlat = diner_lat - reviewer_lat
    dlon = diner_lon - reviewer_lon
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(reviewer_lat) * np.cos(diner_lat) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Earth's radius in kilometers
    radius = 6371.0

    return radius * c


@hydra.main(config_path="../config/", config_name="predict", version_base="1.3.1")
def _main(cfg: DictConfig):
    test, already_reviewed = load_test_dataset(
        cfg.user_name,
        cfg.data.user_engineered_feature_names,
        cfg.data.diner_engineered_feature_names,
    )
    location = get_kakao_lat_lng(cfg.user_address)
    user_lat, user_lon = float(location["lat"]), float(location["lng"])

    test["distance"] = haversine(
        user_lat, user_lon, test["diner_lat"], test["diner_lon"]
    )
    test = test.loc[test["distance"] <= cfg.distance_threshold]
    X_test = test[cfg.models.ranker.features]

    # load model
    trainer = instantiate(cfg.models.ranker)
    predictions = trainer.predict(X_test)

    test["prediction"] = predictions
    test = test.sort_values(by=["prediction"], ascending=False)
    test = test.loc[(test["diner_category_large"].isin([*cfg.diner_category_large]))]
    test = test.head(cfg.top_n)

    table = PrettyTable()
    table.field_names = [
        "diner_name",
        "diner_category_large",
        "diner_category_middle",
        "prediction",
    ]

    for _, row in test.iterrows():
        table.add_row(
            [
                row["diner_name"],
                row["diner_category_large"],
                row["diner_category_middle"],
                row["prediction"],
            ]
        )

    logging.info(
        f"{test['reviewer_user_name'].values[0]}님을 위한 추천 식당 리스트를 알립니다.\n{table}"
    )


if __name__ == "__main__":
    _main()
