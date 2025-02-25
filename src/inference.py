from __future__ import annotations

import logging
import math

import hydra
from omegaconf import DictConfig
from prettytable import PrettyTable

from data import load_test_dataset
from rank import build_model


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points
    on the Earth's surface using the Haversine formula.

    Parameters:
        lat1, lon1: Latitude and longitude of the first point in decimal degrees
        lat2, lon2: Latitude and longitude of the second point in decimal degrees

    Returns:
        Distance between the two points in kilometers
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Earth's radius in kilometers
    radius = 6371.0

    # Calculate the distance
    distance = radius * c

    return distance


@hydra.main(config_path="../config/", config_name="predict", version_base="1.3.1")
def _main(cfg: DictConfig):
    test, already_reviewed = load_test_dataset(
        reviewer_id=cfg.user_name,
        diner_engineered_feature_names=cfg.diner_engineered_feature_names,
    )
    test["user_lat"] = cfg.user_lat
    test["user_lon"] = cfg.user_lon

    test["distance"] = test.apply(
        lambda x: haversine(
            x["user_lat"], x["user_lon"], x["diner_lat"], x["diner_lon"]
        ),
        axis=1,
    )
    test = test.loc[test["distance"] <= cfg.distance_threshold]
    X_test = test[cfg.data.features]

    # load model
    trainer = build_model(cfg)
    predictions = trainer.predict(X_test)

    test["prediction"] = predictions
    test = test.sort_values(by=["prediction"], ascending=False)
    test = test.loc[(test["diner_category_middle"].isin([*cfg.diner_category_middle]))]
    test = test.head(cfg.top_n)

    table = PrettyTable()
    table.field_names = ["diner_name", "diner_category_small", "url", "prediction"]

    for _, row in test.iterrows():
        table.add_row(
            [
                row["diner_name"],
                row["diner_category_small"],
                row["diner_url"],
                row["prediction"],
            ]
        )

    logging.info(
        f"{test['reviewer_user_name'].values[0]}님을 위한 추천 식당 리스트를 알립니다.\n{table}"
    )


if __name__ == "__main__":
    _main()
