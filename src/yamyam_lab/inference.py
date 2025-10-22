import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from prettytable import PrettyTable

from yamyam_lab.data.ranker import load_test_dataset
from yamyam_lab.tools.utils import get_kakao_lat_lng, haversine


@hydra.main(config_path="../config/", config_name="predict", version_base="1.3.1")
def _main(cfg: DictConfig):
    test = load_test_dataset(cfg)
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

    logging.info(f"추천 식당 리스트를 알립니다.\n{table}")


if __name__ == "__main__":
    _main()
