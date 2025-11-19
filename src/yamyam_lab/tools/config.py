import os
from datetime import datetime
from typing import Tuple

import yaml
from easydict import EasyDict


def load_yaml(path: str):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return EasyDict(config)


def load_configs(model: str, config_root_path: str = None) -> Tuple[EasyDict, EasyDict]:
    if model in ["node2vec", "metapath2vec", "graphsage", "lightgcn"]:
        model_type = "graph"
    elif model in ["svd_bias", "als"]:
        model_type = "mf"
    else:
        raise ValueError(f"Unsupported model type: {model}")
    if config_root_path is None:
        config_root_path = os.path.join(os.path.dirname(__file__), "../../../config")

    model_config = load_yaml(
        os.path.join(config_root_path, f"./models/{model_type}/{model}.yaml")
    )
    preprocess_config = load_yaml(
        os.path.join(config_root_path, "./preprocess/preprocess.yaml")
    )

    return model_config, preprocess_config


def generate_result_path(
    model: str, test: bool, result_path: str = None, postfix: str = None
) -> str:
    if result_path is not None:
        return result_path

    root_path = os.path.join(os.path.dirname(__file__), "../../..")
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    test_flag = "test" if test else "untest"
    result_path = os.path.join(
        root_path,
        f"./result/{test_flag}/{model}/{dt}" + (f"_{postfix}" if postfix else ""),
    )
    os.makedirs(result_path, exist_ok=True)
    return result_path
