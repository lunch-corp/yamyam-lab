import os
import traceback
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd

from yamyam_lab.data.config import DataConfig
from yamyam_lab.data.csr import CsrDatasetLoader
from yamyam_lab.evaluation.metric_calculator import MostPopularMetricCalculator
from yamyam_lab.postprocess.most_popular_rerank import RegionPeripheryReranker
from yamyam_lab.tools.config import load_yaml
from yamyam_lab.tools.logger import common_logging, setup_logger
from yamyam_lab.tools.parse_args import (
    parse_args_mostpopular_rerank,
    save_command_to_file,
)
from yamyam_lab.tools.rerank import extract_region_label

ROOT_PATH = os.path.join(os.path.dirname(__file__), "../..")
CONFIG_PATH = os.path.join(ROOT_PATH, "./config/models/mf/{model}.yaml")
RESULT_PATH = os.path.join(ROOT_PATH, "./result/{test}/{model}/{dt}")
PREPROCESS_CONFIG_PATH = os.path.join(ROOT_PATH, "./config/preprocess/preprocess.yaml")


def mask_region_df(df: pd.DataFrame, region_ids_mapped: set[int]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    return df[df["diner_idx"].isin(region_ids_mapped)].copy()


def main(args: ArgumentParser.parse_args) -> None:
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    test_flag = "untest"
    result_path = RESULT_PATH.format(test=test_flag, model="most_popular", dt=dt)
    os.makedirs(result_path, exist_ok=True)

    config = load_yaml(CONFIG_PATH.format(model="als"))  # als 설정 재사용
    preprocess_config = load_yaml(PREPROCESS_CONFIG_PATH)

    save_command_to_file(result_path)

    top_k_values_for_pred = config.training.evaluation.top_k_values_for_pred
    top_k_values_for_candidate = config.training.evaluation.top_k_values_for_candidate
    top_k_values = top_k_values_for_pred + top_k_values_for_candidate
    file_name = config.post_training.file_name
    fe = config.preprocess.feature_engineering

    logger = setup_logger(os.path.join(result_path, file_name.log))

    try:
        logger.info("model: rerank_most_popular")
        logger.info(f"training results will be saved in {RESULT_PATH}")

        data_config = DataConfig(
            X_columns=["diner_idx", "reviewer_id"],
            y_columns=["reviewer_review_score"],
            user_engineered_feature_names=fe.user_engineered_feature_names,
            diner_engineered_feature_names=fe.diner_engineered_feature_names,
            is_timeseries_by_time_point=config.preprocess.data.is_timeseries_by_time_point,
            train_time_point=config.preprocess.data.train_time_point,
            val_time_point=config.preprocess.data.val_time_point,
            test_time_point=config.preprocess.data.test_time_point,
            end_time_point=config.preprocess.data.end_time_point,
            test=False,
        )
        data_loader = CsrDatasetLoader(data_config=data_config)
        data = data_loader.prepare_csr_dataset(
            is_csr=True, filter_config=preprocess_config.filter
        )

        common_logging(config, data, logger)

        _, diner_data, diner_category_data = data_loader.load_dataset()
        diner_meta = pd.merge(
            diner_data[["diner_idx", "diner_lat", "diner_lon", "diner_road_address"]],
            diner_category_data[["diner_idx", "diner_category_large"]],
            on="diner_idx",
            how="left",
        )

        target_region = extract_region_label(args.region_label)
        if "diner_road_address" in diner_meta.columns:
            diner_meta = diner_meta.copy()
            diner_meta["region_label"] = (
                diner_meta["diner_road_address"].map(extract_region_label).astype(str)
            )
        else:
            diner_meta["region_label"] = "unknown"

        dmap = data["diner_mapping"]  # {original_diner_idx: mapped_diner_idx}
        diner_meta["diner_idx_mapped"] = (
            diner_meta["diner_idx"].map(dmap).astype("Int64")
        )
        diner_meta_mp = diner_meta.dropna(subset=["diner_idx_mapped"]).copy()
        diner_meta_mp["diner_idx"] = diner_meta_mp["diner_idx_mapped"].astype(int)
        diner_meta_mp = diner_meta_mp.drop(columns=["diner_idx_mapped"])

        candidates = np.array(data["most_popular_diner_ids"], dtype=np.int64)
        base_scores = 1.0 / (np.arange(len(candidates)) + 1)

        reranker = RegionPeripheryReranker(
            region_label=args.region_label,
            hotspot_coords=args.hotspot_coords,
            n_auto_hotspots=args.n_auto_hotspots,
            periphery_strength=args.periphery_strength,
            periphery_cap=args.periphery_cap,
            lambda_div=args.lambda_div,
            w_cat=args.w_cat,
            w_geo=args.w_geo,
            geo_tau_km=args.geo_tau_km,
        )

        reranked_ids, _ = reranker.rerank(
            item_ids=candidates,
            base_scores=base_scores,
            item_meta=diner_meta_mp,
            k=max(top_k_values),
            popularity_scores=base_scores,
            popularity_weight=0.25,
            normalize_rel=True,
            top_m=None,
        )

        if reranked_ids.size == 0:
            logger.warning(
                "No candidates after reranking (empty reranked_ids). Skipping metrics."
            )
            return

        region_ids_mapped = set(
            diner_meta_mp.loc[
                diner_meta_mp["diner_road_address"].map(extract_region_label)
                == target_region,
                "diner_idx",
            ]
            .astype(int)
            .unique()
        )
        use_region_mask = len(region_ids_mapped) > 0

        X_val_warm = data["X_val_warm_users"]
        X_val_cold = data["X_val_cold_users"]
        X_test_warm = data["X_test_warm_users"]
        X_test_cold = data["X_test_cold_users"]

        if use_region_mask:
            X_val_warm = mask_region_df(X_val_warm, region_ids_mapped)
            X_val_cold = mask_region_df(X_val_cold, region_ids_mapped)
            X_test_warm = mask_region_df(X_test_warm, region_ids_mapped)
            X_test_cold = mask_region_df(X_test_cold, region_ids_mapped)

        metric_calculator = MostPopularMetricCalculator(
            top_k_values=top_k_values,
            filter_already_liked=True,
            recommend_batch_size=config.training.evaluation.recommend_batch_size,
            logger=logger,
        )

        # ===== Validation =====
        metric_dict = metric_calculator.generate_recommendations_and_calculate_metric(
            X_train=data["X_train_df"],
            X_val_warm_users=X_val_warm,
            X_val_cold_users=X_val_cold,
            most_popular_diner_ids=reranked_ids,
            train_csr=data["X_train"],
            filter_already_liked=True,
            most_popular_rec_to_warm_users=True,
        )
        for user_type, metric in metric_dict.items():
            metric_calculator.calculate_mean_metric(metric)
        logger.info(
            "################################ Validation data metric report ################################"
        )
        metric_calculator.report_metric_with_warm_cold_all_users(
            metric_dict=metric_dict, data_type="val"
        )

        # ===== Test =====
        metric_dict = metric_calculator.generate_recommendations_and_calculate_metric(
            X_train=data["X_train_df"],
            X_val_warm_users=X_test_warm,
            X_val_cold_users=X_test_cold,
            most_popular_diner_ids=reranked_ids,
            train_csr=data["X_train"],
            filter_already_liked=True,
            most_popular_rec_to_warm_users=True,
        )
        for user_type, metric in metric_dict.items():
            metric_calculator.calculate_mean_metric(metric)
        logger.info(
            "################################ Test data metric report ################################"
        )
        metric_calculator.report_metric_with_warm_cold_all_users(
            metric_dict=metric_dict, data_type="test"
        )

    except Exception as e:
        print("Error during project run:", repr(e))
        traceback.print_exc()


if __name__ == "__main__":
    args = parse_args_mostpopular_rerank()
    main(args)
