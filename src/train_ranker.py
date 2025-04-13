from __future__ import annotations

import logging

import hydra
import numpy as np
from omegaconf import DictConfig
from prettytable import PrettyTable
from tqdm import tqdm

from data.dataset import DatasetLoader
from evaluation.metric import ranking_metrics_at_k
from model.rank import build_model
from tools.utils import safe_divide


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def main(cfg: DictConfig):
    # load dataset
    data_loader = DatasetLoader(
        test_size=cfg.data.test_size,
        min_reviews=cfg.data.min_reviews,
        category_column_for_meta=cfg.data.category_column_for_meta,
        user_engineered_feature_names=cfg.data.user_engineered_feature_names[0],
        diner_engineered_feature_names=cfg.data.diner_engineered_feature_names[0],
        test=cfg.data.test,
    )
    data = data_loader.prepare_train_val_dataset(
        is_rank=True, is_candidate_dataset=cfg.data.is_candidate_dataset
    )
    # mapping reverse
    X_train, y_train, X_test, y_test = (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
    )
    # build Pmodel
    trainer = build_model(cfg)

    # train model
    trainer.fit(X_train, y_train, X_test, y_test)

    # save model
    trainer.save_model()

    # plot feature importance
    trainer.plot_feature_importance()

    try:
        # candidate predictions
        candidates = data["candidates"]
        batch_size = 100000
        num_batches = (len(candidates) + batch_size - 1) // batch_size
        predictions = np.zeros(len(candidates))

        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(candidates))
            batch = candidates[cfg.data.features].iloc[start_idx:end_idx]
            predictions[start_idx:end_idx] = trainer.predict(batch)

        # Group predictions by user
        candidates["pred_score"] = predictions
        candidates = candidates.sort_values(
            by=["reviewer_id", "pred_score"], ascending=False
        )
        user_predictions = candidates.groupby("reviewer_id")["diner_idx"].apply(
            np.array
        )

        # Calculate metrics
        metric_at_K = {K: {"map": 0, "ndcg": 0, "count": 0} for K in [3, 7, 10, 20]}

        # Get already liked items from test data
        test_liked_items = X_test.groupby("reviewer_id")["diner_idx"].apply(np.array)

        # Get already liked items from training data
        train_liked_items = X_train.groupby("reviewer_id")["diner_idx"].apply(np.array)

        # Calculate metrics for each user
        for user in tqdm(user_predictions.index):
            if user not in test_liked_items:
                continue
            liked_items = test_liked_items[user]
            pred_items = user_predictions[user]

            # Filter out already liked items from training data
            if user in train_liked_items:
                already_liked = train_liked_items[user]

                # Create a mask for items that were not liked in training
                mask = ~np.isin(pred_items, already_liked)
                pred_items = pred_items[mask]

            # Skip if no predictions left after filtering
            if len(pred_items) == 0:
                continue

            for K in metric_at_K.keys():
                # if len(liked_items) < K:
                #     continue
                metric = ranking_metrics_at_k(
                    liked_items=np.array(liked_items), reco_items=pred_items[:K]
                )

                metric_at_K[K]["map"] += metric["ap"]
                metric_at_K[K]["ndcg"] += metric["ndcg"]
                metric_at_K[K]["count"] += 1

        # Average metrics
        table = PrettyTable()
        table.field_names = ["K", "MAP", "NDCG"]
        for K in metric_at_K:
            metric_at_K[K]["map"] = safe_divide(
                numerator=metric_at_K[K]["map"], denominator=metric_at_K[K]["count"]
            )
            metric_at_K[K]["ndcg"] = safe_divide(
                numerator=metric_at_K[K]["ndcg"], denominator=metric_at_K[K]["count"]
            )
            table.add_row(
                [K, f"{metric_at_K[K]['map']:.8f}", f"{metric_at_K[K]['ndcg']:.8f}"]
            )
        logging.info(f"\nEvaluation Results\n{table}")

    except Exception:
        logging.info("No candidates data found just training model")


if __name__ == "__main__":
    main()
