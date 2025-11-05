import argparse
import os
import random
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from yamyam_lab.data.config import DataConfig
from yamyam_lab.data.csr import CsrDatasetLoader
from yamyam_lab.evaluation.metric_calculator.similarity_metric_calculator import (
    ItemBasedMetricCalculator,
)
from yamyam_lab.model.classic_cf.item_based import ItemBasedCollaborativeFiltering
from yamyam_lab.tools.config import load_yaml
from yamyam_lab.tools.logger import logging_data_statistics, setup_logger
from yamyam_lab.tools.parse_args import save_command_to_file

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
CONFIG_PATH = os.path.join(ROOT_PATH, ".././config/models/mf/{model}.yaml")
PREPROCESS_CONFIG_PATH = os.path.join(
    ROOT_PATH, ".././config/preprocess/preprocess.yaml"
)
RESULT_PATH = os.path.join(ROOT_PATH, ".././result/{test}/{model}/{dt}")


def load_model_for_inference():
    """Load model for inference (data loading + model initialization)."""
    config = load_yaml(CONFIG_PATH.format(model="als"))
    preprocess_config = load_yaml(PREPROCESS_CONFIG_PATH)
    fe = config.preprocess.feature_engineering

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

    model = ItemBasedCollaborativeFiltering(
        user_item_matrix=data["X_train"],
        user_mapping=data["user_mapping"],
        item_mapping=data["diner_mapping"],
        diner_df=None,
    )

    item_interaction_counts = np.array(model.item_user_matrix.sum(axis=1)).flatten()

    return model, data, item_interaction_counts


def find_similar_items_cli(
    target_id: int,
    top_k: int = 10,
    method: str = "cosine_matrix",
    min_interactions: int = 10,
):
    """
    Find and display similar restaurants for a specific restaurant ID.

    Args:
        target_id: Target restaurant ID
        top_k: Number of similar restaurants to return (default: 10)
        method: Similarity calculation method - "cosine_matrix" or "jaccard"
        min_interactions: Minimum reviews for candidate restaurants (default: 10)
    """
    model, data, item_interaction_counts = load_model_for_inference()

    if target_id not in model.item_mapping:
        print(f"Error: Restaurant ID {target_id} not found in training data")
        return

    target_idx = model.item_mapping[target_id]
    target_reviews = int(item_interaction_counts[target_idx])

    # Get more candidates than needed
    similar_items = model.find_similar_items(
        target_item_id=target_id,
        top_k=top_k * 5,
        method=method,
    )

    if not similar_items:
        print("No similar restaurants found")
        return

    # Filter out items with very low similarity
    non_zero_items = [
        item for item in similar_items if item["similarity_score"] > 0.0001
    ]

    if not non_zero_items:
        print("No restaurants with meaningful similarity found")
        print("Try using a more popular restaurant or --method jaccard")
        return

    # Prioritize items with >= min_interactions, but fall back if needed
    high_quality_items = [
        item
        for item in non_zero_items
        if item_interaction_counts[model.item_mapping[item["item_id"]]]
        >= min_interactions
    ]

    # Use high quality items if available, otherwise use all non-zero items
    display_items = high_quality_items if high_quality_items else non_zero_items

    # Ensure we have at least top_k items (or as many as available)
    display_items = display_items[:top_k]

    print(f"\nTarget: Restaurant {target_id} ({target_reviews} reviews)")
    print(f"Found {len(display_items)} similar restaurants:\n")

    for i, item in enumerate(display_items, 1):
        similar_idx = model.item_mapping[item["item_id"]]
        similar_reviews = int(item_interaction_counts[similar_idx])
        print(
            f"{i:2d}. ID {item['item_id']:10d} | "
            f"Similarity: {item['similarity_score']:.4f} | "
            f"Reviews: {similar_reviews:,}"
        )


def find_similar_items_demo(
    top_k: int = 10, num_examples: int = 3, min_interactions: int = 10
):
    """Demo: randomly select restaurants and find similar items."""
    model, data, item_interaction_counts = load_model_for_inference()

    valid_indices = np.where(item_interaction_counts >= min_interactions)[0]
    if len(valid_indices) == 0:
        print(f"No items with at least {min_interactions} interactions found")
        return

    valid_item_ids = [
        model.idx_to_item[idx] for idx in valid_indices if idx in model.idx_to_item
    ]

    print(f"Filtered {len(valid_item_ids):,} items with >={min_interactions} reviews")
    print(f"Testing {num_examples} random restaurants\n")

    random_item_ids = random.sample(
        valid_item_ids, min(num_examples, len(valid_item_ids))
    )

    for idx, target_id in enumerate(random_item_ids, 1):
        target_idx = model.item_mapping[target_id]
        target_reviews = int(item_interaction_counts[target_idx])

        print(f"Example {idx}: Restaurant {target_id} ({target_reviews} reviews)")

        # Get more candidates than needed
        similar_items = model.find_similar_items(
            target_item_id=target_id,
            top_k=top_k * 5,
            method="cosine_matrix",
        )

        if not similar_items:
            print("  No similar restaurants found\n")
            continue

        # Filter out items with very low similarity
        non_zero_items = [
            item for item in similar_items if item["similarity_score"] > 0.0001
        ]

        if not non_zero_items:
            print("  No meaningful similarity found\n")
            continue

        # Prioritize items with >= min_interactions, but fall back if needed
        high_quality_items = [
            item
            for item in non_zero_items
            if item_interaction_counts[model.item_mapping[item["item_id"]]]
            >= min_interactions
        ]

        # Use high quality items if available, otherwise use all non-zero items
        display_items = high_quality_items if high_quality_items else non_zero_items

        # Ensure we have at least top_k items (or as many as available)
        display_items = display_items[:top_k]

        for i, item in enumerate(display_items, 1):
            similar_idx = model.item_mapping[item["item_id"]]
            similar_reviews = int(item_interaction_counts[similar_idx])
            print(
                f"  {i}. ID {item['item_id']:10d} | "
                f"Sim: {item['similarity_score']:.4f} | "
                f"Reviews: {similar_reviews}"
            )
        print()


def save_all_similar_items(
    top_k: int = 10,
    method: str = "cosine_matrix",
    min_interactions: int = 10,
    cf_weight: float = 0.8,
    content_weight: float = 0.2,
):
    """
    Find and save similar top-k restaurants for all restaurants using hybrid method.
    Target: Restaurants with more than 1 review
    Candidate: Only restaurants with at least min_interactions reviews

    Args:
        top_k: Number of similar restaurants to find for each restaurant (default: 10)
        method: Similarity calculation method - "cosine_matrix" or "jaccard"
        min_interactions: Minimum reviews for candidate restaurants (default: 10)
        cf_weight: Collaborative filtering similarity weight (default: 0.8)
        content_weight: Content-based similarity weight (default: 0.2)
    """
    import json

    model, data, item_interaction_counts = load_model_for_inference()

    # Target: restaurants with more than 1 review
    target_indices = np.where(item_interaction_counts > 1)[0]
    all_item_ids = [
        model.idx_to_item[idx] for idx in target_indices if idx in model.idx_to_item
    ]

    # Candidate: restaurants with sufficient interactions
    valid_candidate_indices = np.where(item_interaction_counts >= min_interactions)[0]
    valid_candidate_ids = set(
        [
            model.idx_to_item[idx]
            for idx in valid_candidate_indices
            if idx in model.idx_to_item
        ]
    )

    print(f"Target restaurants: {len(all_item_ids):,} (reviews > 1)")
    print(
        f"Candidate restaurants: {len(valid_candidate_ids):,} (reviews >= {min_interactions})"
    )

    results = {}
    warning_count = 0

    for i, target_id in enumerate(all_item_ids, 1):
        if i % 10000 == 0:
            print(f"Progress: {i}/{len(all_item_ids)}")

        # Find similar items using hybrid method
        similar_items = model.find_similar_items_hybrid(
            target_item_id=target_id,
            top_k=top_k * 5,
            cf_weight=cf_weight,
            content_weight=content_weight,
            method=method,
        )

        # Filter by candidate criteria and positive similarity
        filtered_items = [
            item
            for item in similar_items
            if item["item_id"] in valid_candidate_ids and item["hybrid_score"] > 0
        ]

        selected_items = filtered_items[:top_k]

        if len(selected_items) < top_k:
            if warning_count < 5:
                print(
                    f"  Warning: {target_id} has only {len(selected_items)} similar items"
                )
            warning_count += 1

        results[target_id] = [
            [item["item_id"], round(item["hybrid_score"], 2)] for item in selected_items
        ]

    # Save results
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = RESULT_PATH.format(test="similarity", model="item_based", dt=dt)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"similar_items_hybrid_top{top_k}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Statistics
    total_pairs = sum(len(v) for v in results.values())
    if total_pairs > 0:
        avg_similarity = np.mean(
            [score for items in results.values() for _, score in items if items]
        )
        avg_items_per_target = total_pairs / len(results)
        print(f"Total similar pairs: {total_pairs:,}")
        print(f"Average similar items per target: {avg_items_per_target:.1f}")
        print(f"Average similarity score: {avg_similarity:.4f}")

    return output_file


def main() -> None:
    """Train and evaluate Item-based Collaborative Filtering model."""
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    test_flag = "untest"
    result_path = RESULT_PATH.format(test=test_flag, model="item_based", dt=dt)
    os.makedirs(result_path, exist_ok=True)

    config = load_yaml(CONFIG_PATH.format(model="als"))
    preprocess_config = load_yaml(PREPROCESS_CONFIG_PATH)

    save_command_to_file(result_path)

    top_k_values_for_pred = config.training.evaluation.top_k_values_for_pred
    top_k_values_for_candidate = config.training.evaluation.top_k_values_for_candidate
    top_k_values = top_k_values_for_pred + top_k_values_for_candidate
    file_name = config.post_training.file_name
    fe = config.preprocess.feature_engineering

    logger = setup_logger(os.path.join(result_path, file_name.log))

    try:
        logger.info("model: item_based")
        logger.info(f"results will be saved in {result_path}")

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

        logging_data_statistics(config, data, logger)

        logger.info("Initializing Item-based Collaborative Filtering model...")
        item_based_model = ItemBasedCollaborativeFiltering(
            user_item_matrix=data["X_train"],
            user_mapping=data["user_mapping"],
            item_mapping=data["diner_mapping"],
            diner_df=None,
        )

        data["test_dict"] = {
            "train": data["X_train_df"],
            "test": pd.concat(
                [data["X_test_warm_users"], data["X_test_cold_users"]],
                ignore_index=True,
            ),
        }

        logger.info("Evaluating Item-based CF model...")
        metric_calculator = ItemBasedMetricCalculator(
            model=item_based_model,
            test_data=data["test_dict"],
            top_k_values=top_k_values,
            logger=logger,
        )

        metrics = metric_calculator.evaluate()

        logger.info("Evaluation results:")
        for metric_name, score in metrics.items():
            logger.info(f"{metric_name}: {score:.4f}")

    except Exception as e:
        logger.error("An error occurred during training")
        logger.error(traceback.format_exc())
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Item-based Collaborative Filtering")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    train_parser = subparsers.add_parser("train", help="Train and evaluate model")

    demo_parser = subparsers.add_parser(
        "demo", help="Find similar items for random restaurants"
    )
    demo_parser.add_argument("--top_k", type=int, default=10)
    demo_parser.add_argument("--num_examples", type=int, default=3)
    demo_parser.add_argument("--min_interactions", type=int, default=10)

    find_parser = subparsers.add_parser(
        "find", help="Find similar restaurants for a specific restaurant ID"
    )
    find_parser.add_argument("--target_id", type=int, required=True)
    find_parser.add_argument("--top_k", type=int, default=10)
    find_parser.add_argument(
        "--method",
        type=str,
        choices=["cosine_matrix", "jaccard"],
        default="cosine_matrix",
    )
    find_parser.add_argument(
        "--min_interactions",
        type=int,
        default=10,
        help="Minimum reviews for candidate restaurants",
    )

    save_all_parser = subparsers.add_parser(
        "save_all", help="Save all similar items using hybrid method"
    )
    save_all_parser.add_argument("--top_k", type=int, default=10)
    save_all_parser.add_argument(
        "--method",
        type=str,
        default="cosine_matrix",
        choices=["cosine_matrix", "jaccard"],
    )
    save_all_parser.add_argument("--min_interactions", type=int, default=10)
    save_all_parser.add_argument(
        "--cf_weight", type=float, default=0.8, help="CF similarity weight"
    )
    save_all_parser.add_argument(
        "--content_weight", type=float, default=0.2, help="Content similarity weight"
    )

    args = parser.parse_args()

    if args.command == "train":
        main()
    elif args.command == "demo":
        find_similar_items_demo(
            top_k=args.top_k,
            num_examples=args.num_examples,
            min_interactions=args.min_interactions,
        )
    elif args.command == "find":
        find_similar_items_cli(
            target_id=args.target_id,
            top_k=args.top_k,
            method=args.method,
            min_interactions=args.min_interactions,
        )
    elif args.command == "save_all":
        save_all_similar_items(
            top_k=args.top_k,
            method=args.method,
            min_interactions=args.min_interactions,
            cf_weight=args.cf_weight,
            content_weight=args.content_weight,
        )
    else:
        main()
