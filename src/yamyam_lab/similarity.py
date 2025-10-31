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
    """
    Load model for inference (data loading + model initialization).

    Returns:
        tuple: (model, data, item_interaction_counts)
    """
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
        item_embeddings=None,
        user_mapping=data["user_mapping"],
        item_mapping=data["diner_mapping"],
        diner_df=None,
    )

    item_interaction_counts = np.array(model.item_user_matrix.sum(axis=1)).flatten()

    return model, data, item_interaction_counts


def find_similar_items_cli(
    target_id: int, top_k: int = 10, method: str = "cosine_matrix"
):
    """
    Find and display similar restaurants for a specific restaurant ID via CLI.

    Args:
        target_id: Target restaurant ID
        top_k: Number of similar restaurants to return (default: 10)
        method: Similarity calculation method - "cosine_matrix" or "jaccard"
    """
    print("=" * 70)
    print(f"Finding Similar Restaurants for ID: {target_id}")
    print("=" * 70)

    print("\n[1/2] Loading model...")
    model, data, item_interaction_counts = load_model_for_inference()
    print(
        f"  ✓ Loaded {data['X_train'].shape[0]:,} users × {data['X_train'].shape[1]:,} items"
    )

    if target_id not in model.item_mapping:
        print(f"\n❌ Error: Restaurant ID {target_id} not found in training data!")
        return

    target_idx = model.item_mapping[target_id]
    target_reviews = int(item_interaction_counts[target_idx])
    print(f"  ✓ Target Restaurant: ID {target_id} ({target_reviews} reviews)")

    print(f"\n[2/2] Finding top {top_k} similar restaurants (method: {method})...")
    print("-" * 70)

    similar_items = model.find_similar_items(
        target_item_id=target_id,
        top_k=top_k,
        method=method,
    )

    if not similar_items:
        print("  ⚠️  No similar restaurants found")
        return

    non_zero_items = [
        item for item in similar_items if item["similarity_score"] > 0.0001
    ]

    if not non_zero_items:
        print("  ⚠️  No restaurants with meaningful similarity (all scores ≈ 0)")
        print("      Try using a more popular restaurant or --method jaccard")
        return

    print(
        f"\n📊 Similar Restaurants (Top {len(non_zero_items)} with similarity > 0):\n"
    )

    for i, item in enumerate(non_zero_items, 1):
        similar_idx = model.item_mapping[item["item_id"]]
        similar_reviews = int(item_interaction_counts[similar_idx])
        print(f"  {i:2d}. Restaurant ID: {item['item_id']:10d}")
        print(
            f"      Similarity: {item['similarity_score']:.6f}  |  Reviews: {similar_reviews:,}"
        )

    print("\n" + "=" * 70)
    similar_ids = [item["item_id"] for item in non_zero_items]
    print(f"📋 Similar Restaurant IDs: {similar_ids}")
    print("=" * 70)


def find_similar_items_demo(
    top_k: int = 10, num_examples: int = 3, min_interactions: int = 20
):
    """
    Demo function that randomly selects restaurants and finds similar items.

    Args:
        top_k: Number of similar items to show per restaurant (default: 10)
        num_examples: Number of random restaurants to test (default: 3)
        min_interactions: Minimum number of reviews for filtering (default: 20)
    """
    print("=" * 70)
    print("Item-Based Collaborative Filtering - Similar Items Demo")
    print("=" * 70)

    print("\n[1/2] Loading model and filtering restaurants...")
    model, data, item_interaction_counts = load_model_for_inference()
    print(
        f"  ✓ Loaded {data['X_train'].shape[0]:,} users × {data['X_train'].shape[1]:,} items"
    )

    valid_indices = np.where(item_interaction_counts >= min_interactions)[0]

    if len(valid_indices) == 0:
        print(f"⚠️  No items with at least {min_interactions} interactions found!")
        return

    valid_item_ids = [
        model.idx_to_item[idx] for idx in valid_indices if idx in model.idx_to_item
    ]
    print(
        f"  ℹ️  Filtering: {len(valid_item_ids):,} items with ≥{min_interactions} reviews"
    )

    random_item_ids = random.sample(
        valid_item_ids, min(num_examples, len(valid_item_ids))
    )

    print(
        f"\n[2/2] Finding similar restaurants for {num_examples} random restaurants..."
    )
    print("=" * 70)

    for idx, target_id in enumerate(random_item_ids, 1):
        target_idx = model.item_mapping[target_id]
        target_reviews = int(item_interaction_counts[target_idx])

        print(
            f"\n🍽️  Example {idx}: Restaurant ID {target_id} ({target_reviews} reviews)"
        )
        print("-" * 70)

        similar_items = model.find_similar_items(
            target_item_id=target_id,
            top_k=top_k,
            method="cosine_matrix",
        )

        if not similar_items:
            print("  ⚠️  No similar restaurants found")
            continue

        non_zero_items = [
            item for item in similar_items if item["similarity_score"] > 0.0001
        ]

        if not non_zero_items:
            print("  ⚠️  No restaurants with meaningful similarity")
            continue

        print(f"  Similar restaurants (Top {len(non_zero_items)} with similarity > 0):")
        for i, item in enumerate(non_zero_items, 1):
            similar_idx = model.item_mapping[item["item_id"]]
            similar_reviews = int(item_interaction_counts[similar_idx])
            print(
                f"    {i:2d}. Restaurant ID: {item['item_id']:10d}  |  "
                f"Similarity: {item['similarity_score']:.4f}  |  "
                f"Reviews: {similar_reviews:4d}"
            )

    print("\n" + "=" * 70)
    print("✅ Demo completed!")
    print("=" * 70)


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
            item_embeddings=None,
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

    # Train command
    train_parser = subparsers.add_parser("train", help="Train and evaluate model")

    # Demo command
    demo_parser = subparsers.add_parser(
        "demo", help="Find similar items for random restaurants"
    )
    demo_parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of similar items to show per restaurant (default: 10)",
    )
    demo_parser.add_argument(
        "--num_examples",
        type=int,
        default=3,
        help="Number of random restaurants to test (default: 3)",
    )
    demo_parser.add_argument(
        "--min_interactions",
        type=int,
        default=20,
        help="Minimum number of reviews for a restaurant to be selected (default: 20)",
    )

    # Find command
    find_parser = subparsers.add_parser(
        "find", help="Find similar restaurants for a specific restaurant ID"
    )
    find_parser.add_argument(
        "--target_id",
        type=int,
        required=True,
        help="Target restaurant ID to find similar restaurants for",
    )
    find_parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of similar restaurants to return (default: 10)",
    )
    find_parser.add_argument(
        "--method",
        type=str,
        choices=["cosine_matrix", "jaccard"],
        default="cosine_matrix",
        help="Similarity calculation method (default: cosine_matrix)",
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
        )
    else:
        main()
