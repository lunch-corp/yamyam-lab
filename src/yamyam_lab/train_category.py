"""
Train category classifier using the classification module.

Usage:
    poetry run python -m yamyam_lab.train_category --model catboost
    poetry run python -m yamyam_lab.train_category --model catboost --force-reload
    poetry run python -m yamyam_lab.train_category --test  # Quick test with 2000 samples, 10 iterations
"""

import argparse
from datetime import datetime

from yamyam_lab.model.classification import CatBoostCategoryClassifier
from yamyam_lab.model.embedding import TfidfEmbedder


def main(args):
    """Main training function."""

    # Generate output directory with datetime
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_dir = "test" if args.test else "untest"
    output_dir = f"result/{mode_dir}/category_classifier/{args.model}/{dt}"

    # Apply test mode overrides
    sample_size = None
    if args.test:
        print("=" * 60)
        print("TEST MODE: Using reduced data and iterations")
        print("=" * 60)
        sample_size = 2000
        args.iterations = min(args.iterations, 10)
        args.verbose = 1
        args.cache_dir = args.cache_dir + "/test"

    print(f"Output directory: {output_dir}")

    # Create embedder
    embedder = TfidfEmbedder(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(1, 2),
        cache_dir=args.cache_dir,
    )

    # Create classifier based on model type
    if args.model == "catboost":
        classifier = CatBoostCategoryClassifier(
            embedder=embedder,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose=args.verbose,
            output_dir=output_dir,
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            use_menu=args.use_menu,
            use_large_category=args.use_large_category,
            val_ratio=args.val_ratio,
            min_class_samples=args.min_class_samples,
            sample_size=sample_size,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Run training (force reload in test mode to avoid cache mismatch)
    result = classifier.run(
        force_reload_data=args.force_reload or args.test,
        force_recompute_embeddings=args.force_embeddings or args.test,
    )

    print("\nFinal Results:")
    print(f"  Accuracy: {result.accuracy:.4f}")
    print(f"  F1 (macro): {result.f1_macro:.4f}")
    print(f"  F1 (weighted): {result.f1_weighted:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train category classifier")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["catboost"],  # Add more as implemented
        default="catboost",
        help="Model type to use",
    )

    # Data paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing data files",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/processed_category",
        help="Directory for caching preprocessed data",
    )

    # Data preprocessing
    parser.add_argument(
        "--use-menu",
        action="store_true",
        default=True,
        help="Use menu names as features",
    )
    parser.add_argument(
        "--no-menu",
        dest="use_menu",
        action="store_false",
        help="Don't use menu names",
    )
    parser.add_argument(
        "--use-large-category",
        action="store_true",
        default=True,
        help="Use large category as feature",
    )
    parser.add_argument(
        "--no-large-category",
        dest="use_large_category",
        action="store_false",
        help="Don't use large category as feature",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--min-class-samples",
        type=int,
        default=10,
        help="Minimum samples per class",
    )

    # TF-IDF settings
    parser.add_argument(
        "--max-features",
        type=int,
        default=20000,
        help="Max TF-IDF features",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Min document frequency",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.95,
        help="Max document frequency",
    )

    # CatBoost settings
    parser.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="Number of boosting iterations",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="Tree depth",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=30,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=50,
        help="Logging frequency",
    )

    # Cache control
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force reload all data (ignore cache)",
    )
    parser.add_argument(
        "--force-embeddings",
        action="store_true",
        help="Force recompute embeddings",
    )

    # Test mode
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test mode: sample 2000 rows, 10 iterations",
    )

    args = parser.parse_args()
    main(args)
