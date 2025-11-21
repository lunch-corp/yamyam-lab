import itertools
import os
import traceback
from datetime import datetime
from typing import Any

import pandas as pd

from yamyam_lab.postprocess.hidden import HiddenReranker
from yamyam_lab.tools.logger import setup_logger

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
RESULT_PATH = os.path.join(ROOT_PATH, "result", "hidden_reranker")

# Ensure result directory exists
os.makedirs(RESULT_PATH, exist_ok=True)


class HiddenRerankerTester:
    """Tester class for HiddenReranker with hyperparameter experiments."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or self._default_config()
        self.logger = None
        self.dt = datetime.now().strftime("%Y%m%d%H%M%S")
        self.datasets = {}

    def _default_config(self) -> dict[str, Any]:
        return {
            "hyperparams": {
                "n_auto_hotspots": [5, 10],
                "periphery_strength": [0.7],
                "periphery_cap": [0.3, 0.5],
                "rating_weight": [0.2, 0.3],
                "recent_weight": [0.1, 0.2],
            },
            "k": 1000,
            "high_rating_threshold": 4.0,
            "top_n": 20,
        }

    def setup_logger(self):
        log_file = os.path.join(RESULT_PATH, f"hidden_reranker_test_{self.dt}.log")
        self.logger = setup_logger(log_file)
        self.logger.info("Testing HiddenReranker")
        self.logger.info(f"Results will be saved in {RESULT_PATH}")

    def load_data(self):
        """Load and preprocess datasets."""
        self.logger.info("Loading data...")
        diner_df = pd.read_csv("data/diner.csv")
        diner_category_df = pd.read_csv("data/diner_category_raw.csv")
        review_df = pd.read_csv("data/review.csv")

        # Convert dates
        review_df["reviewer_review_date"] = pd.to_datetime(
            review_df["reviewer_review_date"]
        )
        max_date = review_df["reviewer_review_date"].max()
        three_months_ago = max_date - pd.DateOffset(months=3)
        recent_review_df = review_df[
            review_df["reviewer_review_date"] >= three_months_ago
        ]

        # Merge for full data
        self.logger.info("Merging full data...")
        diner_data_full = self._merge_data(diner_df, diner_category_df, review_df)

        # Merge for recent 3 months data
        self.logger.info("Merging recent 3 months data...")
        diner_data_recent = self._merge_data(
            diner_df, diner_category_df, recent_review_df
        )

        self.datasets = {"full": diner_data_full, "recent_3m": diner_data_recent}

    def _merge_data(
        self,
        diner_df: pd.DataFrame,
        diner_category_df: pd.DataFrame,
        review_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Helper method to merge diner, category, and review data."""
        merged = diner_df.merge(diner_category_df, on="diner_idx", how="left")
        merged = merged.merge(
            review_df.groupby("diner_idx")
            .agg(
                avg_rating=("reviewer_review_score", "mean"),
                recent_review_count=(
                    "reviewer_review_date",
                    lambda x: (pd.to_datetime("today") - pd.to_datetime(x))
                    .dt.days.le(30)
                    .sum(),
                ),
            )
            .reset_index(),
            on="diner_idx",
            how="left",
        )
        return merged

    def generate_experiments(self) -> list[dict[str, Any]]:
        """Generate all combinations of hyperparameters."""
        keys = self.config["hyperparams"].keys()
        values = self.config["hyperparams"].values()
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def run_experiments(self):
        """Run all experiments on all datasets."""
        experiments = self.generate_experiments()
        self.experiment_results = []  # Store results for best parameter selection
        for dataset_name, diner_data in self.datasets.items():
            self.logger.info(
                f"Testing on {dataset_name} dataset with {len(diner_data)} diners"
            )
            for exp in experiments:
                result = self._run_single_experiment(dataset_name, diner_data, exp)
                self.experiment_results.append(
                    {"dataset": dataset_name, "params": exp, "results": result}
                )

    def _run_single_experiment(
        self, dataset_name: str, diner_data: pd.DataFrame, exp: dict[str, Any]
    ) -> dict[str, Any]:
        """Run a single experiment."""
        self.logger.info(f"Experiment: {', '.join(f'{k}={v}' for k, v in exp.items())}")

        # Initialize reranker
        reranker = HiddenReranker(**exp)

        # Perform reranking
        reranked_df = reranker.rerank(diner_data, k=self.config["k"])

        # Analyze results
        results = self._analyze_results(reranked_df)

        # Log results
        for region, high_count in results["region_high_counts"].items():
            self.logger.info(
                f"  {region}: {high_count}/{self.config['top_n']} high rating (>={self.config['high_rating_threshold']})"
            )
        self.logger.info(
            f"  Overall high rating ratio: {results['avg_high_rating']:.2f} ({results['total_high_rating']}/{results['total_regions'] * self.config['top_n']})"
        )
        self.logger.info(
            f"  Improvement metric: {results['avg_high_rating'] * 100:.1f}% high rating in top {self.config['top_n']}"
        )

        # Save results
        self._save_results(dataset_name, exp, reranked_df)

        return results

    def _analyze_results(self, reranked_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze reranked results."""
        grouped = reranked_df.groupby("region")
        total_high_rating = 0
        total_regions = 0
        region_high_counts = {}
        for region, group in grouped:
            top_n = group.head(self.config["top_n"])
            top_n_high = top_n[
                top_n["avg_rating"] >= self.config["high_rating_threshold"]
            ]
            high_count = len(top_n_high)
            region_high_counts[region] = high_count
            total_high_rating += high_count
            total_regions += 1

        avg_high_rating = (
            total_high_rating / (total_regions * self.config["top_n"])
            if total_regions > 0
            else 0
        )
        return {
            "total_high_rating": total_high_rating,
            "total_regions": total_regions,
            "avg_high_rating": avg_high_rating,
            "region_high_counts": region_high_counts,
        }

    def _select_best_parameters(self):
        """Select the best parameters based on high rating ratio and periphery emphasis."""
        if not self.experiment_results:
            return

        # Calculate scores for each experiment
        scored_results = []
        for exp_result in self.experiment_results:
            params = exp_result["params"]
            results = exp_result["results"]
            dataset = exp_result["dataset"]

            # Score based on high rating ratio (higher is better)
            high_rating_score = results["avg_high_rating"]

            # Score based on periphery emphasis (higher periphery_strength and cap is better for hidden gems)
            periphery_score = params["periphery_strength"] + params["periphery_cap"]

            # Combined score: prioritize high rating, then periphery
            combined_score = high_rating_score * 0.6 + periphery_score * 0.4

            scored_results.append(
                {
                    "dataset": dataset,
                    "params": params,
                    "high_rating_ratio": results["avg_high_rating"],
                    "periphery_score": periphery_score,
                    "combined_score": combined_score,
                }
            )

        # Find best for each dataset
        for dataset in set(r["dataset"] for r in scored_results):
            dataset_results = [r for r in scored_results if r["dataset"] == dataset]
            best = max(dataset_results, key=lambda x: x["combined_score"])

            self.logger.info(f"Best parameters for {dataset} dataset:")
            self.logger.info(f"  Params: {best['params']}")
            self.logger.info(f"  High rating ratio: {best['high_rating_ratio']:.2f}")
            self.logger.info(f"  Periphery score: {best['periphery_score']:.2f}")
            self.logger.info(f"  Combined score: {best['combined_score']:.2f}")

    def _save_results(
        self, dataset_name: str, exp: dict[str, Any], reranked_df: pd.DataFrame
    ):
        """Save experiment results to CSV."""
        params_str = "_".join(f"{k}{v}" for k, v in exp.items())
        output_file = os.path.join(
            RESULT_PATH, f"hidden_reranker_{dataset_name}_{params_str}_{self.dt}.csv"
        )
        reranked_df.to_csv(output_file, index=False)
        self.logger.info(f"  Results saved to {output_file}")

    def run(self):
        """Main execution method."""
        try:
            self.setup_logger()
            self.load_data()
            self.run_experiments()
            self._select_best_parameters()
            self.logger.info("All experiments completed")
        except Exception as e:
            if self.logger:
                self.logger.error("An error occurred during testing")
                self.logger.error(traceback.format_exc())
            raise e


def main():
    tester = HiddenRerankerTester()
    tester.run()


if __name__ == "__main__":
    main()
