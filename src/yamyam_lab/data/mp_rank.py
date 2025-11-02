from datetime import datetime
from typing import List, Literal, Optional, Self, Union

import numpy as np
import pandas as pd

from yamyam_lab.data.base import BaseDatasetLoader
from yamyam_lab.data.config import DataConfig
from yamyam_lab.inference import haversine


class MostPopularRankDataLoader(BaseDatasetLoader):
    """
    DataLoader for Most Popular Ranking based recommendation.

    This class provides ranking functionality based on various criteria such as
    rating, review count, distance, and custom popularity metrics.
    """

    def __init__(
        self: Self,
        data_config: DataConfig,
        diner_category_large: Optional[str] = None,
        diner_category_middle: Optional[str] = None,
        rank_method: Literal[
            "rating", "review_count", "distance", "yamyam_popularity"
        ] = "rating",
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        period: Literal["1M", "3M", "6M", "all"] = "all",
        reference_date: Optional[Union[str, datetime]] = None,
        topk: int = 10,
        min_review_count: int = 5,
    ):
        """
        Initialize MostPopularRankDataLoader.

        Args:
            data_config: Configuration for dataset loading
            diner_category_large: Large category filter (e.g., "한식", "일식")
            diner_category_middle: Middle category filter (e.g., "고기구이", "라멘")
            rank_method: Ranking method - "rating", "review_count", "distance", "yamyam_popularity"
            lat: Latitude for distance-based ranking
            lon: Longitude for distance-based ranking
            period: Time period for filtering recent reviews - "1M", "3M", "6M", "all"
            reference_date: Reference date for period calculation (default: latest date in data)
                           Can be string ("2024-11-02") or datetime object
            topk: Number of top diners to return
        """
        super().__init__(data_config)

        self.diner_category_large = diner_category_large
        self.diner_category_middle = diner_category_middle
        self.rank_method = rank_method
        self.lat = lat
        self.lon = lon
        self.period = period
        self.reference_date = self._parse_reference_date(reference_date)
        self.topk = topk
        self.min_review_count = min_review_count

        # Convert period to days
        self.period_days = self._convert_period_to_days(period)

        self._validate_rank_params()

    def _parse_reference_date(
        self: Self, reference_date: Optional[Union[str, datetime]]
    ) -> Optional[pd.Timestamp]:
        """
        Parse reference date to pandas Timestamp.

        Args:
            reference_date: Reference date as string or datetime

        Returns:
            Parsed pandas Timestamp or None
        """
        if reference_date is None:
            return None

        if isinstance(reference_date, str):
            return pd.to_datetime(reference_date)
        elif isinstance(reference_date, datetime):
            return pd.Timestamp(reference_date)
        else:
            return pd.Timestamp(reference_date)

    def _convert_period_to_days(self: Self, period: str) -> Optional[int]:
        """
        Convert period string to number of days.

        Args:
            period: Period string ("1M", "3M", "6M", "all")

        Returns:
            Number of days or None for "all"
        """
        period_mapping = {"1M": 30, "3M": 90, "6M": 180, "all": None}
        return period_mapping.get(period)

    def _validate_rank_params(self: Self) -> None:
        """Validate ranking parameters."""
        if self.rank_method == "distance":
            if self.lat is None or self.lon is None:
                raise ValueError(
                    "lat and lon must be provided when rank_method is 'distance'"
                )

        if self.rank_method not in [
            "rating",
            "review_count",
            "distance",
            "yamyam_popularity",
        ]:
            raise ValueError(
                f"Invalid rank_method: {self.rank_method}. "
                "Must be one of 'rating', 'review_count', 'distance', 'yamyam_popularity'"
            )

        if self.period not in ["1M", "3M", "6M", "all"]:
            raise ValueError(
                f"Invalid period: {self.period}. Must be one of '1M', '3M', '6M', 'all'"
            )

        if self.topk <= 0:
            raise ValueError("topk must be greater than 0")

        if self.min_review_count < 0:
            raise ValueError("min_review_count must be non-negative")

    def _filter_by_category(
        self: Self, diner: pd.DataFrame, diner_with_raw_category: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter diners by category.

        Args:
            diner: Diner dataframe
            diner_with_raw_category: Diner dataframe with raw category information

        Returns:
            Filtered diner dataframe
        """
        # Merge to get category information
        diner_merged = pd.merge(
            diner,
            diner_with_raw_category[
                ["diner_idx", "diner_category_large", "diner_category_middle"]
            ],
            on="diner_idx",
            how="left",
        )

        # Apply filters
        if self.diner_category_large:
            diner_merged = diner_merged[
                diner_merged["diner_category_large"] == self.diner_category_large
            ]

        if self.diner_category_middle:
            diner_merged = diner_merged[
                diner_merged["diner_category_middle"] == self.diner_category_middle
            ]

        return diner_merged

    def _filter_by_period(self: Self, review: pd.DataFrame) -> pd.DataFrame:
        """
        Filter reviews by time period from reference date.

        Args:
            review: Review dataframe

        Returns:
            Filtered review dataframe
        """
        if self.period_days is None:  # "all"
            return review

        # Convert to datetime if not already
        # review = review.copy()
        review["reviewer_review_date"] = pd.to_datetime(review["reviewer_review_date"])

        # Use reference_date if provided, otherwise use the most recent date in data
        if self.reference_date is not None:
            today = self.reference_date
            print(f"Using reference date: {today.date()}")
        else:
            today = review["reviewer_review_date"].max()
            print(f"Using latest date in data as reference: {today.date()}")

        cutoff_date = today - pd.Timedelta(days=self.period_days)
        print(f"Filtering reviews from {cutoff_date.date()} to {today.date()}")

        return review[review["reviewer_review_date"] >= cutoff_date]

    def _calculate_distance(self: Self, diner: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate distance from given coordinates using haversine formula.

        Args:
            diner: Diner dataframe with latitude and longitude

        Returns:
            Diner dataframe with distance column
        """

        # Filter valid coordinates
        valid_coords = diner["diner_lat"].notna() & diner["diner_lon"].notna()

        if valid_coords.any():
            # Calculate distance using haversine
            distances = haversine(
                reviewer_lat=self.lat,
                reviewer_lon=self.lon,
                diner_lat=diner.loc[valid_coords, "diner_lat"],
                diner_lon=diner.loc[valid_coords, "diner_lon"],
            )

            # Initialize all distances as inf
            diner["distance"] = np.inf
            # Set valid distances
            diner.loc[valid_coords, "distance"] = distances
        else:
            diner["distance"] = np.inf

        return diner

    def _rank_by_rating(
        self: Self, review: pd.DataFrame, diner: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Rank diners by average rating.

        Args:
            review: Review dataframe
            diner: Diner dataframe

        Returns:
            Ranked diner dataframe
        """
        # Calculate average rating per diner
        rating_stats = (
            review.groupby("diner_idx")
            .agg(
                avg_rating=("reviewer_review_score", "mean"),
                review_count=("reviewer_review_score", "count"),
            )
            .reset_index()
        )

        rating_stats = rating_stats[
            rating_stats["review_count"] >= self.min_review_count
        ]

        # Merge with diner data
        diner_ranked = pd.merge(diner, rating_stats, on="diner_idx", how="inner")

        # Sort by rating (descending) and review count (descending) as tiebreaker
        diner_ranked = diner_ranked.sort_values(
            by=["avg_rating", "review_count"], ascending=[False, False]
        )

        return diner_ranked

    def _rank_by_review_count(
        self: Self, review: pd.DataFrame, diner: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Rank diners by review count.

        Args:
            review: Review dataframe
            diner: Diner dataframe

        Returns:
            Ranked diner dataframe
        """
        # Calculate review count per diner
        review_count = (
            review.groupby("diner_idx").size().reset_index(name="review_count")
        )

        # Merge with diner data
        diner_ranked = pd.merge(diner, review_count, on="diner_idx", how="inner")

        # Sort by review count (descending)
        diner_ranked = diner_ranked.sort_values(by="review_count", ascending=False)

        return diner_ranked

    def _rank_by_distance(self: Self, diner: pd.DataFrame) -> pd.DataFrame:
        """
        Rank diners by distance.

        Args:
            diner: Diner dataframe

        Returns:
            Ranked diner dataframe
        """
        # Calculate distance
        diner_ranked = self._calculate_distance(diner)

        # Sort by distance (ascending)
        diner_ranked = diner_ranked.sort_values(by="distance", ascending=True)

        # Remove diners with invalid coordinates
        diner_ranked = diner_ranked[diner_ranked["distance"] != np.inf]

        return diner_ranked

    def _rank_by_yamyam_popularity(
        self: Self, review: pd.DataFrame, diner: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Rank diners by yamyam custom popularity metric.
        Currently returns pass (to be implemented).

        Args:
            review: Review dataframe
            diner: Diner dataframe

        Returns:
            Empty dataframe (placeholder)
        """
        # 추후 추가 예정
        return pd.DataFrame()

    def load_topk_diners_rank(self: Self) -> List[int]:
        """
        Get top-k diner IDs based on specified ranking method and filters.

        Returns:
            List of top-k diner_idx
        """
        # Load data
        review, diner, diner_with_raw_category = self.load_dataset()

        # Apply category filter
        diner_filtered = self._filter_by_category(diner, diner_with_raw_category)

        if len(diner_filtered) == 0:
            return []

        # Apply period filter to reviews
        review_filtered = self._filter_by_period(review)

        # Filter reviews to only include diners in filtered diner set
        review_filtered = review_filtered[
            review_filtered["diner_idx"].isin(diner_filtered["diner_idx"])
        ]

        # Rank based on method
        if self.rank_method == "rating":
            diner_ranked = self._rank_by_rating(review_filtered, diner_filtered)
        elif self.rank_method == "review_count":
            diner_ranked = self._rank_by_review_count(review_filtered, diner_filtered)
        elif self.rank_method == "distance":
            diner_ranked = self._rank_by_distance(diner_filtered)
        elif self.rank_method == "yamyam_popularity":
            diner_ranked = self._rank_by_yamyam_popularity(
                review_filtered, diner_filtered
            )
            if len(diner_ranked) == 0:
                print("yamyam_popularity is not implemented yet. Returning empty list.")
                return []
        else:
            raise ValueError(f"Unknown rank_method: {self.rank_method}")

        # Return top-k diner IDs
        return diner_ranked["diner_idx"].head(self.topk).tolist()
