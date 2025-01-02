import os

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial import KDTree

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")


class NearCandidateGenerator:
    def __init__(self):
        diners = pd.read_csv(os.path.join(DATA_PATH, "diner/diner_df_20241219_yamyam.csv"))
        diner_ids = diners["diner_idx"].unique()
        self.mapping_diner_idx = {i: id for i, id in enumerate(diner_ids)}

        # Convert latitude and longitude to radians for KDTree
        self.diner_coords = np.radians([(r[1]["diner_lat"], r[1]["diner_lon"]) for r in diners.iterrows()])

        # get kd tree
        self.kd_tree = self.create_kd_tree()

    def create_kd_tree(self) -> KDTree:
        """
        Create a KDTree in advance.
        """
        # Create KDTree
        tree = KDTree(self.diner_coords)
        return tree

    def get_near_candidate(
        self,
        coord: NDArray,
        max_distance_km: float,
        is_radians: bool = True,
    ) -> NDArray:
        """
        Get near max_distance_km diners given user's coordinate.

        Args:
            coord (NDArray): Should be cautious of ordering (latitude, longitude).
                Input could be radians or raw coordinates which is converted to
                radians depending on `is_radians` argument.
            max_distance_km (float): Based on `coord`, distance of how close diners want to get.
            is_radians (bool): Whether `coord` is converted to radians already or not.
        """
        # if coordinate is raw (latitude, longitude), should convert to radians
        if is_radians is False:
            coord = np.radians(coord)
        max_distance_rad = self.get_max_distance_rad(max_distance_km)
        near_diner_ids = self.kd_tree.query_ball_point(coord, max_distance_rad)
        return near_diner_ids

    def get_near_candidates_for_all_diners(self, max_distance_km: float) -> dict[int, list[int]]:
        """
        Get near candidates for all of diners in dataset

        Args:
            max_distance_km (float): Based on `coord`, distance of how close diners want to get.
        """
        # For each of diner, query KDTree for diners within max_distance_rad
        result = {}
        for i, coord_rad in enumerate(self.diner_coords):
            # map to original diner_id: 0 -> 879135
            ref_diner_id = self.mapping_diner_idx[i]
            # Note: `indices` include referenced diner itself
            near_diner_ids = self.get_near_candidate(
                coord=coord_rad,
                max_distance_km=max_distance_km,
                is_radians=True,
            )
            result[ref_diner_id] = [self.mapping_diner_idx[id] for id in near_diner_ids]

        return result

    def get_max_distance_rad(self, max_distance_km: float) -> float:
        """
        Convert kilometer to radians

        Args:
            max_distance_km (float): Based on `coord`, distance of how close diners want to get.

        Returns (float):
            Converted radians value.
        """
        # Earth's radius in kilometers
        earth_radius_km = 6371

        # Convert max_distance_km to radians
        max_distance_rad = max_distance_km / earth_radius_km

        return max_distance_rad


if __name__ == "__main__":
    candidates = NearCandidateGenerator()
    result = candidates.get_near_candidates_for_all_diners(max_distance_km=1)
