"""Unit tests for NearCandidateGenerator."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from yamyam_lab.candidate.near import NearCandidateGenerator


@pytest.fixture
def mock_diner_data():
    """Create mock diner DataFrame with known coordinates."""
    return pd.DataFrame(
        {
            "diner_idx": [100, 200, 300, 400, 500],
            "diner_lat": [37.5665, 37.5672, 37.5660, 37.5800, 37.6000],
            "diner_lon": [126.9780, 126.9785, 126.9775, 126.9900, 127.0200],
        }
    )


@pytest.fixture
def generator(mock_diner_data):
    """Create NearCandidateGenerator with mocked data loading."""
    with (
        patch("yamyam_lab.candidate.near.check_data_and_return_paths") as mock_paths,
        patch("pandas.read_csv", return_value=mock_diner_data),
    ):
        mock_paths.return_value = {"diner": "mock_path.csv"}
        gen = NearCandidateGenerator()
    return gen


class TestNearCandidateGenerator:
    """Tests for NearCandidateGenerator."""

    def test_init_creates_kd_tree(self, generator):
        """Generator initializes KDTree from diner coordinates."""
        assert generator.kd_tree is not None
        assert len(generator.mapping_diner_idx) == 5

    def test_mapping_diner_idx(self, generator):
        """Mapping correctly maps array indices to original diner IDs."""
        assert generator.mapping_diner_idx[0] == 100
        assert generator.mapping_diner_idx[4] == 500

    def test_get_max_distance_rad(self, generator):
        """Converts km to radians correctly."""
        rad = generator.get_max_distance_rad(6371)
        assert np.isclose(rad, 1.0)

        rad_zero = generator.get_max_distance_rad(0)
        assert rad_zero == 0.0

    def test_get_near_candidate_returns_nearby_diners(self, generator):
        """Nearby diners within distance threshold are returned."""
        # Query from first diner's location with small radius
        # Diners 100, 200, 300 are very close (~0.1km apart)
        result = generator.get_near_candidate(
            latitude=37.5665, longitude=126.9780, max_distance_km=1.0
        )
        assert 100 in result
        assert 200 in result
        assert 300 in result

    def test_get_near_candidate_excludes_far_diners(self, generator):
        """Diners beyond distance threshold are excluded."""
        # Diner 500 is ~4km away from diner 100
        result = generator.get_near_candidate(
            latitude=37.5665, longitude=126.9780, max_distance_km=0.5
        )
        assert 500 not in result

    def test_get_near_candidate_zero_distance(self, generator):
        """Zero distance returns only diners at exact location."""
        result = generator.get_near_candidate(
            latitude=37.5665, longitude=126.9780, max_distance_km=0.0
        )
        assert len(result) == 0 or result == [100]

    def test_get_near_candidate_large_distance(self, generator):
        """Large distance returns all diners."""
        result = generator.get_near_candidate(
            latitude=37.5665, longitude=126.9780, max_distance_km=100.0
        )
        assert len(result) == 5

    def test_diner_coords_in_radians(self, generator):
        """Diner coordinates are stored in radians."""
        assert generator.diner_coords[0][0] < 1.0  # ~0.656 rad
        assert generator.diner_coords[0][1] > 2.0  # ~2.216 rad
