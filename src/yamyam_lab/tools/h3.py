from typing import List, Tuple

import h3


def get_h3_index(lat: float, long: float, resolution: int) -> str:
    """
    Get h3 index which covers given latitude and longitude.

    Args:
        lat (float): Latitude value in float.
        long (float): Longitude value in float.
        resolution (int): Resolution value of h3 index controlling size of hexagon.

    Returns (str):
        h3 index.
    """
    return h3.latlng_to_cell(lat, long, resolution)


def get_hexagon_boundary_coordinate(h3_index: str) -> Tuple[Tuple[float, float]]:
    """
    Get hexagon boundary coordinate consisting of latitude and longitude.

    Args:
        h3_index (str): H3 index from `get_h3_index` function.

    Returns (Tuple[Tuple[float, float]]):
        6 coordinates of hexagon.
    """
    return h3.cell_to_boundary(h3_index)


def get_hexagon_neighbors(h3_index: str, k: int) -> List[str]:
    """
    Get neighbors of h3_index whose hop value is integer k.

    Args:
        h3_index (str): H3 index from `get_h3_index` function.
        k (str): Number of hops for neighbors of given h3_index.

    Returns (List[str]):
        List of h3_index hexagon neighbors.
    """
    return h3.grid_ring(h3_index, k)


def get_center_coordinate(coordinate: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Get center coordinate given list of coordinates.

    Args:
        coordinate (List[Tuple[float, float]]): List of coordinates.

    Returns (Tuple[float, float]):
        Center coordinate.
    """
    center_lat = sum(lat for lat, _ in coordinate) / len(coordinate)
    center_lon = sum(lon for _, lon in coordinate) / len(coordinate)
    return center_lat, center_lon
