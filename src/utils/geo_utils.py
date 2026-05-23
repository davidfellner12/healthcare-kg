"""
src/utils/geo_utils.py

Geographic helper functions used across the project.
"""

import math
from typing import Tuple, List, Dict


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the great-circle distance in kilometres between two
    (latitude, longitude) points using the Haversine formula.
    """
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def walking_time_minutes(distance_km: float, speed_kmh: float = 4.5) -> float:
    """Convert a distance to estimated walking time in minutes."""
    return (distance_km / speed_kmh) * 60


def driving_time_minutes(distance_km: float, speed_kmh: float = 50.0) -> float:
    """Convert a distance to estimated driving time in minutes (urban average)."""
    return (distance_km / speed_kmh) * 60


def bounding_box_filter(points: List[Dict], bbox: Tuple[float, float, float, float]) -> List[Dict]:
    """
    Filter a list of {lat, lon, ...} dicts to those within a bounding box.
    bbox = (min_lon, min_lat, max_lon, max_lat)
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    return [
        p for p in points
        if min_lat <= float(p.get("lat", p.get("stop_lat", 0))) <= max_lat
        and min_lon <= float(p.get("lon", p.get("stop_lon", 0))) <= max_lon
    ]


def centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Return the geographic centroid of a list of (lat, lon) tuples."""
    if not points:
        return (0.0, 0.0)
    lats, lons = zip(*points)
    return (sum(lats) / len(lats), sum(lons) / len(lons))
