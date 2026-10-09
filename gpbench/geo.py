"""Small local-geometry helpers. Coordinates are handled in a local metric plane (meters)."""
from __future__ import annotations

import numpy as np

R_EARTH = 6371008.8  # mean Earth radius, meters


def to_xy(lat, lon, lat0, lon0):
    """Equirectangular projection to meters around (lat0, lon0); fine for city-scale areas."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    x = np.radians(lon - lon0) * R_EARTH * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R_EARTH
    return np.column_stack([x, y])


def to_latlon(xy, lat0, lon0):
    xy = np.asarray(xy, float)
    lat = lat0 + np.degrees(xy[:, 1] / R_EARTH)
    lon = lon0 + np.degrees(xy[:, 0] / (R_EARTH * np.cos(np.radians(lat0))))
    return lat, lon


def dist(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))
