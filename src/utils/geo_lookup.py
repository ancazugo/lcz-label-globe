"""Nearest-centroid city assignment against the GUPPD bounding boxes.

Vendored from eo_fm. Only the two functions the builder calls are kept; the
bbox-containment variant and the hard-coded country/continent tables are not
used here and would pull in geopandas and loguru.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_city_centers(bounds_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (city_names, centers) from the GUPPD bounds CSV.

    centers is an (N, 2) array of (lon, lat) bounding-box centroids.
    """
    df = pd.read_csv(bounds_csv)
    names = df["JRC_NAME_MAIN"].to_numpy()
    centers = np.column_stack([
        (df["minx"] + df["maxx"]) / 2.0,
        (df["miny"] + df["maxy"]) / 2.0,
    ])
    return names, centers


def assign_city(coords: np.ndarray, bounds_csv: Path) -> np.ndarray:
    """Assign each (lon, lat) coordinate to its nearest GUPPD city centroid.

    coords: (N, 2) array of (lon, lat). Returns an (N,) array of city-name
    strings. Always matches (nearest centroid), mirroring the notebook.
    """
    from sklearn.neighbors import NearestNeighbors

    names, centers = load_city_centers(bounds_csv)
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=-1)
    nn.fit(centers)
    _, idx = nn.kneighbors(coords)
    return names[idx.flatten()]
