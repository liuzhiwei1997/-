"""Compatibility fallbacks for optional scientific dependencies in constrained CI."""
from __future__ import annotations

import importlib
import importlib.util

import numpy as np

if importlib.util.find_spec("scipy") is not None and importlib.util.find_spec("scipy.ndimage") is not None:  # pragma: no cover - exercised when SciPy is installed
    distance_transform_edt = importlib.import_module("scipy.ndimage").distance_transform_edt
else:  # pragma: no cover - fallback is for minimal environments
    def distance_transform_edt(input, sampling=None):
        """Small exact Euclidean distance-transform fallback.

        This is intentionally simple and slow; production use should install SciPy.
        Distances are computed for true voxels to the nearest false voxel.
        """
        arr = np.asarray(input, dtype=bool)
        sampling_arr = np.ones(arr.ndim, dtype=float) if sampling is None else np.asarray(sampling, dtype=float)
        false_coords = np.argwhere(~arr)
        result = np.zeros(arr.shape, dtype=float)
        true_coords = np.argwhere(arr)
        if len(false_coords) == 0:
            result[arr] = np.inf
            return result
        scaled_false = false_coords * sampling_arr
        for coord in true_coords:
            diff = scaled_false - coord * sampling_arr
            result[tuple(coord)] = float(np.sqrt(np.min(np.sum(diff * diff, axis=1))))
        return result


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.empty(0, dtype=float)
    out = []
    for i in range(len(pts) - 1):
        diff = pts[i + 1 :] - pts[i]
        out.extend(np.sqrt(np.sum(diff * diff, axis=1)))
    return np.asarray(out, dtype=float)
