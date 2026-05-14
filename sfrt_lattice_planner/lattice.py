"""HCP / close-packing lattice generation."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .geometry import bounding_box_physical
from .masks import filter_centers_by_distance


def hcp_basis(spacing_mm: float) -> np.ndarray:
    d = float(spacing_mm)
    return np.array(
        [
            [d, 0.0, 0.0],
            [0.5 * d, math.sqrt(3.0) / 2.0 * d, 0.0],
            [0.5 * d, math.sqrt(3.0) / 6.0 * d, math.sqrt(2.0 / 3.0) * d],
        ],
        dtype=float,
    )


def generate_hcp_candidates(
    origin_center_mm: Sequence[float],
    spacing_mm_lattice: float,
    bbox_min_mm: Sequence[float],
    bbox_max_mm: Sequence[float],
) -> np.ndarray:
    """Generate HCP candidate centers in a physical x,y,z bounding box."""
    origin = np.asarray(origin_center_mm, dtype=float)
    bmin = np.asarray(bbox_min_mm, dtype=float)
    bmax = np.asarray(bbox_max_mm, dtype=float)
    basis = hcp_basis(spacing_mm_lattice)
    extent = float(np.linalg.norm(bmax - bmin)) + 2.0 * spacing_mm_lattice
    n = int(math.ceil(extent / max(float(spacing_mm_lattice), 1e-6))) + 3
    pts: list[np.ndarray] = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            for k in range(-n, n + 1):
                p = origin + i * basis[0] + j * basis[1] + k * basis[2]
                if np.all(p >= bmin) and np.all(p <= bmax):
                    pts.append(p)
    if not pts:
        return np.empty((0, 3), dtype=float)
    arr = np.vstack(pts)
    order = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))
    return arr[order]


def generate_valid_lattice_centers(
    placement_mask: np.ndarray,
    origin_center_mm: Sequence[float],
    lattice_spacing_mm: float,
    sphere_radius_mm: float,
    spacing_mm: Sequence[float],
    image_origin_mm: Sequence[float] | None = None,
    bbox_margin_mm: float | None = None,
) -> np.ndarray:
    margin = float(lattice_spacing_mm if bbox_margin_mm is None else bbox_margin_mm)
    bmin, bmax = bounding_box_physical(placement_mask, spacing_mm, image_origin_mm, margin_mm=margin)
    candidates = generate_hcp_candidates(origin_center_mm, lattice_spacing_mm, bmin, bmax)
    return filter_centers_by_distance(candidates, placement_mask, sphere_radius_mm, spacing_mm, image_origin_mm)
