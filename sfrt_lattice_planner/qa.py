"""QA report metrics for SFRT lattice plans."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from ._compat import distance_transform_edt

from .geometry import mask_volume_cc, min_pairwise_distance
from .masks import physical_to_indices


def _dist_at_points(mask: np.ndarray, centers_mm: np.ndarray, spacing_mm: Sequence[float], origin_mm: Sequence[float] | None) -> float | None:
    if len(centers_mm) == 0:
        return None
    sampling = tuple(np.asarray(spacing_mm, dtype=float)[[2, 1, 0]])
    dist = distance_transform_edt(mask, sampling=sampling)
    idx = physical_to_indices(centers_mm, spacing_mm, origin_mm)
    valid = np.all((idx >= 0) & (idx < np.asarray(mask.shape)), axis=1)
    if not np.any(valid):
        return None
    vals = dist[idx[valid, 0], idx[valid, 1], idx[valid, 2]]
    return float(vals.min()) if len(vals) else None


def build_qa_report(
    tumor_mask: np.ndarray,
    placement_mask: np.ndarray,
    oar_union_mask: np.ndarray,
    initial_centers_mm: np.ndarray,
    optimized_centers_mm: np.ndarray,
    spacing_mm: Sequence[float],
    sphere_diameter_mm: float,
    initial_lattice_spacing_mm: float,
    optimized_lattice_spacing_mm: float,
    initial_total_sphere_volume_cc: float,
    optimized_total_sphere_volume_cc: float,
    origin_mm: Sequence[float] | None = None,
    parameters: dict | None = None,
) -> dict:
    return {
        "tumor_volume_cc": mask_volume_cc(tumor_mask, spacing_mm),
        "placement_volume_cc": mask_volume_cc(placement_mask, spacing_mm),
        "sphere_diameter_mm": float(sphere_diameter_mm),
        "initial_lattice_spacing_mm": float(initial_lattice_spacing_mm),
        "optimized_lattice_spacing_mm": float(optimized_lattice_spacing_mm),
        "initial_number_of_spheres": int(len(initial_centers_mm)),
        "optimized_number_of_spheres": int(len(optimized_centers_mm)),
        "initial_total_sphere_volume_cc": float(initial_total_sphere_volume_cc),
        "optimized_total_sphere_volume_cc": float(optimized_total_sphere_volume_cc),
        "volume_retention_ratio": (float(optimized_total_sphere_volume_cc) / float(initial_total_sphere_volume_cc)) if initial_total_sphere_volume_cc > 0 else None,
        "min_center_to_center_distance_mm": min_pairwise_distance(optimized_centers_mm),
        "min_distance_to_tumor_boundary_mm": _dist_at_points(tumor_mask, optimized_centers_mm, spacing_mm, origin_mm),
        "min_distance_to_oar_mm": None if not np.asarray(oar_union_mask, dtype=bool).any() else _dist_at_points(~np.asarray(oar_union_mask, dtype=bool), optimized_centers_mm, spacing_mm, origin_mm),
        "parameters_used": parameters or {},
    }
