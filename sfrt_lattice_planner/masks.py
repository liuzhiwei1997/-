"""Mask generation, placement-zone, and spherical target utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ._compat import distance_transform_edt

from .geometry import indices_to_physical, mask_volume_cc, physical_to_indices


@dataclass(frozen=True)
class PlacementResult:
    placement_mask: np.ndarray
    tumor_inner: np.ndarray
    oar_safe: np.ndarray
    oar_union: np.ndarray


def make_placement_mask(
    tumor_mask: np.ndarray,
    oar_masks: Sequence[np.ndarray] | None,
    spacing_mm: Sequence[float],
    tumor_inset_mm: float = 20.0,
    oar_clearance_mm: float = 5.0,
) -> PlacementResult:
    tumor = np.asarray(tumor_mask, dtype=bool)
    if oar_masks:
        oar_union = np.logical_or.reduce([np.asarray(m, dtype=bool) for m in oar_masks])
    else:
        oar_union = np.zeros_like(tumor, dtype=bool)
    sampling = tuple(np.asarray(spacing_mm, dtype=float)[[2, 1, 0]])
    tumor_inner = distance_transform_edt(tumor, sampling=sampling) >= float(tumor_inset_mm)
    oar_safe = distance_transform_edt(~oar_union, sampling=sampling) >= float(oar_clearance_mm)
    return PlacementResult(tumor_inner & oar_safe, tumor_inner, oar_safe, oar_union)


def choose_origin_center(
    placement_mask: np.ndarray,
    spacing_mm: Sequence[float],
    origin_mm: Sequence[float] | None = None,
    custom_center_mm: Sequence[float] | None = None,
) -> np.ndarray:
    """Choose first lattice center in mm, preferring the placement-mask centroid."""
    placement = np.asarray(placement_mask, dtype=bool)
    if not placement.any():
        raise ValueError("placement_mask is empty; relax inset/clearance or inspect input ROIs.")
    if custom_center_mm is not None:
        pt = np.asarray(custom_center_mm, dtype=float)
        idx = physical_to_indices(pt[None, :], spacing_mm, origin_mm)[0]
        if np.any(idx < 0) or np.any(idx >= placement.shape) or not placement[tuple(idx)]:
            raise ValueError("Custom center is outside placement_mask.")
        return pt

    coords = np.argwhere(placement)
    centroid_idx = coords.mean(axis=0)
    centroid_mm = indices_to_physical(centroid_idx[None, :], spacing_mm, origin_mm)[0]
    nearest_idx = np.rint(centroid_idx).astype(int)
    if np.all(nearest_idx >= 0) and np.all(nearest_idx < placement.shape) and placement[tuple(nearest_idx)]:
        return indices_to_physical(nearest_idx[None, :], spacing_mm, origin_mm)[0]
    valid_pts = indices_to_physical(coords, spacing_mm, origin_mm)
    return valid_pts[np.argmin(np.linalg.norm(valid_pts - centroid_mm, axis=1))]


def filter_centers_by_distance(
    centers_mm: np.ndarray,
    placement_mask: np.ndarray,
    radius_mm: float,
    spacing_mm: Sequence[float],
    origin_mm: Sequence[float] | None = None,
) -> np.ndarray:
    if len(centers_mm) == 0:
        return np.empty((0, 3), dtype=float)
    sampling = tuple(np.asarray(spacing_mm, dtype=float)[[2, 1, 0]])
    dist = distance_transform_edt(np.asarray(placement_mask, dtype=bool), sampling=sampling)
    idxs = physical_to_indices(centers_mm, spacing_mm, origin_mm)
    valid = np.all((idxs >= 0) & (idxs < np.asarray(placement_mask.shape)), axis=1)
    ok = np.zeros(len(centers_mm), dtype=bool)
    inside = idxs[valid]
    ok[np.where(valid)[0]] = dist[inside[:, 0], inside[:, 1], inside[:, 2]] >= float(radius_mm)
    return np.asarray(centers_mm, dtype=float)[ok]


def sphere_mask(shape: Sequence[int], center_mm: Sequence[float], radius_mm: float, spacing_mm: Sequence[float], origin_mm: Sequence[float] | None = None) -> np.ndarray:
    """Create a spherical mask; array is z,y,x and physical coordinates are x,y,z."""
    shape = tuple(int(s) for s in shape)
    spacing = np.asarray(spacing_mm, dtype=float)
    origin = np.zeros(3, dtype=float) if origin_mm is None else np.asarray(origin_mm, dtype=float)
    center = np.asarray(center_mm, dtype=float)
    zz, yy, xx = np.indices(shape)
    x = origin[0] + xx * spacing[0]
    y = origin[1] + yy * spacing[1]
    z = origin[2] + zz * spacing[2]
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 <= float(radius_mm) ** 2


def sphere_masks(shape: Sequence[int], centers_mm: np.ndarray, radius_mm: float, spacing_mm: Sequence[float], origin_mm: Sequence[float] | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    individual = [sphere_mask(shape, c, radius_mm, spacing_mm, origin_mm) for c in np.asarray(centers_mm, dtype=float)]
    union = np.logical_or.reduce(individual) if individual else np.zeros(shape, dtype=bool)
    return union, individual
