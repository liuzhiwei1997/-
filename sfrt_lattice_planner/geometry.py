"""Geometry and coordinate utilities for SFRT lattice planning."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from ._compat import pairwise_distances


@dataclass(frozen=True)
class LatticeParameters:
    tumor_volume_cc: float
    sphere_diameter_cm: float
    lattice_spacing_cm: float
    sphere_diameter_mm: float
    lattice_spacing_mm: float
    radius_mm: float


def lattice_parameters_from_volume(volume_cc: float) -> LatticeParameters:
    """Return patent-formula sphere diameter and close-packing spacing."""
    volume_cc = float(volume_cc)
    if volume_cc < 50:
        d_sphere_cm = 0.5
        d_lattice_cm = 2.0
    elif volume_cc < 1000:
        d_sphere_cm = 0.001 * volume_cc + 0.4258
        d_lattice_cm = 0.002 * volume_cc + 1.9
    else:
        d_sphere_cm = 1.5
        d_lattice_cm = 4.0
    return LatticeParameters(
        tumor_volume_cc=volume_cc,
        sphere_diameter_cm=d_sphere_cm,
        lattice_spacing_cm=d_lattice_cm,
        sphere_diameter_mm=d_sphere_cm * 10.0,
        lattice_spacing_mm=d_lattice_cm * 10.0,
        radius_mm=d_sphere_cm * 5.0,
    )


def mask_volume_cc(mask: np.ndarray, spacing_mm: Sequence[float]) -> float:
    return float(np.count_nonzero(mask) * np.prod(np.asarray(spacing_mm, dtype=float)) / 1000.0)


def sphere_volume_cc(radius_mm: float) -> float:
    return float((4.0 / 3.0) * math.pi * float(radius_mm) ** 3 / 1000.0)


def min_pairwise_distance(points_mm: np.ndarray) -> float | None:
    points_mm = np.asarray(points_mm, dtype=float)
    if len(points_mm) < 2:
        return None
    return float(np.min(pairwise_distances(points_mm)))


def rotation_matrix_xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Return Rx(alpha) @ Ry(beta) @ Rz(gamma)."""
    ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sa, sb, sg = math.sin(alpha), math.sin(beta), math.sin(gamma)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=float)
    rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]], dtype=float)
    return rx @ ry @ rz


def apply_rigid_transform(
    points_mm: np.ndarray,
    center_mm: Sequence[float],
    translation_mm: Sequence[float],
    angles_rad: Sequence[float],
) -> np.ndarray:
    points = np.asarray(points_mm, dtype=float)
    center = np.asarray(center_mm, dtype=float)
    trans = np.asarray(translation_mm, dtype=float)
    rot = rotation_matrix_xyz(*angles_rad)
    return (points - center) @ rot.T + center + trans


def indices_to_physical(indices_zyx: np.ndarray, spacing_mm: Sequence[float], origin_mm: Sequence[float] | None = None) -> np.ndarray:
    """Convert z,y,x array indices to x,y,z physical coordinates for axis-aligned arrays."""
    idx = np.asarray(indices_zyx, dtype=float)
    spacing = np.asarray(spacing_mm, dtype=float)
    origin = np.zeros(3, dtype=float) if origin_mm is None else np.asarray(origin_mm, dtype=float)
    return origin + idx[:, [2, 1, 0]] * spacing


def physical_to_indices(points_xyz_mm: np.ndarray, spacing_mm: Sequence[float], origin_mm: Sequence[float] | None = None) -> np.ndarray:
    """Convert x,y,z physical coordinates to nearest z,y,x array indices for axis-aligned arrays."""
    pts = np.asarray(points_xyz_mm, dtype=float)
    spacing = np.asarray(spacing_mm, dtype=float)
    origin = np.zeros(3, dtype=float) if origin_mm is None else np.asarray(origin_mm, dtype=float)
    ijk_xyz = np.rint((pts - origin) / spacing).astype(int)
    return ijk_xyz[:, [2, 1, 0]]


def bounding_box_physical(mask: np.ndarray, spacing_mm: Sequence[float], origin_mm: Sequence[float] | None = None, margin_mm: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot compute a bounding box for an empty mask.")
    pts = indices_to_physical(coords, spacing_mm, origin_mm)
    margin = float(margin_mm)
    return pts.min(axis=0) - margin, pts.max(axis=0) + margin
