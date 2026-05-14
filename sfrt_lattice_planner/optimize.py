"""Simulated annealing optimizer for rigidly transformed HCP lattices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

import numpy as np

from .geometry import apply_rigid_transform, sphere_volume_cc
from .lattice import generate_hcp_candidates
from .masks import filter_centers_by_distance
from .geometry import bounding_box_physical


@dataclass(frozen=True)
class OptimizationResult:
    initial_centers_mm: np.ndarray
    optimized_centers_mm: np.ndarray
    initial_spacing_mm: float
    optimized_spacing_mm: float
    initial_total_volume_cc: float
    optimized_total_volume_cc: float
    accepted_steps: int


def _score(centers_mm: np.ndarray, radius_mm: float) -> float:
    return len(centers_mm) * sphere_volume_cc(radius_mm)


def anneal_for_spacing(
    candidates_mm: np.ndarray,
    placement_mask: np.ndarray,
    radius_mm: float,
    spacing_mm: Sequence[float],
    image_origin_mm: Sequence[float] | None,
    rotation_center_mm: Sequence[float],
    lattice_spacing_mm: float,
    iterations: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best = filter_centers_by_distance(candidates_mm, placement_mask, radius_mm, spacing_mm, image_origin_mm)
    best_score = _score(best, radius_mm)
    current_trans = np.zeros(3)
    current_angles = np.zeros(3)
    current_score = best_score
    center = np.asarray(rotation_center_mm, dtype=float)
    temp0 = max(best_score, sphere_volume_cc(radius_mm), 1e-9)
    for it in range(max(0, int(iterations))):
        frac = 1.0 - it / max(1, int(iterations))
        temp = temp0 * max(frac, 1e-3)
        trans = rng.uniform(-lattice_spacing_mm, lattice_spacing_mm, size=3)
        angles = rng.uniform(-math.pi, math.pi, size=3)
        # Mix broad global proposals with local refinements around the current state.
        if rng.random() < 0.5:
            trans = current_trans + rng.normal(0.0, lattice_spacing_mm * 0.15 * max(frac, 0.05), size=3)
            trans = np.clip(trans, -lattice_spacing_mm, lattice_spacing_mm)
            angles = current_angles + rng.normal(0.0, math.pi * 0.15 * max(frac, 0.05), size=3)
            angles = (angles + math.pi) % (2.0 * math.pi) - math.pi
        transformed = apply_rigid_transform(candidates_mm, center, trans, angles)
        valid = filter_centers_by_distance(transformed, placement_mask, radius_mm, spacing_mm, image_origin_mm)
        sc = _score(valid, radius_mm)
        if sc >= current_score or rng.random() < math.exp((sc - current_score) / temp):
            current_score = sc
            current_trans = trans
            current_angles = angles
        if sc > best_score:
            best_score = sc
            best = valid
    return best


def optimize_lattice_spacing(
    placement_mask: np.ndarray,
    origin_center_mm: Sequence[float],
    initial_lattice_spacing_mm: float,
    radius_mm: float,
    spacing_mm: Sequence[float],
    image_origin_mm: Sequence[float] | None = None,
    volume_tolerance: float = 0.98,
    max_outer_steps: int = 20,
    anneal_iterations: int = 1000,
    seed: int | None = None,
) -> OptimizationResult:
    bmin, bmax = bounding_box_physical(placement_mask, spacing_mm, image_origin_mm, margin_mm=initial_lattice_spacing_mm + max_outer_steps)
    initial_candidates = generate_hcp_candidates(origin_center_mm, initial_lattice_spacing_mm, bmin, bmax)
    initial = filter_centers_by_distance(initial_candidates, placement_mask, radius_mm, spacing_mm, image_origin_mm)
    v0 = _score(initial, radius_mm)
    best_centers = initial
    best_spacing = float(initial_lattice_spacing_mm)
    accepted = 0
    rng = np.random.default_rng(seed)
    for n in range(1, int(max_outer_steps) + 1):
        d_n = float(initial_lattice_spacing_mm) + n * 1.0
        bmin, bmax = bounding_box_physical(placement_mask, spacing_mm, image_origin_mm, margin_mm=d_n)
        candidates = generate_hcp_candidates(origin_center_mm, d_n, bmin, bmax)
        optimized = anneal_for_spacing(
            candidates, placement_mask, radius_mm, spacing_mm, image_origin_mm, origin_center_mm, d_n,
            iterations=anneal_iterations, seed=int(rng.integers(0, 2**32 - 1)),
        )
        vol = _score(optimized, radius_mm)
        if vol >= v0 * float(volume_tolerance):
            best_centers = optimized
            best_spacing = d_n
            accepted = n
        else:
            break
    return OptimizationResult(
        initial_centers_mm=initial,
        optimized_centers_mm=best_centers,
        initial_spacing_mm=float(initial_lattice_spacing_mm),
        optimized_spacing_mm=best_spacing,
        initial_total_volume_cc=v0,
        optimized_total_volume_cc=_score(best_centers, radius_mm),
        accepted_steps=accepted,
    )
