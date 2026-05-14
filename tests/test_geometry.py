import numpy as np
import pytest

from sfrt_lattice_planner.geometry import mask_volume_cc, sphere_volume_cc
from sfrt_lattice_planner.masks import choose_origin_center, make_placement_mask, sphere_mask
from sfrt_lattice_planner.optimize import optimize_lattice_spacing


def ellipsoid_mask(shape=(33, 33, 33), radii=(12, 10, 9), center=None):
    center = np.array(center if center is not None else [(s - 1) / 2 for s in shape], dtype=float)
    zz, yy, xx = np.indices(shape)
    return (((zz - center[0]) / radii[0]) ** 2 + ((yy - center[1]) / radii[1]) ** 2 + ((xx - center[2]) / radii[2]) ** 2) <= 1.0


def test_sphere_mask_radius_and_volume_are_reasonable():
    spacing = (1.0, 1.0, 1.0)
    radius = 8.0
    mask = sphere_mask((41, 41, 41), [20.0, 20.0, 20.0], radius, spacing)
    vol = mask_volume_cc(mask, spacing)
    assert vol == pytest.approx(sphere_volume_cc(radius), rel=0.08)
    assert mask[20, 20, 20]
    assert mask[20, 20, 28]
    assert not mask[20, 20, 29]


def test_sphere_mask_handles_anisotropic_spacing():
    spacing = (1.0, 1.0, 2.0)
    mask = sphere_mask((21, 41, 41), [20.0, 20.0, 20.0], 4.0, spacing)
    assert mask[10, 20, 20]
    assert mask[12, 20, 20]  # +4 mm in z
    assert not mask[13, 20, 20]  # +6 mm in z


def test_placement_center_uses_nearest_valid_voxel_when_centroid_invalid():
    placement = np.zeros((7, 7, 7), dtype=bool)
    placement[1, 1, 1] = True
    placement[5, 5, 5] = True
    center = choose_origin_center(placement, (1, 1, 1))
    assert tuple(center) in {(1.0, 1.0, 1.0), (5.0, 5.0, 5.0)}


def test_make_placement_mask_distance_transform_logic():
    tumor = np.zeros((25, 25, 25), dtype=bool)
    tumor[3:22, 3:22, 3:22] = True
    oar = np.zeros_like(tumor)
    oar[:, :, :5] = True
    result = make_placement_mask(tumor, [oar], (1, 1, 1), tumor_inset_mm=3, oar_clearance_mm=5)
    assert result.placement_mask[12, 12, 12]
    assert not result.placement_mask[4, 12, 12]
    assert not result.placement_mask[12, 12, 7]


def test_optimization_generates_valid_result_in_synthetic_ellipsoid():
    tumor = ellipsoid_mask()
    placement = make_placement_mask(tumor, [], (1, 1, 1), tumor_inset_mm=3, oar_clearance_mm=0).placement_mask
    center = choose_origin_center(placement, (1, 1, 1))
    result = optimize_lattice_spacing(
        placement,
        center,
        initial_lattice_spacing_mm=7.0,
        radius_mm=2.0,
        spacing_mm=(1, 1, 1),
        volume_tolerance=0.75,
        max_outer_steps=2,
        anneal_iterations=5,
        seed=1,
    )
    assert len(result.initial_centers_mm) > 0
    assert len(result.optimized_centers_mm) > 0
    assert result.optimized_total_volume_cc >= result.initial_total_volume_cc * 0.75
