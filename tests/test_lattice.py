import numpy as np
import pytest
from sfrt_lattice_planner.lattice import generate_hcp_candidates, hcp_basis


def test_hcp_basis_nearest_neighbor_distance():
    d = 12.5
    basis = hcp_basis(d)
    pts = []
    origin = np.zeros(3)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                pts.append(origin + i * basis[0] + j * basis[1] + k * basis[2])
    pts = np.vstack(pts)
    distances = np.linalg.norm(pts, axis=1)
    distances = np.sort(distances[distances > 1e-9])
    assert distances[0] == pytest.approx(d)


def test_hcp_candidates_respect_bounding_box():
    pts = generate_hcp_candidates([0, 0, 0], 10, [-15, -15, -15], [15, 15, 15])
    assert len(pts) > 0
    assert np.all(pts >= -15)
    assert np.all(pts <= 15)
