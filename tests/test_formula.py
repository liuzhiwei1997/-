import pytest

from sfrt_lattice_planner.geometry import lattice_parameters_from_volume


def test_lattice_formulas_low_volume():
    p = lattice_parameters_from_volume(49.9)
    assert p.sphere_diameter_cm == pytest.approx(0.5)
    assert p.lattice_spacing_cm == pytest.approx(2.0)
    assert p.sphere_diameter_mm == pytest.approx(5.0)
    assert p.lattice_spacing_mm == pytest.approx(20.0)


def test_lattice_formulas_mid_volume():
    p = lattice_parameters_from_volume(100.0)
    assert p.sphere_diameter_cm == pytest.approx(0.5258)
    assert p.lattice_spacing_cm == pytest.approx(2.1)
    assert p.radius_mm == pytest.approx(2.629)


def test_lattice_formulas_high_volume():
    p = lattice_parameters_from_volume(1000.0)
    assert p.sphere_diameter_cm == pytest.approx(1.5)
    assert p.lattice_spacing_cm == pytest.approx(4.0)
    assert p.sphere_diameter_mm == pytest.approx(15.0)
