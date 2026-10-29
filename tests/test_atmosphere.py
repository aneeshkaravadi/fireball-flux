"""US Standard Atmosphere 1976 density against the official table."""
from pathlib import Path

import numpy as np
import pytest

from fireballs import atmosphere

TABLE = Path(__file__).parent / "fixtures" / "us76_table1_density.csv"


def test_density_matches_us76_table():
    z, rho = np.loadtxt(TABLE, delimiter=",", comments="#", skiprows=4, unpack=True)
    # the table prints 5 significant figures, so allow a little more than half a unit in the last place
    assert atmosphere.density(z) == pytest.approx(rho, rel=6e-5)


def test_sea_level_and_tropopause():
    t, p, rho = atmosphere.us76(0.0)
    assert (t, p) == (pytest.approx(288.15), pytest.approx(101325.0))
    t, p, _ = atmosphere.us76(atmosphere.R_EARTH * 11000.0 / (atmosphere.R_EARTH - 11000.0))  # geopotential 11 km
    assert t == pytest.approx(216.65, abs=1e-6)
    assert p == pytest.approx(22632, rel=1e-4)


def test_density_is_smooth_and_decreasing_including_above_86_km():
    z = np.linspace(0, 120e3, 4801)
    rho = atmosphere.density(z)
    assert np.all(np.diff(rho) < 0)
    jump = atmosphere.density(atmosphere.Z_TOP + 1.0) / atmosphere.density(atmosphere.Z_TOP - 1.0)
    assert jump == pytest.approx(np.exp(-2.0 / atmosphere.H86), rel=1e-3)


def test_inverse_and_exponential():
    z = np.array([5e3, 23.3e3, 40e3, 75e3, 95e3])
    assert atmosphere.altitude_for_density(atmosphere.density(z)) == pytest.approx(z, abs=5.0)
    assert atmosphere.exponential(8000.0) == pytest.approx(np.exp(-1.0))
    assert atmosphere.scale_height(np.array([20e3]))[0] == pytest.approx(
        atmosphere.R_STAR * 216.65 / (atmosphere.M0 * atmosphere.G0) * ((atmosphere.R_EARTH + 20e3) / atmosphere.R_EARTH) ** 2,
        rel=2e-3)  # isothermal layer: H = R T / (M g)
