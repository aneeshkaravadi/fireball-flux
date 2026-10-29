"""Strength inversion with the entry-speed correction, on SYNTHETIC bodies with known answers."""
import numpy as np
import pytest

from fireballs import airburst, atmosphere, entry


def test_ram_pressure():
    assert airburst.ram_pressure(23.3e3, 18.6e3) == pytest.approx(atmosphere.density(23.3e3) * 18.6e3**2)


def test_constant_strength_gives_slope_minus_two():
    """SYNTHETIC: bodies that all burst exactly where rho v^2 = Y give a slope of -2."""
    rng = np.random.default_rng(0)
    v = rng.uniform(12e3, 40e3, 200)
    z = entry.ram_pressure_altitude(3e6, v)
    s = airburst.density_speed_slope(z, v, n_boot=200)
    assert s["ols"] == pytest.approx(-2.0, abs=0.01)
    assert s["theil_sen"] == pytest.approx(-2.0, abs=0.01)
    # and with a strength that scatters by a factor of ~3 but doesn't depend on speed, still about -2
    y = 3e6 * np.exp(rng.normal(0, 1.0, v.size))
    s = airburst.density_speed_slope(entry.ram_pressure_altitude(y, v), v, n_boot=200)
    assert s["ols_95"][0] < -2.0 < s["ols_95"][1]


def test_inversion_recovers_strength_and_entry_speed():
    """SYNTHETIC: fly bodies with known strength and entry speed, keep only what CNEOS would report
    (burst altitude, speed at the burst, angle, energy), and get both back."""
    rng = np.random.default_rng(1)
    n = 24
    rho_i = 3000.0
    L0 = rng.uniform(0.8, 6.0, n)
    v0 = rng.uniform(13e3, 28e3, n)
    th = rng.uniform(20.0, 75.0, n)
    y = 10 ** rng.uniform(5.0, 7.0, n)
    r = entry.simulate(L0, rho_i, v0, th, y)
    energy = 0.5 * entry.mass(L0, rho_i) * v0**2
    inv = airburst.implied_strength(r.burst_alt, r.burst_speed, th, energy, rho_i)
    ok = (inv.flag == 0) & inv.speed_matched
    assert ok.sum() >= 15
    assert inv.strength[ok] == pytest.approx(y[ok], rel=0.12)
    assert inv.entry_speed[ok] == pytest.approx(v0[ok], rel=0.02)
    assert inv.diameter[ok] == pytest.approx(L0[ok], rel=0.03)
    # without the correction (CNEOS speed taken as the entry speed) the strengths come out low
    s_raw, f_raw, _ = entry.invert_strength(r.burst_alt, entry.diameter_from_energy(energy, r.burst_speed, rho_i),
                                            rho_i, r.burst_speed, th)
    both = ok & (f_raw == 0)
    assert np.median(s_raw[both] / y[both]) < 0.9


def test_single_strength_fit_finds_the_common_strength():
    """SYNTHETIC: if every body has the same strength, the one-strength fit finds it."""
    rng = np.random.default_rng(2)
    n = 20
    L0, v0, th = rng.uniform(1, 4, n), rng.uniform(14e3, 25e3, n), rng.uniform(25, 70, n)
    grid = entry.strength_grid(1e4, 1e8, 10)
    res = entry.simulate(L0[:, None], 3000.0, v0[:, None], th[:, None], grid[None, :])
    true = 22   # 1.6 MPa, on the sloped part (below ~0.1 MPa every body bursts at its weak-body plateau)
    y, rms, _ = airburst.single_strength_fit(res.burst_alt[:, true], res.burst_alt, grid)
    assert y == pytest.approx(grid[true]) and rms == 0.0
    q, pred, rms_q = airburst.ram_pressure_fit(entry.ram_pressure_altitude(2e6, v0), v0)
    assert q == pytest.approx(2e6, rel=0.03) and rms_q < 100.0
