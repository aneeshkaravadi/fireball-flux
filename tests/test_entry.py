"""The entry model against closed forms (Collins et al. 2005), an independent ODE solve, and itself."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from fireballs import atmosphere, entry

EXPO = atmosphere.exponential  # rho0 = 1 kg/m^3, H = 8 km, as in Collins et al.


def test_intact_body_slows_down_as_collins_eq8():
    """No breakup: v(z) = v0 exp(-3 rho(z) C_D H / (4 rho_i L0 sin(theta))) in an exponential atmosphere."""
    for L0, th in ((1.0, 20.0), (5.0, 45.0), (30.0, 80.0)):
        r = entry.simulate(L0, 3000.0, 20e3, th, 1e12, rho=EXPO, keep_profile=True)
        vc = entry.collins_speed(r.profile["z"], 20e3, L0, 3000.0, th)
        # the closed form starts at infinite altitude and the model at 100 km, so compare v(z) / v(100 km)
        assert r.profile["v"] / 20e3 == pytest.approx(vc / vc[0], rel=1e-5)
        assert np.isnan(r.breakup_alt)


def test_breakup_without_deceleration_is_where_ram_pressure_equals_strength():
    for s in (1e4, 3e5, 1e7):
        r = entry.simulate(5.0, 3000.0, 20e3, 45.0, s, rho=EXPO, decelerate=False)
        assert r.breakup_alt == pytest.approx(8000.0 * np.log(20e3**2 / s), abs=1.0)
    # and in US76
    r = entry.simulate(5.0, 3000.0, 18e3, 45.0, 2e6, decelerate=False)
    assert r.breakup_alt == pytest.approx(entry.ram_pressure_altitude(2e6, 18e3), abs=5.0)


@pytest.mark.parametrize("L0,s,th", [(5, 1e6, 45), (20, 2e5, 18), (2, 5e6, 30), (50, 1e7, 60), (1, 1e5, 15)])
def test_breakup_altitude_with_drag_matches_collins_eq11(L0, s, th):
    """Collins et al. call eq. 11 an excellent approximation; here it lands within 50 m."""
    r = entry.simulate(L0, 3000.0, 19e3, th, s, rho=EXPO)
    assert r.breakup_alt == pytest.approx(entry.collins_breakup_altitude(s, 19e3, L0, 3000.0, th), abs=50.0)


def exact_pancake_altitude(L0, rho_i, th, z_star, fp=7.0, cd=2.0, h=8000.0):
    """Solve Collins eq. 14, L L'' = C_D rho(z) / (rho_i sin^2 theta), with scipy from L(z*) = L0, L'(z*) = 0."""
    s = np.sin(np.radians(th))
    c = cd / (rho_i * s * s)

    def f(depth, y):
        return [y[1], c * np.exp(-(z_star - depth) / h) / y[0]]

    def reach(depth, y):
        return y[0] - fp * L0
    reach.terminal = True
    sol = solve_ivp(f, [0, z_star], [L0, 0.0], events=reach, rtol=1e-11, atol=1e-12, method="DOP853")
    return z_star - sol.t_events[0][0]


CASES = np.array([(L0, th, s) for L0 in (1.0, 5.0, 20.0) for th in (15.0, 45.0, 75.0) for s in (1e5, 1e6, 1e7)])


@pytest.fixture(scope="module")
def fixed_speed_runs():
    """All 27 cases in one vectorized run, speed held fixed (Collins's assumption for the pancake)."""
    L0, th, s = CASES.T
    r = entry.simulate(L0, 3000.0, 19e3, th, s, rho=EXPO, decelerate=False, dz=10.0)
    exact = np.array([exact_pancake_altitude(a, 3000.0, b, z) for a, b, z in zip(L0, th, r.breakup_alt)])
    return r, exact


def test_pancake_spread_matches_an_independent_ode_solve(fixed_speed_runs):
    """My RK4 matches scipy to within two altitude steps (the spread starts at the step after breakup,
    and reaching the cap is recorded at the end of a step)."""
    r, exact = fixed_speed_runs
    assert np.all((r.pancake_alt - exact >= -20.0) & (r.pancake_alt - exact <= 1.0))


def test_collins_closed_form_pancake_vs_exact_solution(fixed_speed_runs):
    """Collins's eq. 15-16 closed form is an approximation to eq. 14. Over these 27 cases it puts the
    airburst between 2.3 km lower and 0.6 km higher than the exact solution. Documenting, not tuning."""
    r, exact = fixed_speed_runs
    L0, th, _ = CASES.T
    diffs = entry.collins_airburst_altitude(r.breakup_alt, L0, 3000.0, th) - exact
    assert diffs.min() > -2400.0 and diffs.max() < 600.0
    assert np.mean(diffs < 0) > 0.9


def test_burst_altitude_decreases_as_strength_increases():
    strengths = entry.strength_grid(1e3, 1e8, 10)
    for L0, v, th in ((0.5, 15e3, 30.0), (3.0, 18e3, 45.0), (15.0, 25e3, 70.0)):
        r = entry.simulate(L0, 3000.0, v, th, strengths)
        broke = np.isfinite(r.breakup_alt)
        assert np.all(np.diff(r.breakup_alt[broke]) < 0)          # stronger bodies break lower
        assert np.all(np.diff(r.burst_alt) <= 0)                   # and never burst higher
        assert r.burst_alt[0] - r.burst_alt[-1] > 5e3
        # small strong bodies slow down before the ram pressure gets high enough to break them
        assert not broke[-1] or L0 > 1.0


def test_step_size_is_converged():
    s = entry.strength_grid(1e4, 1e8, 4)
    a = entry.simulate(4.0, 3000.0, 18e3, 40.0, s, dz=25.0)
    b = entry.simulate(4.0, 3000.0, 18e3, 40.0, s, dz=5.0)
    np.testing.assert_allclose(a.breakup_alt, b.breakup_alt, atol=5.0)  # nan where it never breaks, in both
    np.testing.assert_allclose(a.burst_alt, b.burst_alt, atol=50.0)


def test_energy_bookkeeping():
    """The power the model reports, integrated over time, equals the kinetic energy it loses."""
    L0, rho_i, v0, th = 3.0, 3000.0, 18e3, 35.0
    r = entry.simulate(L0, rho_i, v0, th, 1e6, keep_profile=True, dz=5.0)
    z, v, p = r.profile["z"], r.profile["v"], r.profile["power"]
    t = np.concatenate([[0.0], np.cumsum(np.diff(-z) / (0.5 * (v[1:] + v[:-1]) * np.sin(np.radians(th))))])
    lost = 0.5 * entry.mass(L0, rho_i) * (v0**2 - v[-1] ** 2)
    assert np.trapezoid(p, t) == pytest.approx(lost, rel=2e-3)


def test_diameter_from_energy():
    d = entry.diameter_from_energy(1e14, 2e4, 3000.0)
    assert 0.5 * entry.mass(d, 3000.0) * 2e4**2 == pytest.approx(1e14)


def test_inversion_recovers_a_known_strength():
    """SYNTHETIC: fly bodies with known strengths, then recover them from the burst altitude alone."""
    rng = np.random.default_rng(0)
    n = 30
    L0 = rng.uniform(0.5, 8.0, n)
    v = rng.uniform(12e3, 30e3, n)
    th = rng.uniform(15.0, 80.0, n)
    s_true = 10 ** rng.uniform(4.5, 7.5, n)
    zb = entry.simulate(L0, 3000.0, v, th, s_true).burst_alt
    s_fit, flag, _ = entry.invert_strength(zb, L0, 3000.0, v, th)
    ok = flag == 0
    assert ok.sum() >= 15
    # the burst altitude is only resolved to one 25 m step, which is a few percent in strength
    assert s_fit[ok] == pytest.approx(s_true[ok], rel=0.1)
    # where the burst altitude doesn't pin the strength down, the limits are still right
    assert np.all(s_true[flag == -1] <= s_fit[flag == -1] * 1.1)
    assert np.all(s_true[flag == 1] >= s_fit[flag == 1] / 1.1)


def test_small_shallow_weak_bodies_stay_finite():
    """Regression: a 0.5-0.75 m body at 5-7 degrees used to overflow once the crushed swarm got low,
    and the overflowed power showed up as a fake burst a few km above the ground."""
    strengths = entry.strength_grid(1e3, 1e5, 10)
    for L0, th, v in ((0.75, 6.7, 40.4e3), (0.5, 5.1, 71.1e3)):
        r = entry.simulate(L0, 3000.0, v, th, strengths, keep_profile=True)
        assert np.all(np.isfinite(r.peak_power)) and np.all(np.isfinite(r.profile["v"]))
        assert np.all(r.profile["v"] >= 0)    # the swarm can stop dead (no gravity in the model)
        assert np.all(r.burst_alt > 40e3)   # weak and fast: these peak high, not near the ground
