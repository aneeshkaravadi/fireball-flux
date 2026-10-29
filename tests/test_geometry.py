"""Entry geometry from velocity vectors."""
import numpy as np
import pytest

from fireballs import geometry


def test_basis_is_orthonormal_and_right_handed():
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(-89, 89, 50), rng.uniform(-180, 180, 50)
    e, n, u = geometry.enu_basis(lat, lon)
    for a in (e, n, u):
        assert np.linalg.norm(a, axis=-1) == pytest.approx(np.ones(50))
    assert np.sum(e * n, -1) == pytest.approx(np.zeros(50), abs=1e-12)
    assert np.cross(e, n) == pytest.approx(u)


def test_hand_worked_directions():
    # at (0, 0) up is +x, east is +y, north is +z
    assert geometry.entry_angle_deg(0, 0, -10, 0, 0) == pytest.approx(90.0)        # straight down
    assert geometry.entry_angle_deg(0, 0, 0, 10, 0) == pytest.approx(0.0)          # horizontal, heading east
    assert geometry.radiant_azimuth_deg(0, 0, 0, 10, 0) == pytest.approx(270.0)    # came from the west
    assert geometry.radiant_azimuth_deg(0, 0, 0, 0, -10) == pytest.approx(0.0)     # heading south, came from north
    assert geometry.entry_angle_deg(0, 0, -1, 0, -1) == pytest.approx(45.0)
    # at the north pole, up is +z
    assert geometry.entry_angle_deg(90, 0, 3, 4, -5) == pytest.approx(45.0)


def test_isotropic_angle_distribution_by_monte_carlo():
    """SYNTHETIC: isotropic directions weighted by cos(zenith) should give CDF sin^2(h)."""
    rng = np.random.default_rng(1)
    mu = rng.uniform(size=400000)
    w = mu  # flux through a horizontal plane goes as cos(zenith)
    h = np.degrees(np.arcsin(mu))  # elevation of the direction
    for a in (15, 30, 45, 60, 75):
        frac = np.sum(w[h < a]) / np.sum(w)
        assert frac == pytest.approx(geometry.isotropic_entry_angle_cdf(a), abs=3e-3)


def test_local_solar_time():
    lst = geometry.local_solar_time_h(["2013-02-15 03:20:26"], [61.1])
    assert lst[0] == pytest.approx(3 + 20 / 60 + 26 / 3600 + 61.1 / 15)


def test_straight_line_keeps_r_cos_angle_constant():
    """The 3D trace and the closed form (R + z) cos(angle) = constant agree, and tracing to the same
    altitude changes nothing."""
    rng = np.random.default_rng(3)
    lat, lon = rng.uniform(-80, 80, 40), rng.uniform(-180, 180, 40)
    v = rng.normal(size=(40, 3)) * 15
    alt = rng.uniform(20, 45, 40)
    h0 = geometry.entry_angle_deg(lat, lon, *v.T)
    ok = h0 > 8      # descending, and steep enough that the line reaches 10 km
    for target in (alt, np.full(40, 100.0), np.full(40, 10.0)):
        la, lo, h, az = geometry.trace_to_altitude(lat, lon, alt, *v.T, target)
        expect = geometry.angle_at_altitude(h0, alt, target)
        assert h[ok] == pytest.approx(expect[ok], abs=1e-6)
    la, lo, h, az = geometry.trace_to_altitude(lat, lon, alt, *v.T, alt)
    assert la[ok] == pytest.approx(lat[ok]) and h[ok] == pytest.approx(h0[ok])
    # a 3 degree path at 30 km never gets down to the ground
    assert np.isnan(geometry.angle_at_altitude(3.0, 30.0, 0.0))


def test_chelyabinsk_direction_agrees_with_popova_2013():
    """CNEOS's Chelyabinsk vector vs the video trajectory in Popova et al. (2013, Science 342, 1069), Table 1:
    beginning height 97.1 km, entry elevation 18.3 +/- 0.4 deg and azimuth 283.2 +/- 0.4 deg from south
    (103.2 deg from north), all 2 sigma.

    The CNEOS angle (15.9 deg at the peak-brightness point, 23.3 km) and Popova's (at 97.1 km) belong
    to points about 250 km apart, and I first compared them directly, which gave a fake 2.4 deg gap.
    Following the CNEOS line back up to 97.1 km gives 18.1 deg and 102.9 deg. Both are inside Popova's
    2 sigma once the rounding of the CNEOS numbers (about 0.1 deg at 1 sigma) is added.
    """
    lat, lon, v = 54.8, 61.1, np.array([12.8, -13.3, -2.4])
    assert geometry.entry_angle_deg(lat, lon, *v) == pytest.approx(15.9, abs=0.05)
    assert geometry.radiant_azimuth_deg(lat, lon, *v) == pytest.approx(99.9, abs=0.05)
    _, _, h, az = geometry.trace_to_altitude(lat, lon, 23.3, *v, 97.1)
    tol = np.hypot(0.4, 2 * 0.1)
    assert abs(h - 18.3) < tol
    assert abs(az - (283.2 - 180.0)) < tol
    # and rounding: each printed number anywhere within half its last digit moves the angles ~0.1 deg
    rng = np.random.default_rng(2)
    vv = v + rng.uniform(-0.05, 0.05, (20000, 3))
    la = lat + rng.uniform(-0.05, 0.05, 20000)
    lo = lon + rng.uniform(-0.05, 0.05, 20000)
    al = 23.3 + rng.uniform(-0.05, 0.05, 20000)
    _, _, hh, aa = geometry.trace_to_altitude(la, lo, al, *vv.T, 97.1)
    assert np.std(hh) < 0.15 and np.std(aa) < 0.15
