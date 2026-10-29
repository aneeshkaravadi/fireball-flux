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


def test_chelyabinsk_direction_against_popova_2013():
    """CNEOS's Chelyabinsk vector vs the video trajectory in Popova et al. (2013, Science 342, 1069), Table 1.

    Popova: entry elevation 18.3 +/- 0.4 deg, radiant azimuth 283.2 +/- 0.4 deg measured from south,
    which is 103.2 deg from north. CNEOS (lat 54.8 N, lon 61.1 E, v = (12.8, -13.3, -2.4) km/s) gives
    a path 2.4 deg shallower and a radiant 3.3 deg further north. This test pins down that the gap is
    real: rounding the CNEOS numbers to their printed precision moves the angles by about 0.1 deg (1 sigma),
    and by less than 0.3 deg either way at the extremes.
    """
    lat, lon, v = 54.8, 61.1, np.array([12.8, -13.3, -2.4])
    h = geometry.entry_angle_deg(lat, lon, *v)
    az = geometry.radiant_azimuth_deg(lat, lon, *v)
    assert h == pytest.approx(15.9, abs=0.05)
    assert az == pytest.approx(99.9, abs=0.05)
    rng = np.random.default_rng(2)
    vv = v + rng.uniform(-0.05, 0.05, (20000, 3))
    la = lat + rng.uniform(-0.05, 0.05, 20000)
    lo = lon + rng.uniform(-0.05, 0.05, 20000)
    hh = geometry.entry_angle_deg(la, lo, *vv.T)
    aa = geometry.radiant_azimuth_deg(la, lo, *vv.T)
    assert np.ptp(hh) < 0.6 and np.ptp(aa) < 0.6          # rounding can't explain a 2-3 degree gap
    assert 18.3 - 0.4 - hh.max() > 1.5                       # disagreement in elevation
    assert (283.2 - 180.0) - 0.4 - aa.max() > 2.0            # disagreement in azimuth
