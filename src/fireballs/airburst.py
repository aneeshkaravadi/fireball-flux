"""What the CNEOS altitudes and speeds say about how strong the incoming bodies were.

Two levels of "simple physics":

1. Ram pressure. A body breaks when rho_air v^2 reaches its strength, so the ram pressure at the
   altitude of peak brightness is a rough strength. No size, no angle, no deceleration.
2. The pancake entry model in `entry.py`. Size comes from the impact energy and an assumed bulk
   density, the angle from the CNEOS velocity vector, and the strength is whatever value makes the
   model's peak energy deposition land at the reported altitude.

CNEOS gives the speed at peak brightness, not at the top of the atmosphere, and in the model a body
has already lost some speed by then. So `implied_strength` also solves for the entry speed: it
repeats the inversion, scaling the entry speed each time until the model's speed at the burst
matches the reported one. The mass follows from the impact energy at that entry speed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import atmosphere, entry


def ram_pressure(alt_m, speed_m_s):
    """rho_air(z) v^2 in Pa, US76 density."""
    return atmosphere.density(np.asarray(alt_m, dtype=float)) * np.asarray(speed_m_s, dtype=float) ** 2


@dataclass
class Inversion:
    strength: np.ndarray      # Pa
    flag: np.ndarray          # 0 resolved, -1 upper limit, +1 lower limit (see entry.invert_strength)
    entry_speed: np.ndarray   # m/s at 100 km
    diameter: np.ndarray      # m
    burst_speed: np.ndarray   # m/s, model speed at the burst for the implied strength
    strengths: np.ndarray     # the grid used
    grid_burst_alt: np.ndarray  # m, model burst altitude for every body and every grid strength
    speed_matched: np.ndarray   # True where the model's burst speed is within 2% of the reported speed
    crossings: np.ndarray       # times the burst-altitude curve crosses the observed altitude (2+: not unique)
    rise: np.ndarray            # m, how far the burst-altitude curve climbs back up with strength

    @property
    def ambiguous(self) -> np.ndarray:
        return self.crossings > 1


def _interp_rows(x_grid, y_rows, x):
    return np.array([np.interp(xi, x_grid, row) for xi, row in zip(x, y_rows)])


def implied_strength(alt_m, speed_obs, angle_deg, energy_J, density=3000.0, *, iterations: int = 6, relax: float = 0.7,
                     strengths=None, **kw) -> Inversion:
    """Strength (and entry speed) that put the model's peak energy deposition at `alt_m`.

    alt_m and speed_obs are the reported altitude (m) and speed (m/s) at peak brightness, angle_deg the
    entry angle below horizontal, energy_J the total impact (kinetic) energy, density the assumed bulk
    density in kg/m^3. Extra keywords go to `entry.simulate` (fp, dz, rho, ...).

    The entry-speed update is damped (`relax` < 1) because a small body that bursts low can need a
    large correction, and the undamped update overshoots. For a few bodies no entry speed makes the
    model match both the altitude and the speed; `speed_matched` marks the ones that converged.
    """
    z = np.asarray(alt_m, dtype=float).ravel()
    v_obs = np.asarray(speed_obs, dtype=float).ravel()
    th = np.broadcast_to(np.asarray(angle_deg, dtype=float), z.shape).ravel()
    e = np.broadcast_to(np.asarray(energy_J, dtype=float), z.shape).ravel()
    rho_i = np.broadcast_to(np.asarray(density, dtype=float), z.shape).ravel()
    strengths = entry.strength_grid() if strengths is None else np.asarray(strengths, dtype=float)
    logs = np.log10(strengths)

    v_entry = v_obs.copy()
    for _ in range(max(1, iterations)):
        L0 = entry.diameter_from_energy(e, v_entry, rho_i)
        s, flag, res = entry.invert_strength(z, L0, rho_i, v_entry, th, strengths, **kw)
        vb = _interp_rows(logs, res.burst_speed, np.log10(s))
        v_entry = v_entry * (v_obs / vb) ** relax
    # one last pass so the reported strength goes with the final entry speed
    L0 = entry.diameter_from_energy(e, v_entry, rho_i)
    s, flag, res = entry.invert_strength(z, L0, rho_i, v_entry, th, strengths, **kw)
    vb = _interp_rows(logs, res.burst_speed, np.log10(s))
    matched = np.abs(vb / v_obs - 1.0) < 0.02
    tol = kw.get("tol", 50.0)
    return Inversion(s, flag, v_entry, L0, vb, strengths, res.burst_alt, matched,
                     entry.burst_curve_crossings(z, res.burst_alt, tol), entry.burst_curve_rise(res.burst_alt))


def single_strength_fit(alt_m, grid_burst_alt, strengths):
    """Best one-strength-for-everyone model: the grid strength whose burst altitudes have the smallest
    RMS difference from the observed ones. Returns (strength, rms_m, rms for every grid strength)."""
    z = np.asarray(alt_m, dtype=float).ravel()
    rms = np.sqrt(np.mean((np.asarray(grid_burst_alt) - z[:, None]) ** 2, axis=0))
    i = int(np.argmin(rms))
    return float(strengths[i]), float(rms[i]), rms


def ram_pressure_fit(alt_m, speed_m_s):
    """Best one-strength ram-pressure model: the Q for which the altitudes where rho v^2 = Q are closest
    (RMS) to the observed ones. Returns (Q, predicted altitudes, rms)."""
    z = np.asarray(alt_m, dtype=float).ravel()
    v = np.asarray(speed_m_s, dtype=float).ravel()
    qs = np.logspace(4, 8.5, 451)
    pred = atmosphere.altitude_for_density(qs[None, :] / v[:, None] ** 2)
    rms = np.sqrt(np.mean((pred - z[:, None]) ** 2, axis=0))
    i = int(np.argmin(rms))
    return float(qs[i]), pred[:, i], float(rms[i])


def density_speed_slope(alt_m, speed_m_s, n_boot: int = 2000, seed: int = 0):
    """Slope of ln rho_air(burst altitude) against ln speed.

    If every body had the same strength Y, it would burst where rho v^2 = Y, so ln rho = ln Y - 2 ln v
    and the slope would be exactly -2. Returns the least-squares slope with a bootstrap 95% interval,
    and the Theil-Sen slope (median of pairwise slopes, less sensitive to outliers) with its interval.
    """
    from scipy import stats

    lr = np.log(atmosphere.density(np.asarray(alt_m, dtype=float)))
    lv = np.log(np.asarray(speed_m_s, dtype=float))
    ols = stats.linregress(lv, lr)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, lv.size, (n_boot, lv.size))
    x, y = lv[idx], lr[idx]
    xm, ym = x.mean(1, keepdims=True), y.mean(1, keepdims=True)
    boot = np.sum((x - xm) * (y - ym), 1) / np.sum((x - xm) ** 2, 1)
    ts = stats.theilslopes(lr, lv, alpha=0.95)
    return {
        "ols": float(ols.slope), "ols_95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
        "r2": float(ols.rvalue**2),
        "theil_sen": float(ts.slope), "theil_sen_95": [float(ts.low_slope), float(ts.high_slope)],
    }
