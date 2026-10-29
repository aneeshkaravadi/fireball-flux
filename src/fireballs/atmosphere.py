"""Air density: the U.S. Standard Atmosphere 1976 below 86 km, and a plain exponential atmosphere.

US76 is built from seven layers with a constant temperature gradient in geopotential altitude
(NOAA/NASA/USAF, U.S. Standard Atmosphere 1976, NASA-TM-X-74335). Above 86 km I continue the density
exponentially with the scale height at 86 km. That part isn't US76 and isn't tested against it,
but the air there is so thin that it hardly matters for objects big enough to make a CNEOS fireball.
"""
from __future__ import annotations

import numpy as np

G0 = 9.80665            # m/s^2
R_STAR = 8.31432        # J/(mol K), the US76 value
M0 = 0.0289644          # kg/mol, sea-level mean molar mass
R_EARTH = 6356766.0     # m, the radius US76 uses to convert geometric to geopotential altitude
T0 = 288.15             # K
P0 = 101325.0           # Pa

# layer bases in geopotential metres, and temperature gradients in K/m
_H_BASE = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0])
_LAPSE = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020])
Z_TOP = 86000.0         # geometric metres


def _layer_bases():
    t = [T0]
    p = [P0]
    for i in range(len(_H_BASE) - 1):
        dh = _H_BASE[i + 1] - _H_BASE[i]
        tb, pb, lb = t[-1], p[-1], _LAPSE[i]
        if lb == 0.0:
            p.append(pb * np.exp(-G0 * M0 * dh / (R_STAR * tb)))
        else:
            p.append(pb * (tb / (tb + lb * dh)) ** (G0 * M0 / (R_STAR * lb)))
        t.append(tb + lb * dh)
    return np.array(t), np.array(p)


_T_BASE, _P_BASE = _layer_bases()


def geopotential(z):
    return R_EARTH * z / (R_EARTH + z)


def us76(z):
    """Temperature (K), pressure (Pa) and density (kg/m^3) at geometric altitude z (m), 0 to 86 km."""
    z = np.asarray(z, dtype=float)
    h = geopotential(z)
    i = np.clip(np.searchsorted(_H_BASE, h, side="right") - 1, 0, len(_H_BASE) - 1)
    tb, pb, lb, hb = _T_BASE[i], _P_BASE[i], _LAPSE[i], _H_BASE[i]
    t = tb + lb * (h - hb)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_grad = pb * (tb / t) ** (G0 * M0 / (R_STAR * np.where(lb == 0.0, 1.0, lb)))
    p_iso = pb * np.exp(-G0 * M0 * (h - hb) / (R_STAR * tb))
    p = np.where(lb == 0.0, p_iso, p_grad)
    rho = p * M0 / (R_STAR * t)
    return t, p, rho


_T86, _P86, _RHO86 = (float(v) for v in us76(Z_TOP))
H86 = R_STAR * _T86 / (M0 * G0 * (R_EARTH / (R_EARTH + Z_TOP)) ** 2)  # scale height at 86 km, m


def density(z):
    """Air density (kg/m^3) at geometric altitude z (m): US76 up to 86 km, exponential above."""
    z = np.asarray(z, dtype=float)
    _, _, rho = us76(np.minimum(z, Z_TOP))
    return np.where(z > Z_TOP, _RHO86 * np.exp(-(z - Z_TOP) / H86), rho)


def scale_height(z, dz: float = 50.0):
    """Local density scale height -rho / (drho/dz), m."""
    z = np.asarray(z, dtype=float)
    return 2 * dz / np.log(density(z - dz) / density(z + dz))


def altitude_for_density(rho):
    """Inverse of `density` (m), by interpolation on a fine grid from 0 to 120 km."""
    zz = np.linspace(0.0, 120e3, 24001)
    lr = np.log(density(zz))
    return np.interp(-np.log(np.asarray(rho, dtype=float)), -lr, zz)


def exponential(z, rho0: float = 1.0, h: float = 8000.0):
    """The exponential atmosphere Collins et al. (2005) use: rho0 = 1 kg/m^3, H = 8 km."""
    return rho0 * np.exp(-np.asarray(z, dtype=float) / h)
