"""Atmospheric entry of a fragmenting meteoroid: drag, breakup, and the "pancake" spread.

The model is the one Collins, Melosh & Marcus (2005, Meteoritics & Planetary Science 40, 817-840)
use in the Earth Impact Effects Program, which follows the pancake model of Chyba, Thomas & Zahnle
(1993, Nature 361, 40-44):

* The body flies in a straight line at a constant angle theta to the horizon (no gravity, flat Earth).
* Drag slows it: dv/dt = -(3/4) C_D rho_air v^2 L^2 / (rho_i L0^3), with C_D = 2. (Collins eq. 6, with
  the frontal area growing as L^2 once it spreads; the mass stays constant because ablation is ignored.)
* It breaks up when the ram pressure rho_air v^2 first exceeds its strength Y (Collins eq. 10).
* After that the crushed body spreads sideways: d^2L/dt^2 = C_D rho_air v^2 / (rho_i L) (Collins eq. 13),
  until its diameter reaches f_p = 7 times the original (Collins's "pancake factor"). Then it keeps
  that size and decelerates as one swarm.

I integrate in altitude rather than time, for many bodies at once with NumPy (fixed-step RK4).
Collins et al. give the "airburst altitude" as the point where L = f_p L0. I also report the altitude
where the rate of energy loss, dE/dt, peaks, which is what I compare with the CNEOS altitude of peak
brightness, assuming the light output follows the energy deposition.

What it leaves out: ablation, several fragmentation events, strength that changes with fragment size,
gravity, and any luminous efficiency that changes during the flight. `simulate` itself uses a flat
Earth; `straight_path` turns a straight path over a round Earth into the inputs `simulate` needs, one
path at a time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import atmosphere

CD = 2.0     # Collins et al. 2005
FP = 7.0     # pancake factor, Collins et al. 2005


@dataclass
class EntryResult:
    breakup_alt: np.ndarray      # m, where rho v^2 first reached the strength (nan if never)
    pancake_alt: np.ndarray      # m, where L reached f_p L0 (Collins's airburst altitude; nan if never)
    burst_alt: np.ndarray        # m, altitude of peak dE/dt (0 if the peak is at the ground)
    burst_speed: np.ndarray      # m/s at burst_alt
    peak_power: np.ndarray       # W
    profile: dict = field(default_factory=dict)  # per-step arrays when keep_profile=True


def mass(diameter, density):
    return density * np.pi * np.asarray(diameter, dtype=float) ** 3 / 6.0


def diameter_from_energy(energy_J, speed, density):
    """Diameter (m) of a sphere with kinetic energy `energy_J` at `speed` (m/s)."""
    m = 2.0 * np.asarray(energy_J, dtype=float) / np.asarray(speed, dtype=float) ** 2
    return (6.0 * m / (np.pi * density)) ** (1.0 / 3.0)


def simulate(diameter, density, speed, angle_deg, strength, *, rho=atmosphere.density, cd: float = CD,
             fp: float = FP, z_top: float = 100e3, dz: float = 25.0, decelerate: bool = True,
             keep_profile: bool = False) -> EntryResult:
    """Fly every combination of the (broadcast) inputs from z_top down to the ground.

    diameter m, density kg/m^3, speed m/s at z_top, angle_deg below horizontal, strength Pa.
    `rho` is a density function of altitude in m (US76 by default). decelerate=False holds the speed
    fixed, which is the assumption behind Collins et al.'s analytic pancake solution.
    """
    L0, rho_i, v, ang, Y = (a.ravel() for a in np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (diameter, density, speed, angle_deg, strength))))
    shape = np.broadcast_shapes(*(np.shape(x) for x in (diameter, density, speed, angle_deg, strength)))
    n = L0.size
    sin_t = np.sin(np.radians(ang))
    m = mass(L0, rho_i)
    kfac = 0.75 * cd / (rho_i * L0**3)          # k = kfac * rho * L^2, dv/dt = -k v^2
    spread = cd / (rho_i * sin_t**2)

    steps = int(round(z_top / dz))
    z = z_top - dz * np.arange(steps + 1)
    rho_full = rho(z)
    rho_half = rho(z[:-1] - dz / 2)

    v = v.copy()
    L = L0.copy()
    P = np.zeros(n)                              # dL/ds, s = depth below z_top
    broken = np.zeros(n, bool)
    capped = np.zeros(n, bool)
    breakup_alt = np.full(n, np.nan)
    pancake_alt = np.full(n, np.nan)
    best = kfac * rho_full[0] * L**2 * m * v**3
    burst_alt = np.full(n, z_top)
    burst_speed = v.copy()
    prof = {"z": z, "v": [v.copy()], "L": [L.copy()], "power": [best.copy()]} if keep_profile else {}

    # breakup can already be due at the top
    q = rho_full[0] * v**2
    now = q >= Y
    broken |= now
    breakup_alt[now] = z_top

    # The speed is integrated as u = ln v. Stepping v itself is unstable once a crushed swarm is low in
    # the air: d(ln v)/ds = -k / sin(theta) can be several per metre, and a 25 m RK4 step on v then
    # overflows. In ln v the right-hand side doesn't depend on u, so it can't run away.
    u = np.log(v)

    def deriv(r, L_, P_, spreading):
        k = kfac * r * L_**2
        du = -k / sin_t if decelerate else np.zeros_like(L_)
        dP = np.where(spreading, spread * r / L_ + (P_ * k / sin_t if decelerate else 0.0), 0.0)
        return du, P_, dP

    for i in range(steps):
        spreading = broken & ~capped
        r0, rh, r1 = rho_full[i], rho_half[i], rho_full[i + 1]
        a1 = deriv(r0, L, P, spreading)
        a2 = deriv(rh, L + 0.5 * dz * a1[1], P + 0.5 * dz * a1[2], spreading)
        a3 = deriv(rh, L + 0.5 * dz * a2[1], P + 0.5 * dz * a2[2], spreading)
        a4 = deriv(r1, L + dz * a3[1], P + dz * a3[2], spreading)
        u = u + dz / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
        v = np.exp(u)
        L = L + dz / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
        P = P + dz / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])

        hit_cap = spreading & (L >= fp * L0)
        if hit_cap.any():
            L[hit_cap] = fp * L0[hit_cap]
            P[hit_cap] = 0.0
            capped |= hit_cap
            pancake_alt[hit_cap] = z[i + 1]

        q1 = r1 * v**2
        new = ~broken & (q1 >= Y)
        if new.any():
            # interpolate the crossing in log(ram pressure) between the two steps
            frac = np.log(Y[new] / q[new]) / np.log(q1[new] / q[new])
            breakup_alt[new] = z[i] - dz * np.clip(frac, 0.0, 1.0)
            broken |= new
        q = q1

        power = kfac * r1 * L**2 * m * v**3
        better = power > best
        best = np.where(better, power, best)
        burst_alt = np.where(better, z[i + 1], burst_alt)
        burst_speed = np.where(better, v, burst_speed)
        if keep_profile:
            prof["v"].append(v.copy())
            prof["L"].append(L.copy())
            prof["power"].append(power.copy())

    if keep_profile:
        prof = {"z": z, **{k: np.array(prof[k]).reshape((steps + 1, *shape)) for k in ("v", "L", "power")}}
    rs = lambda a: a.reshape(shape)  # noqa: E731
    return EntryResult(rs(breakup_alt), rs(pancake_alt), rs(burst_alt), rs(burst_speed), rs(best), prof)


def straight_path(angle_deg: float, angle_alt: float, z_top: float = 100e3, radius: float = 6371e3,
                  rho=atmosphere.density):
    """A straight path over a spherical Earth, in the flat-Earth coordinates `simulate` works in.

    `angle_deg` is the angle below the local horizontal at altitude `angle_alt` (m). Along a straight
    line (R + z) cos(angle) is constant, so the angle grows going up the path and shrinks going down.

    Measured along the path, none of the model's equations involve the angle; it only decides how fast
    the altitude, and so the air density, changes with distance. A flat-Earth run at a constant angle
    with the density re-mapped to the real altitude at each distance is therefore the same calculation.
    The flat angle is picked so the flat run reaches z = 0 where the real path reaches the ground (or
    its lowest point, if it misses), so `simulate` covers the whole path with its usual altitude steps.

    Returns (flat_angle_deg, rho_flat, to_true, to_flat): call simulate(..., flat_angle_deg, ...,
    rho=rho_flat, z_top=z_top), map the altitudes it returns back with to_true(z), and map observed
    altitudes into its coordinates with to_flat(z).
    """
    p = (radius + angle_alt) * np.cos(np.radians(angle_deg))     # closest approach to the Earth's center
    q_top = np.sqrt((radius + z_top) ** 2 - p * p)               # path length from z_top to that point
    s_end = q_top - np.sqrt(radius**2 - p * p) if p < radius else q_top
    sin_f = z_top / s_end

    def to_true(zf):
        s_path = (z_top - np.asarray(zf, dtype=float)) / sin_f
        return np.sqrt(p * p + (q_top - s_path) ** 2) - radius

    def to_flat(z):
        return z_top - (q_top - np.sqrt((radius + np.asarray(z, dtype=float)) ** 2 - p * p)) * sin_f

    def rho_flat(zf):
        return rho(np.maximum(to_true(zf), 0.0))

    return float(np.degrees(np.arcsin(sin_f))), rho_flat, to_true, to_flat


def ram_pressure_altitude(strength, speed):
    """Altitude (m) where rho_US76(z) v^2 = strength, ignoring deceleration."""
    return atmosphere.altitude_for_density(np.asarray(strength, dtype=float) / np.asarray(speed, dtype=float) ** 2)


# --------------------------------------------------------------- Collins et al. (2005) closed forms
# These use their exponential atmosphere, rho = rho0 exp(-z/H) with rho0 = 1 kg/m^3 and H = 8 km.

def collins_speed(z, v0, diameter, density, angle_deg, cd=CD, h=8000.0, rho0=1.0):
    """Speed of an intact body at altitude z (Collins eq. 8)."""
    s = np.sin(np.radians(angle_deg))
    return v0 * np.exp(-3.0 * rho0 * np.exp(-np.asarray(z) / h) * cd * h / (4.0 * density * diameter * s))


def collins_breakup_altitude(strength, v0, diameter, density, angle_deg, cd=CD, h=8000.0, rho0=1.0):
    """Collins's analytic approximation to the breakup altitude (eqs. 11 and 12). Needs I_f < 1."""
    s = np.sin(np.radians(angle_deg))
    i_f = 4.07 * cd * h * strength / (density * diameter * v0**2 * s)
    return -h * (np.log(strength / (rho0 * v0**2)) + 1.308 - 0.314 * i_f - 1.303 * np.sqrt(1.0 - i_f))


def collins_airburst_altitude(z_breakup, diameter, density, angle_deg, cd=CD, fp=FP, h=8000.0, rho0=1.0):
    """Altitude where the pancake reaches fp times its size (Collins eqs. 15 and 16)."""
    s = np.sin(np.radians(angle_deg))
    rho_star = rho0 * np.exp(-z_breakup / h)
    ell = diameter * s * np.sqrt(density / (cd * rho_star))
    return z_breakup - 2.0 * h * np.log(1.0 + ell / (2.0 * h) * np.sqrt(fp**2 - 1.0))


# --------------------------------------------------------------- inverting for strength

def strength_grid(lo: float = 1e3, hi: float = 1e9, per_decade: int = 10):
    return np.logspace(np.log10(lo), np.log10(hi), int(round(np.log10(hi / lo) * per_decade)) + 1)


def invert_strength(burst_alt_obs, diameter, density, speed, angle_deg, strengths=None, tol: float = 50.0, **kw):
    """Strength that makes the model's peak-power altitude equal the observed one, for each body.

    Runs the model on a log-spaced grid of strengths and interpolates in log(strength).

    The burst altitude only depends on strength over a limited range. A weak body breaks up high,
    spreads to f_p times its size and then always peaks at about the same altitude, however weak it
    is; a strong enough body never breaks up at all. Both ends are flat, so the inversion returns a
    flag with each strength:
        0   resolved: the observed altitude is on the sloped part
        -1  upper limit: the observed burst is at or above the weak-body plateau, so strength <= value
        +1  lower limit: the observed burst is at or below what an unbroken body does, so strength >= value
    """
    strengths = strength_grid() if strengths is None else np.asarray(strengths, dtype=float)
    obs = np.asarray(burst_alt_obs, dtype=float).ravel()
    args = [np.asarray(a, dtype=float).ravel()[:, None] for a in (diameter, density, speed, angle_deg)]
    res = simulate(*args, strengths[None, :], **kw)
    zb = np.minimum.accumulate(res.burst_alt, axis=1)  # non-increasing in strength
    logs = np.log10(strengths)
    out = np.full(obs.size, np.nan)
    flag = np.zeros(obs.size, int)
    for j in range(obs.size):
        steep = np.flatnonzero(zb[j, :-1] - zb[j, 1:] > tol)
        if steep.size == 0:
            flag[j], out[j] = (-1, strengths[0]) if obs[j] >= zb[j, 0] - tol else (1, strengths[-1])
            continue
        lo, hi = steep[0], steep[-1] + 1        # sloped part is lo..hi
        if obs[j] >= zb[j, lo] - tol:
            flag[j], out[j] = -1, strengths[lo]
        elif obs[j] <= zb[j, hi] + tol:
            flag[j], out[j] = 1, strengths[hi]
        else:
            out[j] = 10 ** np.interp(-obs[j], -zb[j, lo:hi + 1], logs[lo:hi + 1])
    return out, flag, res
