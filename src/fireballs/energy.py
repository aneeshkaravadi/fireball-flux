"""Optical (radiated) energy -> total impact energy, the empirical relation CNEOS uses.

From the CNEOS fireball introduction page (https://cneos.jpl.nasa.gov/fireballs/intro.html),
which attributes it to Brown et al. (2002), Nature 420, 294-296, doi:10.1038/nature01238:

    E = 8.2508 * E0**0.885        E, E0 in kilotons of TNT, 1 kt = 4.185e12 J

It comes from a fit of the integral luminous efficiency tau = E0 / E against E0, which Hibberd (2023,
arXiv:2307.09085) quotes from Brown et al. (2002) as tau = 0.1212 * E0**0.115. Solving for E gives
the form above, with 8.2508 = 1 / 0.1212 and 0.885 = 1 - 0.115.
"""
from __future__ import annotations

import numpy as np

KT_J = 4.185e12          # J per kiloton, as defined on the CNEOS page
BROWN_C = 8.2508         # E = C * E0**D
BROWN_D = 0.885
TAU_A = 0.1212           # tau = A * E0**B
TAU_B = 0.115


def impact_energy_kt(radiated_J):
    """Total impact energy (kt) from the total radiated energy (J)."""
    e0 = np.asarray(radiated_J, dtype=float) / KT_J
    return BROWN_C * e0**BROWN_D


def radiated_energy_J(impact_kt):
    """Inverse of `impact_energy_kt`."""
    e = np.asarray(impact_kt, dtype=float)
    return (e / BROWN_C) ** (1.0 / BROWN_D) * KT_J


def luminous_efficiency(radiated_J):
    """Fraction of the impact energy that came out as light, tau = E0 / E."""
    e0 = np.asarray(radiated_J, dtype=float) / KT_J
    return TAU_A * e0**TAU_B


def round_sig(x, sig: int = 2):
    """Round to `sig` significant figures (CNEOS prints impact energies this way)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        mag = np.floor(np.log10(np.abs(x)))
    scale = 10.0 ** (sig - 1 - mag)
    return np.round(x * scale) / scale


def check_column(radiated_J, impact_kt):
    """Compare the published impact-energy column with the relation applied to the radiated energy.

    Returns the relative difference of each row and whether the row matches once the prediction is
    rounded to the number of significant figures printed in the table (2, or 3 above 100 kt).
    """
    pred = impact_energy_kt(radiated_J)
    impact_kt = np.asarray(impact_kt, dtype=float)
    rel = impact_kt / pred - 1.0
    match = np.isclose(round_sig(pred, 2), impact_kt, rtol=1e-9) | np.isclose(round_sig(pred, 3), impact_kt, rtol=1e-9)
    return rel, match
