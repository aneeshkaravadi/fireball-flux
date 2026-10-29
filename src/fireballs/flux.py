"""Cumulative impact rate N(>=E) per year, and power-law fits to it.

A cumulative power law N(>=E) = N1 * E**-b (E in kt, N per year over the whole Earth) means the
energies above a threshold Emin follow a Pareto distribution with index b. I fit b by maximum
likelihood, b = n / sum(ln(E_i / Emin)), and pick Emin the way Clauset, Shalizi & Newman (2009,
SIAM Review 51, 661, arXiv:0706.1062) suggest: the threshold that makes the fitted law closest to the
data in the Kolmogorov-Smirnov sense.

Published comparison: Brown et al. (2002, Nature 420, 294) fit N = 3.7 E**-0.90 to satellite
fireballs, valid for about 0.1 to 10 kt. The paper itself is paywalled; that form and range are as
quoted in Borovicka, Spurny & Brown (2015, Asteroids IV, arXiv:1502.03307), which Brown co-wrote.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

BROWN2002_N1 = 3.7        # impacts per year at or above 1 kt
BROWN2002_B = 0.90
BROWN2002_RANGE_KT = (0.1, 10.0)


def brown2002_rate(e_kt):
    """Brown et al. (2002): impacts per year with energy >= E (kt)."""
    return BROWN2002_N1 * np.asarray(e_kt, dtype=float) ** -BROWN2002_B


def cumulative_rate(e_kt, years: float):
    """Energies sorted high to low, and the number per year at or above each one."""
    e = np.sort(np.asarray(e_kt, dtype=float))[::-1]
    return e, np.arange(1, e.size + 1) / years


def dispersion_test(counts):
    """Are yearly counts consistent with one Poisson rate? Returns (index of dispersion, p-value)."""
    c = np.asarray(counts, dtype=float)
    d = np.sum((c - c.mean()) ** 2) / c.mean()
    return d / (c.size - 1), float(stats.chi2.sf(d, c.size - 1))


def pareto_mle(e_kt, emin: float) -> float:
    x = np.asarray(e_kt, dtype=float)
    x = x[x >= emin]
    return x.size / np.sum(np.log(x / emin))


def ks_distance(e_kt, emin: float, b: float) -> float:
    """Largest gap between the empirical and fitted cumulative distributions above emin."""
    x = np.sort(np.asarray(e_kt, dtype=float))
    x = x[x >= emin]
    n = x.size
    model = 1.0 - (x / emin) ** -b
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    return float(max(np.max(np.abs(upper - model)), np.max(np.abs(lower - model))))


def choose_threshold(e_kt, candidates, min_events: int = 50):
    """Emin that minimizes the KS distance, plus the distance for every candidate tried."""
    e = np.asarray(e_kt, dtype=float)
    cands = np.array([c for c in np.unique(candidates) if np.sum(e >= c) >= min_events])
    d = np.array([ks_distance(e, c, pareto_mle(e, c)) for c in cands])
    return float(cands[np.argmin(d)]), cands, d


@dataclass
class PowerLawFit:
    emin: float          # kt
    n: int               # events at or above emin
    years: float
    b: float             # N(>=E) proportional to E**-b

    @property
    def rate_at_emin(self) -> float:
        return self.n / self.years

    @property
    def n1(self) -> float:
        """Fitted impacts per year at or above 1 kt."""
        return self.rate_at_emin * self.emin**self.b

    def rate(self, e_kt):
        return self.n1 * np.asarray(e_kt, dtype=float) ** -self.b

    def energy_at_rate(self, per_year: float) -> float:
        """Energy exceeded `per_year` times a year on average."""
        return (self.n1 / per_year) ** (1.0 / self.b)


def fit(e_kt, years: float, emin: float) -> PowerLawFit:
    e = np.asarray(e_kt, dtype=float)
    return PowerLawFit(emin=emin, n=int(np.sum(e >= emin)), years=years, b=pareto_mle(e, emin))


def bootstrap(e_kt, years: float, emin: float, n_boot: int = 4000, seed: int = 0):
    """Resample the events (with a Poisson-distributed total) and refit.

    Returns arrays of b and N1 (rate at 1 kt) for each resample, so the spread includes both the
    shape uncertainty and the counting noise on the total rate.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(e_kt, dtype=float)
    x = x[x >= emin]
    logs = np.log(x / emin)
    b = np.empty(n_boot)
    n1 = np.empty(n_boot)
    for i in range(n_boot):
        n = rng.poisson(x.size)
        s = logs[rng.integers(0, x.size, n)]
        b[i] = n / s.sum()
        n1[i] = n / years * emin ** b[i]
    return b, n1
