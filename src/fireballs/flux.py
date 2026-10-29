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


def gof_pvalue(e_kt, candidates, n_sims: int = 1000, seed: int = 0, sig_figs: int | None = 2,
               min_events: int = 50):
    """Goodness-of-fit p-value for the power law, the semi-parametric bootstrap of Clauset et al. (2009).

    Each synthetic catalog has the same size as the real one: with the real tail fraction, an event
    comes from the fitted power law above Emin, otherwise it is resampled from the real events below
    Emin. Each one is then refit from scratch, threshold choice included, and the p-value is the
    fraction whose KS distance is at least the real one. Small p (Clauset suggest p <= 0.1) means
    the power law is ruled out.

    CNEOS prints impact energies to two significant figures, which creates ties that make the KS
    distance a bit bigger. Rounding the synthetic catalogs the same way (sig_figs=2) keeps that from
    counting against the power law.
    """
    from .energy import round_sig

    rng = np.random.default_rng(seed)
    e = np.asarray(e_kt, dtype=float)
    emin, _, _ = choose_threshold(e, candidates, min_events)
    b = pareto_mle(e, emin)
    d_obs = ks_distance(e, emin, b)
    body = e[e < emin]
    p_tail = np.mean(e >= emin)
    d_sim = np.empty(n_sims)
    for i in range(n_sims):
        n_t = rng.binomial(e.size, p_tail)
        x = np.concatenate([emin * rng.uniform(size=n_t) ** (-1.0 / b),
                            rng.choice(body, e.size - n_t) if body.size else np.empty(0)])
        if sig_figs:
            x = round_sig(x, sig_figs)
        em, _, _ = choose_threshold(x, candidates, min_events)
        d_sim[i] = ks_distance(x, em, pareto_mle(x, em))
    return float(np.mean(d_sim >= d_obs)), float(d_obs)
