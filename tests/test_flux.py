"""Power-law fitting on SYNTHETIC samples with a known answer, and the published Brown et al. law."""
import numpy as np
import pytest

from fireballs import flux


def synthetic_pareto(b, emin, n, rng):
    """Synthetic energies with N(>=E) proportional to E**-b above emin (inverse-CDF sampling)."""
    return emin * rng.uniform(size=n) ** (-1.0 / b)


def test_mle_recovers_known_exponent():
    rng = np.random.default_rng(1)
    for b_true in (0.6, 0.9, 1.2):
        x = synthetic_pareto(b_true, 0.3, 4000, rng)
        b = flux.pareto_mle(x, 0.3)
        assert b == pytest.approx(b_true, abs=4 * b_true / np.sqrt(x.size))  # MLE sigma is b / sqrt(n)


def test_bootstrap_interval_covers_the_true_exponent_at_the_right_rate():
    """Over 200 synthetic data sets of 300 events, the 68% interval should contain b about 68% of the time."""
    rng = np.random.default_rng(2)
    hits = 0
    for k in range(200):
        x = synthetic_pareto(0.9, 0.2, 300, rng)
        b, _, _ = flux.bootstrap(x, years=20.0, emin=0.2, n_boot=400, seed=k)
        lo, hi = np.percentile(b, [16, 84])
        hits += lo <= 0.9 <= hi
    assert 0.58 <= hits / 200 <= 0.78


def test_rounding_edge():
    assert flux.rounding_edge(0.24, 2) == pytest.approx(0.235)
    assert flux.rounding_edge(1.0, 2) == pytest.approx(0.95)
    assert flux.rounding_edge(0.1, 2) == pytest.approx(0.095)
    assert flux.rounding_edge(12.0, 2) == pytest.approx(11.5)
    assert flux.rounding_edge(0.24, None) == 0.24


def test_fit_recovers_the_exponent_from_a_rounded_catalog():
    """SYNTHETIC: Pareto samples printed to 2 significant figures, like the CNEOS table. With the
    rounding edge as the lower bound the fit gets b back; treating the printed threshold as exact
    biases it upward, which is the mistake I first made."""
    from fireballs.energy import round_sig

    rng = np.random.default_rng(11)
    exact, edge = [], []
    for _ in range(40):
        x = round_sig(synthetic_pareto(0.9, 0.05, 6000, rng), 2)
        exact.append(flux.pareto_mle(x, 0.24))
        edge.append(flux.pareto_mle(x, 0.24, sig_figs=2))
    se = 0.9 / np.sqrt(np.sum(x >= 0.24)) / np.sqrt(40)           # standard error of the mean over 40 catalogs
    assert np.mean(edge) == pytest.approx(0.9, abs=3 * se)
    assert np.mean(exact) - 0.9 > 5 * se


def test_bootstrap_with_the_threshold_chosen_again_is_wider():
    """SYNTHETIC: choosing the threshold again in every resample adds its uncertainty to the spread."""
    rng = np.random.default_rng(12)
    x = synthetic_pareto(0.9, 0.05, 8000, rng)
    x = x[rng.uniform(size=x.size) < np.clip((x / 0.25) ** 3, 0, 1)]        # incomplete below ~0.25
    cands = np.geomspace(0.1, 2.0, 30)
    emin, _, _ = flux.choose_threshold(x, cands)
    b_fix, _, em_fix = flux.bootstrap(x, 20.0, emin, n_boot=300, seed=1)
    b_var, n1_var, em_var = flux.bootstrap(x, 20.0, emin, n_boot=300, seed=1, candidates=cands)
    assert np.all(em_fix == emin) and np.ptp(em_var) > 0
    assert np.std(b_var) > np.std(b_fix)
    lo, hi = np.percentile(b_var, [2.5, 97.5])
    assert lo < 0.9 < hi


def test_rate_normalization():
    rng = np.random.default_rng(3)
    years = 25.0
    x = synthetic_pareto(0.9, 0.2, 5000, rng)
    f = flux.fit(x, years, 0.2)
    assert f.rate_at_emin == pytest.approx(5000 / 25.0)
    # N1 = rate(emin) * emin**b, and rate(1 kt) should match the count above 1 kt
    assert f.rate(1.0) == pytest.approx(np.sum(x >= 1.0) / years, rel=0.06)
    assert f.energy_at_rate(f.rate(3.0)) == pytest.approx(3.0)


def test_threshold_choice_finds_where_detection_becomes_complete():
    """Synthetic catalog that misses most events below 0.25 kt; the KS choice should land near 0.25."""
    rng = np.random.default_rng(4)
    x = synthetic_pareto(0.9, 0.05, 40000, rng)
    keep = rng.uniform(size=x.size) < np.clip((x / 0.25) ** 3, 0, 1)  # detection efficiency
    x = x[keep]
    emin, _, _ = flux.choose_threshold(x, np.geomspace(0.06, 2.0, 40))
    assert 0.2 <= emin <= 0.45
    assert flux.pareto_mle(x, emin) == pytest.approx(0.9, abs=0.05)


def test_cumulative_rate_counts():
    e, n = flux.cumulative_rate([0.1, 5.0, 1.0, 0.3], years=2.0)
    assert list(e) == [5.0, 1.0, 0.3, 0.1]
    assert list(n) == [0.5, 1.0, 1.5, 2.0]


def test_dispersion_test():
    rng = np.random.default_rng(5)
    _, p = flux.dispersion_test(rng.poisson(25, size=30))
    assert p > 0.01
    _, p = flux.dispersion_test(np.r_[np.full(15, 10), np.full(15, 30)])
    assert p < 1e-6


def test_brown2002_law_matches_the_abstract():
    """Nature 420, 294 abstract: about one 5 kt event a year, and Tunguska-like (10 Mt) about every 1,000 years."""
    assert flux.brown2002_rate(5.0) == pytest.approx(1.0, rel=0.15)
    assert 1.0 / flux.brown2002_rate(1.0e4) == pytest.approx(1000, rel=0.1)


def test_goodness_of_fit_accepts_a_power_law_and_rejects_a_lognormal():
    """SYNTHETIC: a rounded power-law catalog passes the Clauset et al. test, a lognormal one fails."""
    from fireballs.energy import round_sig

    rng = np.random.default_rng(7)
    x = round_sig(synthetic_pareto(0.9, 0.2, 600, rng), 2)
    p, _ = flux.gof_pvalue(x, np.unique(x), n_sims=100)
    assert p > 0.1
    y = round_sig(np.exp(rng.normal(np.log(0.2), 1.0, 3000)), 2)
    p, _ = flux.gof_pvalue(y, np.unique(y), n_sims=100)
    assert p < 0.05
