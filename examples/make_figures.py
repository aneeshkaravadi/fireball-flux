"""Regenerate every figure in docs/figures/ and every number quoted in the README (docs/results.json).

    python examples/make_figures.py

Reads the committed CNEOS snapshot, so it needs no network. Takes about a minute on one core,
most of it the entry-model inversions.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from fireballs import airburst, atmosphere, data, energy, entry, flux, geometry

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "cneos_fireballs_2026-10-29.csv"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#898781"
plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "font.size": 10, "axes.titlesize": 10.5, "legend.frameon": False,
                     "axes.prop_cycle": plt.cycler(color=[BLUE, ORANGE, AQUA, YELLOW])})

FIRST_YEAR, LAST_YEAR = 2000, 2025      # full calendar years used for rates (see section 1)
BULK_DENSITY = 3000.0                   # kg/m^3, the value Brown et al. (2002) assume for telescopic energies
CHELYABINSK_DENSITY = 3300.0            # Popova et al. (2013), from the meteorites
results: dict = {}
t_start = time.time()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def rnd(x, n=3):
    """Round for the JSON: floats to n significant figures, recursively."""
    if isinstance(x, dict):
        return {k: rnd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, n) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if x == 0 or not np.isfinite(x) else float(f"{x:.{n}g}")
    return x


def pct(a, q=(16, 50, 84)):
    return [float(v) for v in np.percentile(a, q)]


def log(msg):
    print(f"[{time.time() - t_start:5.1f} s] {msg}", flush=True)


df = data.load_snapshot(SNAPSHOT)
df["year"] = df.date.dt.year
results["data"] = {
    "snapshot": SNAPSHOT.name, "events": len(df),
    "first": str(df.date.min()), "last": str(df.date.max()),
    "with_location": int(df.lat.notna().sum()), "with_altitude": int(df.alt_km.notna().sum()),
    "with_velocity": int(df.speed.notna().sum()),
    "with_altitude_and_velocity": int((df.alt_km.notna() & df.speed.notna()).sum()),
}

# ------------------------------------------------------------------ 1. which years are complete enough
yearly = df.groupby("year").size().reindex(range(df.year.min(), LAST_YEAR + 1), fill_value=0)
period = df[(df.year >= FIRST_YEAR) & (df.year <= LAST_YEAR)]
years = LAST_YEAR - FIRST_YEAR + 1
e_all = period.impact_kt.to_numpy()

# threshold: the Emin that makes the power law closest to the data (KS), over all printed energies
emin, cands, ks = flux.choose_threshold(e_all, np.unique(e_all), min_events=50)
counts_above = period[period.impact_kt >= emin].groupby("year").size().reindex(range(FIRST_YEAR, LAST_YEAR + 1), fill_value=0)
counts_1kt = period[period.impact_kt >= 1.0].groupby("year").size().reindex(range(FIRST_YEAR, LAST_YEAR + 1), fill_value=0)
disp_all = flux.dispersion_test(yearly.loc[FIRST_YEAR:LAST_YEAR].to_numpy())
disp_emin = flux.dispersion_test(counts_above.to_numpy())
early, late = counts_above.loc[:2020], counts_above.loc[2021:]
results["completeness"] = {
    "mean_per_year_1994_1999": float(yearly.loc[1994:1999].mean()),
    "mean_per_year_2000_2025": float(yearly.loc[FIRST_YEAR:LAST_YEAR].mean()),
    "events_2000_2025": len(period),
    "dispersion_all": {"index": disp_all[0], "p": disp_all[1]},
    "dispersion_above_emin": {"index": disp_emin[0], "p": disp_emin[1]},
    "rate_above_emin_2000_2020": float(early.mean()), "rate_above_emin_2021_2025": float(late.mean()),
    "p_late_this_low_given_early_rate": float(stats.poisson.cdf(late.sum(), early.mean() * late.size)),
    "ks_shape_early_vs_late_p": float(stats.ks_2samp(period[(period.year <= 2020) & (period.impact_kt >= emin)].impact_kt,
                                                     period[(period.year > 2020) & (period.impact_kt >= emin)].impact_kt).pvalue),
}

fig, ax = plt.subplots(figsize=(8, 3.8))
yrs = yearly.index.to_numpy()
ax.bar(yrs, yearly.to_numpy(), color="#cde2fb", width=0.8, label="all events")
c_e = df[df.impact_kt >= emin].groupby("year").size().reindex(yrs, fill_value=0)
c_1 = df[df.impact_kt >= 1.0].groupby("year").size().reindex(yrs, fill_value=0)
ax.plot(yrs, c_e.to_numpy(), "o-", color=BLUE, ms=4, lw=1.5, label=f"impact energy ≥ {emin:g} kt")
ax.plot(yrs, c_1.to_numpy(), "s-", color=ORANGE, ms=4, lw=1.5, label="≥ 1 kt")
ax.axvspan(FIRST_YEAR - 0.5, LAST_YEAR + 0.5, color="#f0efec", zorder=-5)
ax.text(FIRST_YEAR + 0.2, yearly.max() * 1.02, f"years used for rates: {FIRST_YEAR}-{LAST_YEAR}", color=MUTED, fontsize=8.5)
ax.set_xlabel("year")
ax.set_ylabel("events per year")
ax.set_xlim(1987, LAST_YEAR + 1)
ax.set_title("CNEOS fireballs per year (2026 left out: not a full year)")
ax.legend(loc="upper left", fontsize=8.5)
save(fig, "yearly_counts.png")
log("yearly counts")

# ------------------------------------------------------------------ 2. power-law fit and Brown et al. (2002)
fit = flux.fit(e_all, years, emin)
b_boot, n1_boot = flux.bootstrap(e_all, years, emin, n_boot=4000, seed=0)
p_gof, d_obs = flux.gof_pvalue(e_all, np.unique(e_all), n_sims=1000, seed=1)
n_ge1 = int(np.sum(e_all >= 1.0))
rate_ge1 = n_ge1 / years


def per_year_energy(b, n1):
    return (n1 / 1.0) ** (1.0 / b)


def return_period(b, n1, e_kt):
    return 1.0 / (n1 * e_kt ** -b)


e1_boot = per_year_energy(b_boot, n1_boot)
chel_boot = return_period(b_boot, n1_boot, 440.0)
tung_boot = return_period(b_boot, n1_boot, 1.0e4)
sens = {}
for t in (0.1, 0.2, 0.3, 0.5, 1.0):
    f = flux.fit(e_all, years, t)
    sens[f"{t:g}"] = {"n": f.n, "b": f.b, "b_sigma": f.b / np.sqrt(f.n), "n1": f.n1}
post = df[(df.year >= 2003) & (df.year <= LAST_YEAR)].impact_kt.to_numpy()
f_post = flux.fit(post, LAST_YEAR - 2003 + 1, emin)
bp, np_ = flux.bootstrap(post, LAST_YEAR - 2003 + 1, emin, n_boot=4000, seed=2)
pre21 = df[(df.year >= FIRST_YEAR) & (df.year <= 2020)].impact_kt.to_numpy()
f_pre21 = flux.fit(pre21, 2020 - FIRST_YEAR + 1, emin)
b21, n21 = flux.bootstrap(pre21, 2020 - FIRST_YEAR + 1, emin, n_boot=4000, seed=3)
results["flux"] = {
    "years": [FIRST_YEAR, LAST_YEAR], "emin_kt": emin, "n_above_emin": fit.n, "rate_above_emin": fit.rate_at_emin,
    "b": fit.b, "b_68": pct(b_boot, (16, 84)), "b_95": pct(b_boot, (2.5, 97.5)), "b_boot_sd": float(np.std(b_boot)),
    "n1": fit.n1, "n1_68": pct(n1_boot, (16, 84)), "n1_95": pct(n1_boot, (2.5, 97.5)), "n1_boot_sd": float(np.std(n1_boot)),
    "gof_p": p_gof, "ks_distance": d_obs,
    "direct_count_ge_1kt": n_ge1, "direct_rate_ge_1kt": rate_ge1, "direct_rate_ge_1kt_sigma": np.sqrt(n_ge1) / years,
    "energy_once_per_year_kt": fit.energy_at_rate(1.0), "energy_once_per_year_68": pct(e1_boot, (16, 84)),
    "years_between_440kt": return_period(fit.b, fit.n1, 440.0), "years_between_440kt_68": pct(chel_boot, (16, 84)),
    "years_between_10Mt": return_period(fit.b, fit.n1, 1e4), "years_between_10Mt_68": pct(tung_boot, (16, 84)),
    "threshold_sensitivity": sens,
    "after_brown_2003_2025": {"n": f_post.n, "b": f_post.b, "b_68": pct(bp, (16, 84)), "n1": f_post.n1, "n1_68": pct(np_, (16, 84))},
    "without_2021_2025": {"n": f_pre21.n, "b": f_pre21.b, "b_68": pct(b21, (16, 84)), "n1": f_pre21.n1, "n1_68": pct(n21, (16, 84))},
    "brown2002": {"n1": flux.BROWN2002_N1, "b": flux.BROWN2002_B,
                  "energy_once_per_year_kt": (flux.BROWN2002_N1) ** (1 / flux.BROWN2002_B),
                  "years_between_440kt": 1 / flux.brown2002_rate(440.0), "years_between_10Mt": 1 / flux.brown2002_rate(1e4)},
    "b_minus_brown_in_sd": (fit.b - flux.BROWN2002_B) / np.std(b_boot),
    "n1_minus_brown_in_sd": (fit.n1 - flux.BROWN2002_N1) / np.std(n1_boot),
    "brown_b_percentile_in_bootstrap": float(np.mean(b_boot <= flux.BROWN2002_B) * 100),
}

fig, ax = plt.subplots(figsize=(7.2, 5))
e_sorted, n_cum = flux.cumulative_rate(e_all, years)
ax.loglog(e_sorted, n_cum, ".", color=INK, ms=3.5, label=f"CNEOS {FIRST_YEAR}-{LAST_YEAR} ({len(e_all)} events)")
ee = np.geomspace(emin, 1e3, 100)
band = np.array([n1 * ee ** -b for b, n1 in zip(b_boot[:1000], n1_boot[:1000])])
lo, hi = np.percentile(band, [2.5, 97.5], axis=0)
ax.fill_between(ee, lo, hi, color=BLUE, alpha=0.18, lw=0)
ax.loglog(ee, fit.rate(ee), color=BLUE, lw=2,
          label=f"my fit above {emin:g} kt: N = {fit.n1:.2f} E$^{{-{fit.b:.2f}}}$ (band: 95%)")
eb = np.geomspace(0.05, 1e3, 100)
ax.loglog(eb, flux.brown2002_rate(eb), color=ORANGE, lw=1.5, ls="--",
          label="Brown et al. (2002): N = 3.7 E$^{-0.90}$")
ax.axvspan(*flux.BROWN2002_RANGE_KT, color=ORANGE, alpha=0.06, lw=0)
ax.text(0.11, 0.012, "range Brown et al.\nfit (0.1-10 kt)", color=ORANGE, fontsize=8)
ax.axvline(emin, color=MUTED, lw=0.8, ls=":")
ax.text(emin * 1.05, 60, f"completeness\nthreshold {emin:g} kt", color=MUTED, fontsize=8)
ax.annotate("Chelyabinsk 2013", (e_sorted[0], n_cum[0]), xytext=(80, 0.15), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xlabel("impact energy E (kt TNT)")
ax.set_ylabel("impacts per year with energy ≥ E (whole Earth)")
ax.set_ylim(0.01, 80)
ax.set_xlim(0.04, 1e3)
ax.legend(loc="upper right", fontsize=8.5)
save(fig, "cumulative_flux.png")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
a1.semilogx(cands, ks, color=BLUE, lw=1.5)
a1.axvline(emin, color=MUTED, ls=":", lw=0.8)
a1.set_xlabel("threshold Emin (kt)")
a1.set_ylabel("KS distance, data vs fitted law")
a1.set_title("Picking the threshold")
tt = cands[cands <= 2.0]
bb = np.array([flux.pareto_mle(e_all, t) for t in tt])
nn = np.array([np.sum(e_all >= t) for t in tt])
a2.semilogx(tt, bb, color=BLUE, lw=1.5, label="my fit")
a2.fill_between(tt, bb - bb / np.sqrt(nn), bb + bb / np.sqrt(nn), color=BLUE, alpha=0.18, lw=0)
a2.axhline(flux.BROWN2002_B, color=ORANGE, ls="--", lw=1.5, label="Brown et al. (2002)")
a2.axvline(emin, color=MUTED, ls=":", lw=0.8)
a2.set_xlabel("threshold Emin (kt)")
a2.set_ylabel("exponent b (band: ±1σ)")
a2.set_title("Exponent vs threshold")
a2.legend(fontsize=8.5)
save(fig, "threshold.png")
log("power law")

# ------------------------------------------------------------------ 3. the impact-energy column
rel, match = energy.check_column(df.radiated_J, df.impact_kt)
odd = df[~match]
results["energy_relation"] = {
    "rows": len(df), "rows_matching_after_rounding": int(match.sum()),
    "max_abs_rel_diff_matching_rows": float(np.max(np.abs(rel[match]))),
    "nonmatching": [{"date": str(r.date), "radiated_J": r.radiated_J, "impact_kt": r.impact_kt,
                     "predicted_kt": float(energy.impact_energy_kt(r.radiated_J))} for r in odd.itertuples()],
    "luminous_efficiency_range": [float(energy.luminous_efficiency(df.radiated_J.min())),
                                  float(energy.luminous_efficiency(df.radiated_J.max()))],
}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
a1.loglog(df.radiated_J, df.impact_kt, ".", color=INK, ms=3, label="CNEOS table")
xr = np.geomspace(df.radiated_J.min() / 1.5, df.radiated_J.max() * 1.5, 50)
a1.loglog(xr, energy.impact_energy_kt(xr), color=ORANGE, lw=1.5, label="E = 8.2508 E$_0^{0.885}$")
a1.set_xlabel("total radiated energy (J)")
a1.set_ylabel("listed impact energy (kt)")
a1.legend(fontsize=8.5)
a2.semilogx(df.radiated_J, 100 * rel, ".", color=INK, ms=3)
a2.semilogx(odd.radiated_J, 100 * rel[~match], "o", mfc="none", color=ORANGE, ms=8, label="not 2-3 sig. fig. rounding")
a2.set_xlabel("total radiated energy (J)")
a2.set_ylabel("listed / relation - 1 (%)")
a2.legend(fontsize=8.5)
fig.suptitle("The impact-energy column is the radiated energy run through one formula", fontsize=10.5)
save(fig, "energy_relation.png")
log("energy relation")

# ------------------------------------------------------------------ 4. airburst altitude
ab = df[df.alt_km.notna() & df.speed.notna()].copy()
ab["angle"] = geometry.entry_angle_deg(ab.lat, ab.lon, ab.vx, ab.vy, ab.vz)
n_up = int((ab.angle < 0).sum())
n_graze = int(((ab.angle >= 0) & (ab.angle < 5)).sum())
ab = ab[ab.angle >= 5].reset_index(drop=True)
z_obs = ab.alt_km.to_numpy() * 1e3
v_obs = ab.speed.to_numpy() * 1e3
q_peak = airburst.ram_pressure(z_obs, v_obs)
slope = airburst.density_speed_slope(z_obs, v_obs, n_boot=4000)
q_best, z_ram, rms_ram = airburst.ram_pressure_fit(z_obs, v_obs)
rms_null = float(np.sqrt(np.mean((z_obs - z_obs.mean()) ** 2)))

inv = airburst.implied_strength(z_obs, v_obs, ab.angle.to_numpy(), ab.impact_kt.to_numpy() * energy.KT_J, BULK_DENSITY)
y_single, rms_single, rms_curve = airburst.single_strength_fit(z_obs, inv.grid_burst_alt, inv.strengths)
i_single = int(np.argmin(rms_curve))
good = (inv.flag == 0) & inv.speed_matched
s_good = inv.strength[good]
log("strength inversion (baseline)")

variants = {}
for name, kw in {"density_1500": dict(density=1500.0), "density_3300": dict(density=3300.0),
                 "pancake_factor_4": dict(density=BULK_DENSITY, fp=4.0)}.items():
    d = kw.pop("density")
    iv = airburst.implied_strength(z_obs, v_obs, ab.angle.to_numpy(), ab.impact_kt.to_numpy() * energy.KT_J, d, **kw)
    ok = (iv.flag == 0) & iv.speed_matched
    variants[name] = {"n_resolved": int(ok.sum()), "median_MPa": float(np.median(iv.strength[ok]) / 1e6),
                      "p16_p84_MPa": [v / 1e6 for v in pct(iv.strength[ok], (16, 84))]}
    log(f"strength inversion ({name})")

# the simplest choice: take the CNEOS speed (at peak brightness) as the speed at the top of the atmosphere
L0_raw = entry.diameter_from_energy(ab.impact_kt.to_numpy() * energy.KT_J, v_obs, BULK_DENSITY)
s_raw, f_raw, _ = entry.invert_strength(z_obs, L0_raw, BULK_DENSITY, v_obs, ab.angle.to_numpy())
variants["speed_as_entry_speed"] = {"n_resolved": int(np.sum(f_raw == 0)), "median_MPa": float(np.median(s_raw[f_raw == 0]) / 1e6),
                                    "p16_p84_MPa": [v / 1e6 for v in pct(s_raw[f_raw == 0], (16, 84))]}
log("strength inversion (speed as entry speed)")

rho_sv = stats.spearmanr(s_good, v_obs[good])
rho_se = stats.spearmanr(s_good, ab.impact_kt.to_numpy()[good])
rho_ze = stats.spearmanr(z_obs, ab.impact_kt)
results["airburst"] = {
    "events_with_alt_and_speed": int(len(ab) + n_up + n_graze), "velocity_points_upward": n_up,
    "grazing_under_5deg": n_graze, "used": len(ab),
    "alt_km_16_50_84": [v / 1e3 for v in pct(z_obs)],
    "ram_pressure_at_peak_MPa_5_16_50_84_95": [v / 1e6 for v in pct(q_peak, (5, 16, 50, 84, 95))],
    "density_speed_slope": slope,
    "rms_altitude_spread_km": rms_null / 1e3,
    "ram_pressure_one_strength": {"Q_MPa": q_best / 1e6, "rms_km": rms_ram / 1e3,
                                  "variance_explained": 1 - (rms_ram / rms_null) ** 2},
    "pancake_one_strength": {"Y_MPa": y_single / 1e6, "rms_km": rms_single / 1e3,
                             "variance_explained": 1 - (rms_single / rms_null) ** 2},
    "inversion": {
        "bulk_density": BULK_DENSITY, "pancake_factor": entry.FP,
        "resolved": int(np.sum(inv.flag == 0)), "upper_limits": int(np.sum(inv.flag == -1)),
        "lower_limits": int(np.sum(inv.flag == 1)), "speed_not_matched": int(np.sum(~inv.speed_matched)),
        "resolved_and_matched": int(good.sum()),
        "strength_MPa_5_16_50_84_95": [v / 1e6 for v in pct(s_good, (5, 16, 50, 84, 95))],
        "strength_over_ram_pressure_median": float(np.median(s_good / q_peak[good])),
        "entry_over_reported_speed_5_50_95": pct(inv.entry_speed[inv.speed_matched] / v_obs[inv.speed_matched], (5, 50, 95)),
        "diameter_m_5_50_95": pct(inv.diameter, (5, 50, 95)),
        "spearman_strength_speed": {"rho": rho_sv.statistic, "p": rho_sv.pvalue},
        "spearman_strength_energy": {"rho": rho_se.statistic, "p": rho_se.pvalue},
        "variants": variants,
    },
    "spearman_altitude_energy": {"rho": rho_ze.statistic, "p": rho_ze.pvalue},
}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
for ax, x, rms, title in ((a1, z_ram, rms_ram, f"Ram pressure: burst where ρv² = {q_best / 1e6:.1f} MPa for all"),
                          (a2, inv.grid_burst_alt[:, i_single], rms_single,
                           f"Pancake model, one strength ({y_single / 1e6:.1f} MPa) for all")):
    ax.plot(x / 1e3, z_obs / 1e3, ".", color=BLUE, ms=4)
    ax.plot([10, 95], [10, 95], color=MUTED, lw=0.8)
    ax.set_xlim(10, 95)
    ax.set_ylim(10, 95)
    ax.set_aspect("equal")
    ax.set_xlabel("predicted burst altitude (km)")
    ax.set_title(title, fontsize=9.5)
    ax.text(12, 88, f"RMS error {rms / 1e3:.1f} km\n(spread of the data: {rms_null / 1e3:.1f} km)", fontsize=8.5, va="top")
a1.set_ylabel("CNEOS altitude of peak brightness (km)")
save(fig, "burst_altitude_prediction.png")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
styles = {0: dict(marker="o", color=BLUE, label="resolved"), -1: dict(marker="v", color=ORANGE, label="upper limit"),
          1: dict(marker="^", color=AQUA, label="lower limit")}
for f, st in styles.items():
    m = (inv.flag == f) & inv.speed_matched
    a1.semilogy(v_obs[m] / 1e3, inv.strength[m] / 1e6, st["marker"], color=st["color"], ms=4, ls="none",
                mfc="none" if f else st["color"], label=f"{st['label']} ({m.sum()})")
    a2.loglog(ab.impact_kt.to_numpy()[m], inv.strength[m] / 1e6, st["marker"], color=st["color"], ms=4, ls="none",
              mfc="none" if f else st["color"])
a1.set_xlabel("reported speed at peak brightness (km/s)")
a1.set_ylabel("implied strength (MPa)")
a1.legend(fontsize=8, loc="lower right")
a1.set_title(f"Spearman ρ = {rho_sv.statistic:.2f} (resolved)", fontsize=9.5)
a2.set_xlabel("impact energy (kt)")
a2.set_title(f"Spearman ρ = {rho_se.statistic:.2f} (resolved)", fontsize=9.5)
fig.suptitle(f"Strength that puts the pancake model's peak at the reported altitude ({BULK_DENSITY:.0f} kg/m³, f$_p$ = 7)",
             fontsize=10.5)
save(fig, "implied_strength.png")

fig, ax = plt.subplots(figsize=(6.5, 4))
binsq = np.geomspace(1e4, 3e8, 36)
ax.hist(q_peak, bins=binsq, color="#9ec5f4", label="ram pressure ρv² at peak brightness")
ax.hist(s_good, bins=binsq, histtype="step", color=BLUE, lw=2, label="pancake-model strength (resolved)")
ax.set_xscale("log")
ax.set_xlabel("pressure (Pa)")
ax.set_ylabel("events")
ax.legend(fontsize=8.5)
save(fig, "strength_hist.png")
log("airburst figures")

# ------------------------------------------------------------------ 5. Chelyabinsk
che_row = df.loc[df.impact_kt.idxmax()]
che_angle = float(geometry.entry_angle_deg(che_row.lat, che_row.lon, che_row.vx, che_row.vy, che_row.vz))
che_az = float(geometry.radiant_azimuth_deg(che_row.lat, che_row.lon, che_row.vx, che_row.vy, che_row.vz))
che_inv = airburst.implied_strength([che_row.alt_km * 1e3], [che_row.speed * 1e3], che_angle,
                                    che_row.impact_kt * energy.KT_J, CHELYABINSK_DENSITY, dz=10.0)
# Popova et al. (2013): 19.16 km/s at 97 km, 18.3 deg, 590 kt, 3.3 g/cm^3, peak radiation 29.7 km
POP = dict(v0=19.16e3, angle=18.3, energy_kt=590.0, z_peak=29.7e3, v_peak_satellite=18.6e3, y_onset=0.2e6, diameter=19.8)
pop_L0 = float(entry.diameter_from_energy(POP["energy_kt"] * energy.KT_J, POP["v0"], CHELYABINSK_DENSITY))
s_pop, f_pop, r_pop = entry.invert_strength([POP["z_peak"]], pop_L0, CHELYABINSK_DENSITY, POP["v0"], POP["angle"], dz=10.0)
grid = entry.strength_grid()
v_burst_pop = float(np.interp(np.log10(s_pop[0]), np.log10(grid), r_pop.burst_speed[0]))


def che_strength(z, v0, angle, e_kt):
    L0 = entry.diameter_from_energy(e_kt * energy.KT_J, v0, CHELYABINSK_DENSITY)
    s, f, _ = entry.invert_strength([z], L0, CHELYABINSK_DENSITY, v0, angle, dz=10.0)
    return float(s[0]) / 1e6


steps = {
    "popova_inputs": che_strength(POP["z_peak"], POP["v0"], POP["angle"], POP["energy_kt"]),
    "cneos_altitude": che_strength(che_row.alt_km * 1e3, POP["v0"], POP["angle"], POP["energy_kt"]),
    "cneos_altitude_angle": che_strength(che_row.alt_km * 1e3, POP["v0"], che_angle, POP["energy_kt"]),
    "cneos_altitude_angle_energy": che_strength(che_row.alt_km * 1e3, POP["v0"], che_angle, che_row.impact_kt),
    "cneos_all_speed_matched": float(che_inv.strength[0]) / 1e6,
}
pop_matched = airburst.implied_strength([POP["z_peak"]], [POP["v_peak_satellite"]], POP["angle"],
                                        POP["energy_kt"] * energy.KT_J, CHELYABINSK_DENSITY, dz=10.0)
runs = {
    "cneos": entry.simulate(che_inv.diameter[0], CHELYABINSK_DENSITY, che_inv.entry_speed[0], che_angle, che_inv.strength[0],
                            dz=10.0, keep_profile=True),
    "popova_implied": entry.simulate(pop_L0, CHELYABINSK_DENSITY, POP["v0"], POP["angle"], s_pop[0], dz=10.0, keep_profile=True),
    "popova_0.2MPa": entry.simulate(pop_L0, CHELYABINSK_DENSITY, POP["v0"], POP["angle"], POP["y_onset"], dz=10.0, keep_profile=True),
}
results["chelyabinsk"] = {
    "cneos": {"date": str(che_row.date), "alt_km": che_row.alt_km, "speed_km_s": che_row.speed, "impact_kt": che_row.impact_kt,
              "radiated_J": che_row.radiated_J, "entry_angle_deg": che_angle, "radiant_azimuth_deg": che_az,
              "ram_pressure_at_peak_MPa": float(airburst.ram_pressure(che_row.alt_km * 1e3, che_row.speed * 1e3)) / 1e6,
              "implied_strength_MPa": float(che_inv.strength[0]) / 1e6, "flag": int(che_inv.flag[0]),
              "model_entry_speed_km_s": float(che_inv.entry_speed[0]) / 1e3, "model_diameter_m": float(che_inv.diameter[0]),
              "breakup_alt_km": float(runs["cneos"].breakup_alt) / 1e3},
    "popova2013_inputs": {"diameter_from_energy_m": pop_L0,
                          "ram_pressure_at_peak_MPa": float(airburst.ram_pressure(POP["z_peak"], POP["v_peak_satellite"])) / 1e6,
                          "implied_strength_MPa": float(s_pop[0]) / 1e6, "flag": int(f_pop[0]),
                          "model_speed_at_burst_km_s": v_burst_pop / 1e3,
                          "breakup_alt_km": float(runs["popova_implied"].breakup_alt) / 1e3,
                          "with_0.2MPa": {"breakup_alt_km": float(runs["popova_0.2MPa"].breakup_alt) / 1e3,
                                          "burst_alt_km": float(runs["popova_0.2MPa"].burst_alt) / 1e3,
                                          "pancake_alt_km": float(runs["popova_0.2MPa"].pancake_alt) / 1e3}},
    "popova2013": POP,
    "strength_MPa_swapping_inputs": steps,
    "popova_altitude_speed_matched": {"strength_MPa": float(pop_matched.strength[0]) / 1e6,
                                      "entry_speed_needed_km_s": float(pop_matched.entry_speed[0]) / 1e3},
    "altitude_difference_km": POP["z_peak"] / 1e3 - che_row.alt_km,
}

fig, ax = plt.subplots(figsize=(6.5, 5))
labels = {"cneos": (BLUE, "-", f"CNEOS inputs, implied strength {che_inv.strength[0] / 1e6:.1f} MPa"),
          "popova_implied": (ORANGE, "-", f"Popova et al. inputs, implied strength {s_pop[0] / 1e6:.1f} MPa"),
          "popova_0.2MPa": (ORANGE, "--", "Popova et al. inputs, 0.2 MPa (their fragmentation onset)")}
for k, r in runs.items():
    z = r.profile["z"]
    th = che_angle if k == "cneos" else POP["angle"]
    dEdz = r.profile["power"] / (r.profile["v"] * np.sin(np.radians(th)))   # J per metre of altitude
    col, ls, lab = labels[k]
    ax.plot(dEdz * 1e3 / energy.KT_J, z / 1e3, color=col, ls=ls, lw=1.8, label=lab)
ax.axhline(che_row.alt_km, color=BLUE, lw=0.8, ls=":")
ax.axhline(POP["z_peak"] / 1e3, color=ORANGE, lw=0.8, ls=":")
ax.text(ax.get_xlim()[1] * 0.98, che_row.alt_km - 3.2, f"CNEOS peak brightness {che_row.alt_km} km", ha="right", fontsize=8, color=INK)
ax.text(ax.get_xlim()[1] * 0.98, POP["z_peak"] / 1e3 + 0.6, "Popova et al. peak radiation 29.7 km", ha="right", fontsize=8, color=INK)
ax.set_ylim(10, 60)
ax.set_xlabel("energy deposited (kt per km of altitude)")
ax.set_ylabel("altitude (km)")
ax.set_title("Chelyabinsk in the pancake model")
ax.legend(fontsize=8, loc="upper right")
save(fig, "chelyabinsk.png")
log("Chelyabinsk")

# ------------------------------------------------------------------ 6. speeds, angles, where and when
vel = df[df.speed.notna()].copy()
vel["angle"] = geometry.entry_angle_deg(vel.lat, vel.lon, vel.vx, vel.vy, vel.vz)
mism = vel[np.abs(vel.speed - vel.vel_api) > 0.15]
ang = vel.angle[vel.angle > 0].to_numpy()
ks_ang = stats.kstest(ang, lambda a: geometry.isotropic_entry_angle_cdf(np.clip(a, 0, 90)))
with_v = df[(df.year >= 2008) & (df.year <= LAST_YEAR)]
mw = stats.mannwhitneyu(with_v[with_v.speed.notna()].impact_kt, with_v[with_v.speed.isna()].impact_kt)
loc = df[df.lat.notna()].copy()
sinlat = np.sin(np.radians(loc.lat))
ks_lat = stats.kstest(sinlat, "uniform", args=(-1, 2))
ks_lon = stats.kstest((loc.lon + 180) / 360, "uniform")
lst = geometry.local_solar_time_h(loc.date, loc.lon)
h_all = np.histogram(lst, bins=8, range=(0, 24))[0]
h_bright = np.histogram(lst[loc.impact_kt.to_numpy() >= emin], bins=8, range=(0, 24))[0]
results["population"] = {
    "speed_km_s": {"n": len(vel), "mean": float(vel.speed.mean()), "p5_16_50_84_95": pct(vel.speed, (5, 16, 50, 84, 95)),
                   "max": float(vel.speed.max())},
    "model_entry_speed_mean_km_s": float(np.mean(inv.entry_speed[inv.speed_matched]) / 1e3),
    "brown2002_assumed_mean_speed_km_s": 20.3,
    "vel_field_disagrees": len(mism), "vel_field_disagrees_dates": [str(d) for d in mism.date],
    "entry_angle": {"n": len(ang), "median": float(np.median(ang)), "ks_vs_isotropic_p": float(ks_ang.pvalue),
                    "frac_below_30": float(np.mean(ang < 30)), "isotropic_frac_below_30": 0.25},
    "velocity_reported_events_more_energetic": {"median_with_kt": float(with_v[with_v.speed.notna()].impact_kt.median()),
                                                "median_without_kt": float(with_v[with_v.speed.isna()].impact_kt.median()),
                                                "p": float(mw.pvalue), "years": [2008, LAST_YEAR]},
    "location": {"n": len(loc), "north_fraction": float(np.mean(loc.lat > 0)),
                 "fraction_within_30deg_of_equator": float(np.mean(np.abs(loc.lat) < 30)),
                 "ks_lat_uniform_sphere_p": float(ks_lat.pvalue), "ks_lon_uniform_p": float(ks_lon.pvalue)},
    "local_time": {"bins_3h_all": h_all.tolist(), "chi2_p_all": float(stats.chisquare(h_all).pvalue),
                   "bins_3h_above_emin": h_bright.tolist(), "chi2_p_above_emin": float(stats.chisquare(h_bright).pvalue)},
}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
bins = np.arange(8, 76, 2)
a1.hist(vel.speed, bins=bins, color="#9ec5f4", label=f"reported, at peak brightness ({len(vel)})")
a1.hist(inv.entry_speed[inv.speed_matched] / 1e3, bins=bins, histtype="step", color=ORANGE, lw=1.8,
        label="model entry speed (pancake fit)")
a1.axvline(20.3, color=INK, lw=0.8, ls=":")
a1.text(20.8, a1.get_ylim()[1] * 0.97, "20.3 km/s, the average\nBrown et al. (2002) assume", fontsize=8, va="top")
a1.set_xlabel("speed (km/s)")
a1.set_ylabel("events")
a1.legend(fontsize=8, loc="center right")
aa = np.sort(ang)
a2.plot(aa, np.arange(1, aa.size + 1) / aa.size, color=BLUE, lw=1.8, label=f"CNEOS vectors ({aa.size})")
a2.plot(np.linspace(0, 90, 91), geometry.isotropic_entry_angle_cdf(np.linspace(0, 90, 91)), color=ORANGE, ls="--", lw=1.5,
        label="isotropic: sin²(angle)")
a2.set_xlabel("entry angle above horizontal (deg)")
a2.set_ylabel("fraction shallower than this")
a2.legend(fontsize=8.5)
save(fig, "speed_angle.png")

fig = plt.figure(figsize=(9.5, 5.2))
gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
ax = fig.add_subplot(gs[0])
axh = fig.add_subplot(gs[1], sharey=ax)
order = np.argsort(loc.impact_kt.to_numpy())
lk = np.log10(loc.impact_kt.to_numpy()[order])
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#86b6ef", "#2a78d6", "#104281", "#0b0b0b"])
sc = ax.scatter(loc.lon.to_numpy()[order], loc.lat.to_numpy()[order], c=lk, cmap=cmap, vmin=-1.3, vmax=2.7,
                s=6 + 10 * (lk + 1.4), edgecolors="white", linewidths=0.3)
cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.12, aspect=40)
cb.set_ticks([-1, 0, 1, 2])
cb.set_ticklabels(["0.1", "1", "10", "100"])
cb.set_label("impact energy (kt)")
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xticks(range(-180, 181, 60))
ax.set_yticks(range(-90, 91, 30))
ax.set_xlabel("longitude (deg)")
ax.set_ylabel("latitude (deg)")
ax.set_title(f"Where the {len(loc)} fireballs with a location were", fontsize=9.5)
lat_bins = np.degrees(np.arcsin(np.linspace(-1, 1, 13)))
axh.hist(loc.lat, bins=lat_bins, orientation="horizontal", color="#9ec5f4")
axh.axvline(len(loc) / 12, color=ORANGE, ls="--", lw=1.5, label="uniform on\nthe sphere")
axh.set_xlabel("events per\nequal-area band")
axh.legend(fontsize=7.5, loc="lower right")
plt.setp(axh.get_yticklabels(), visible=False)
fig.savefig(FIG / "world_map.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 3.6))
centers = np.arange(1.5, 24, 3)
ax.bar(centers - 0.6, h_all / h_all.sum(), width=1.2, color="#9ec5f4", label=f"all located ({h_all.sum()}), χ² p = {stats.chisquare(h_all).pvalue:.0e}")
ax.bar(centers + 0.6, h_bright / h_bright.sum(), width=1.2, color=BLUE,
       label=f"≥ {emin:g} kt ({h_bright.sum()}), χ² p = {stats.chisquare(h_bright).pvalue:.2f}")
ax.axhline(1 / 8, color=MUTED, lw=0.8, ls=":")
ax.set_xticks(range(0, 25, 3))
ax.set_xlabel("local solar time (h)")
ax.set_ylabel("fraction of events")
ax.legend(fontsize=8)
save(fig, "local_time.png")
log("population figures")

(ROOT / "docs" / "results.json").write_text(json.dumps(rnd(results), indent=2) + "\n")
log(f"wrote {FIG.relative_to(ROOT)}/ and docs/results.json")
