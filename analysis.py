"""End-to-end analysis of an e-commerce landing-page A/B test.

Usage: python analysis.py data/ab_data.csv data/countries.csv
Writes results/results.json and figures to results/.
"""
import json
import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
ALPHA, POWER = 0.05, 0.80
RNG = np.random.default_rng(42)


def run_sql_file(con, path):
    """Run every statement in a .sql file; return the result frames of the SELECTs."""
    text = "\n".join(l for l in path.read_text().splitlines() if not l.strip().startswith("--"))
    frames = []
    for stmt in (s.strip() for s in text.split(";")):
        if stmt.upper().startswith(("SELECT", "WITH")):
            frames.append(con.sql(stmt).df())
        elif stmt:
            con.execute(stmt)
    return frames


def load(ab_path, countries_path):
    con = duckdb.connect()
    con.execute("CREATE TABLE ab AS SELECT * FROM read_csv_auto(?)", [ab_path])
    con.execute("CREATE TABLE countries AS SELECT * FROM read_csv_auto(?)", [countries_path])
    raw_rows = con.execute("SELECT COUNT(*) FROM ab").fetchall()[0][0]
    assignment, dupes, window = run_sql_file(con, ROOT / "sql/01_data_quality.sql")
    arms, daily, by_country = run_sql_file(con, ROOT / "sql/02_clean_and_metrics.sql")
    clean = con.sql("SELECT * FROM clean").df()

    mismatched = int(assignment.loc[
        ((assignment.group == "control") & (assignment.landing_page == "new_page"))
        | ((assignment.group == "treatment") & (assignment.landing_page == "old_page")),
        "n_rows"].sum())
    quality = {
        "raw_rows": int(raw_rows),
        "mismatched_assignment_rows": mismatched,
        "duplicated_users": int(dupes.duplicated_users[0]),
        "clean_users": int(len(clean)),
        "rows_dropped_pct": round(100 * (1 - len(clean) / raw_rows), 2),
        "first_event": str(window.first_event[0]),
        "last_event": str(window.last_event[0]),
        "n_days": int(window.n_days[0]),
    }
    return clean, arms.set_index("group"), daily, by_country, quality


def srm_check(arms):
    """Sample-ratio mismatch: chi-square of observed arm sizes against a 50/50 split."""
    n = arms.users.to_numpy()
    chi2, p = stats.chisquare(n, [n.sum() / 2] * 2)
    return {"control_users": int(n[0]), "treatment_users": int(n[1]),
            "chi2": round(float(chi2), 3), "p_value": round(float(p), 4), "srm_detected": bool(p < 0.001)}


def power_analysis(p_control, n_per_arm, daily_users):
    """Minimum detectable effect at the achieved sample, and sample needed for target lifts."""
    solver = NormalIndPower()
    es = solver.solve_power(nobs1=n_per_arm, alpha=ALPHA, power=POWER, ratio=1.0)
    # invert Cohen's h back to an absolute lift on the baseline
    mde_abs = np.sin(np.arcsin(np.sqrt(p_control)) + es / 2) ** 2 - p_control
    needed = {}
    for rel_lift in (0.01, 0.02, 0.05):
        h = proportion_effectsize(p_control * (1 + rel_lift), p_control)
        n = solver.solve_power(effect_size=h, alpha=ALPHA, power=POWER, ratio=1.0)
        needed[f"{int(rel_lift * 100)}pct_relative_lift"] = {
            "users_per_arm": int(np.ceil(n)),
            "days_at_current_traffic": round(2 * n / daily_users, 1),
        }
    return {"baseline_rate": round(p_control, 5),
            "mde_absolute_pp": round(100 * mde_abs, 3),
            "mde_relative_pct": round(100 * mde_abs / p_control, 2),
            "sample_needed": needed}


def frequentist(arms):
    conv = arms.conversions.to_numpy()
    n = arms.users.to_numpy()
    p_c, p_t = conv / n
    z, p_two = proportions_ztest(conv[::-1], n[::-1])  # treatment - control
    _, p_one = proportions_ztest(conv[::-1], n[::-1], alternative="larger")
    diff = p_t - p_c
    se = np.sqrt(p_c * (1 - p_c) / n[0] + p_t * (1 - p_t) / n[1])
    zc = stats.norm.ppf(1 - ALPHA / 2)
    # bootstrap: resampling Bernoulli outcomes is a binomial draw per arm
    B = 20_000
    boot = RNG.binomial(n[1], p_t, B) / n[1] - RNG.binomial(n[0], p_c, B) / n[0]
    return {"control_rate": round(p_c, 5), "treatment_rate": round(p_t, 5),
            "abs_diff_pp": round(100 * diff, 3), "rel_lift_pct": round(100 * diff / p_c, 2),
            "z": round(float(z), 3), "p_two_sided": round(float(p_two), 4),
            "p_one_sided_new_better": round(float(p_one), 4),
            "wald_ci_pp": [round(100 * (diff - zc * se), 3), round(100 * (diff + zc * se), 3)],
            "bootstrap_ci_pp": [round(100 * np.quantile(boot, 0.025), 3),
                                round(100 * np.quantile(boot, 0.975), 3)]}


def bayesian(arms, draws=200_000):
    """Beta(1,1) priors; probability treatment beats control and expected loss of shipping it."""
    post = {g: RNG.beta(1 + r.conversions, 1 + r.users - r.conversions, draws) for g, r in arms.iterrows()}
    delta = post["treatment"] - post["control"]
    return {"p_treatment_better": round(float((delta > 0).mean()), 4),
            "expected_loss_if_ship_pp": round(100 * float(np.maximum(-delta, 0).mean()), 4),
            "credible_interval_pp": [round(100 * float(np.quantile(delta, q)), 3) for q in (0.025, 0.975)]}


def peeking_simulation(p, daily_users, n_sims=4000):
    """A/A tests on this experiment's traffic: checking a z-test every day and stopping at the
    first p<0.05 inflates the false-positive rate far above the nominal 5%."""
    per_arm = np.asarray(daily_users) // 2
    conv_a = np.cumsum(RNG.binomial(per_arm, p, (n_sims, len(per_arm))), axis=1)
    conv_b = np.cumsum(RNG.binomial(per_arm, p, (n_sims, len(per_arm))), axis=1)
    n_cum = np.cumsum(per_arm)
    pa, pb = conv_a / n_cum, conv_b / n_cum
    pool = (conv_a + conv_b) / (2 * n_cum)
    z = (pb - pa) / np.sqrt(pool * (1 - pool) * 2 / n_cum)
    sig = np.abs(z) > stats.norm.ppf(1 - ALPHA / 2)
    return {"days": len(per_arm), "fpr_fixed_horizon_pct": round(100 * sig[:, -1].mean(), 2),
            "fpr_daily_peeking_pct": round(100 * sig.any(axis=1).mean(), 2)}


def segments(clean, by_country):
    """Per-country z-tests with Holm correction, plus a logistic model with an interaction."""
    rows, pvals = [], []
    for country, g in by_country.groupby("country"):
        g = g.set_index("group")
        conv = (g.users * g.conversion_rate).round().astype(int)
        _, p = proportions_ztest([conv["treatment"], conv["control"]], [g.users["treatment"], g.users["control"]])
        rows.append({"country": country, "control_rate": round(g.conversion_rate["control"], 5),
                     "treatment_rate": round(g.conversion_rate["treatment"], 5),
                     "abs_diff_pp": round(100 * (g.conversion_rate["treatment"] - g.conversion_rate["control"]), 3),
                     "p_raw": round(float(p), 4)})
        pvals.append(p)
    reject, p_holm, _, _ = multipletests(pvals, alpha=ALPHA, method="holm")
    for r, ph, rej in zip(rows, p_holm, reject):
        r["p_holm"], r["significant_after_holm"] = round(float(ph), 4), bool(rej)

    df = clean.assign(treat=(clean.group == "treatment").astype(int))
    full = smf.logit("converted ~ treat * C(country, Treatment('US'))", data=df).fit(disp=0)
    reduced = smf.logit("converted ~ treat + C(country, Treatment('US'))", data=df).fit(disp=0)
    lr = 2 * (full.llf - reduced.llf)
    dof = full.df_model - reduced.df_model
    return {"per_country": rows,
            "interaction_lr_test": {"chi2": round(float(lr), 3), "df": int(dof),
                                    "p_value": round(float(stats.chi2.sf(lr, dof)), 4)},
            "treatment_odds_ratio": round(float(np.exp(reduced.params["treat"])), 4),
            "treatment_or_ci": [round(float(np.exp(c)), 4) for c in reduced.conf_int().loc["treat"]]}


def novelty(daily):
    """Does the daily lift trend over time (novelty or change aversion)?"""
    wide = daily.pivot(index="day", columns="group", values="conversion_rate").sort_index()
    wide = wide.iloc[1:-1]  # first and last days are partial
    lift = 100 * (wide["treatment"] - wide["control"])
    slope = stats.linregress(np.arange(len(lift)), lift.to_numpy())
    return wide, lift, {"days_used": len(lift), "slope_pp_per_day": round(float(slope.slope), 4),
                        "slope_p_value": round(float(slope.pvalue), 4)}


def plots(wide, lift, freq, peek):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    ax.plot(wide.index, 100 * wide["control"], label="control (old page)")
    ax.plot(wide.index, 100 * wide["treatment"], label="treatment (new page)")
    ax.set_title("Daily conversion rate"), ax.set_ylabel("%"), ax.legend()
    ax.tick_params(axis="x", rotation=45)
    ax = axes[1]
    ax.axhline(0, color="grey", lw=1)
    ax.plot(lift.index, lift.to_numpy(), marker="o")
    ax.set_title("Daily lift (treatment - control)"), ax.set_ylabel("pp")
    ax.tick_params(axis="x", rotation=45)
    ax = axes[2]
    lo, hi = freq["bootstrap_ci_pp"]
    ax.errorbar([0], [freq["abs_diff_pp"]], yerr=[[freq["abs_diff_pp"] - lo], [hi - freq["abs_diff_pp"]]],
                fmt="o", capsize=8)
    ax.axhline(0, color="grey", lw=1)
    ax.set_xticks([0], ["overall lift\n95% bootstrap CI"]), ax.set_ylabel("pp")
    ax.set_title(f"A/A peeking FPR: {peek['fpr_daily_peeking_pct']}% vs {peek['fpr_fixed_horizon_pct']}%")
    fig.tight_layout()
    fig.savefig(OUT / "summary.png", dpi=120)


def main(ab_path, countries_path):
    OUT.mkdir(exist_ok=True)
    clean, arms, daily, by_country, quality = load(ab_path, countries_path)
    daily_users = daily.groupby("day").users.sum().sort_index()
    p_c = float(arms.loc["control", "conversion_rate"])

    freq = frequentist(arms)
    peek = peeking_simulation(p_c, daily_users.to_numpy())
    wide, lift, nov = novelty(daily)
    results = {
        "data_quality": quality,
        "srm": srm_check(arms),
        "power": power_analysis(p_c, int(arms.users.min()), float(daily_users.iloc[1:-1].mean())),
        "frequentist": freq,
        "bayesian": bayesian(arms),
        "peeking_simulation": peek,
        "segments": segments(clean, by_country),
        "novelty": nov,
    }
    plots(wide, lift, freq, peek)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:3])
