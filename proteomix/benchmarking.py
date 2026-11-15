"""
Benchmarking study: type I error, FDR, type II error (power) and logFC accuracy of every
combination of normalisation x missing-value handling x variance moderation, with a Welch
t-test on the observed values as baseline.

Run:  python -m proteomix.benchmarking               (full grid, 10 simulations per scenario)
      python -m proteomix.benchmarking --quick       (small grid, quick check)
      python -m proteomix.benchmarking --n-sims 20 --out results_dir
"""
import argparse
import warnings
import itertools
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
from scipy import stats

from .normalisation import NORMALISATIONS, normalise
from .pipeline import MISSING_HANDLING, run_differential_abundance
from .replicates import collapse_replicates
from .simulation import simulate_dataset
from .testing import benjamini_hochberg

METRICS = ["type1_per_test", "ks_null", "fdp", "any_false_disc", "power", "type2",
           "power_tested", "lfc_bias", "lfc_rmse", "n_tested", "seconds"]
BASELINE = "baseline | Welch t-test (observed)"


# --------------------------------------------------------------------------------------
# Baseline and evaluation
# --------------------------------------------------------------------------------------
def welch_test(y, x, g1=1, g0=0):
    """Welch t-test on observed values (NaN dropped), per feature. Needs >= 2 values per group."""
    a, b = y[x[:, g1] == 1], y[x[:, g0] == 1]
    na, nb = (~np.isnan(a)).sum(0), (~np.isnan(b)).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ma, mb = np.nanmean(a, 0), np.nanmean(b, 0)
        va, vb = np.nanvar(a, 0, ddof=1) / na, np.nanvar(b, 0, ddof=1) / nb
        se = np.sqrt(va + vb)
        t = (ma - mb) / se
        df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    p = 2 * stats.t.sf(np.abs(t), df)
    p[(na < 2) | (nb < 2) | ~np.isfinite(p)] = np.nan
    return ma - mb, p


def evaluate(pvalue, padj, estimate, truth, alpha=0.05) -> dict:
    """
    type1_per_test : fraction of null features with p < alpha (should be ~ alpha)
    ks_null        : Kolmogorov-Smirnov distance of null p-values from Uniform(0, 1)
    fdp            : false discoveries / discoveries at padj < alpha (should be <= alpha on average)
    any_false_disc : at least one false discovery (family-wise error, informative under full null)
    power / type2  : fraction of DE features discovered at padj < alpha / its complement;
                     features that could not be tested count as missed
    power_tested   : the same, among DE features that could be tested
    lfc_bias       : mean signed error of the DE estimates, + = |logFC| overestimated
    lfc_rmse       : root mean squared error of the DE estimates
    """
    is_de, true_lfc = truth["is_de"], truth["true_logfc"]
    null = ~is_de
    tested = ~np.isnan(pvalue)
    disc = np.nan_to_num(padj, nan=1.0) < alpha
    p_null = pvalue[null & tested]
    n_disc = disc.sum()
    out = {
        "n_tested": int(tested.sum()),
        "type1_per_test": float(np.mean(p_null < alpha)) if p_null.size else np.nan,
        "ks_null": float(stats.kstest(p_null, "uniform").statistic) if p_null.size else np.nan,
        "n_discoveries": int(n_disc),
        "fdp": float((disc & null).sum() / n_disc) if n_disc else 0.0,
        "any_false_disc": float((disc & null).any()),
    }
    if is_de.any():
        power = (disc & is_de).sum() / is_de.sum()
        de_t = is_de & tested & np.isfinite(estimate)
        err = estimate[de_t] - true_lfc[de_t]
        out.update({
            "power": float(power),
            "type2": float(1 - power),
            "power_tested": float((disc & de_t).sum() / max(de_t.sum(), 1)),
            "lfc_bias": float(np.mean(err * np.sign(true_lfc[de_t]))),
            "lfc_rmse": float(np.sqrt(np.mean(err ** 2))),
        })
    return out


# --------------------------------------------------------------------------------------
# Method grid
# --------------------------------------------------------------------------------------
def method_grid(normalisations=NORMALISATIONS, missing=MISSING_HANDLING,
                moderate=(False, True), **common) -> dict:
    """
    All combinations as {name: kwargs for run_differential_abundance}.
    Names look like "median_ratio | selection | moderated". Extra kwargs (e.g. robust=True,
    include_sigma=False, min_present=2) are passed to every method.
    """
    methods = {}
    for norm, miss, mod in itertools.product(normalisations, missing, moderate):
        name = f"{norm} | {miss} | {'moderated' if mod else 'plain'}"
        methods[name] = dict(normalisation=norm, missing=miss, moderate=mod, **common)
    return methods


# --------------------------------------------------------------------------------------
# Study runner
# --------------------------------------------------------------------------------------
def run_study(scenarios: dict, methods: dict, n_sims: int = 10, alpha: float = 0.05,
              seed: int = 0, baseline: bool = True, verbose: bool = True,
              checkpoint: str | None = None):
    """
    scenarios: {name: kwargs for simulate_dataset}
    methods:   {name: kwargs for run_differential_abundance}, e.g. from method_grid()
    Avoid scenario names that pandas reads as missing in CSV files ("null", "NA", "None").
    Every method sees the same simulated datasets. A method that raises an error is recorded
    with NaN metrics and the error message, instead of stopping the study.
    Method names are free text: the normalisation / missing / variance columns are taken from
    each method's kwargs (pipeline defaults where not given), not parsed from the name.
    checkpoint: CSV path; the per-simulation rows are rewritten there after every simulation,
    so a crash or interruption late in a long study loses at most one simulation.
    Returns (per-simulation DataFrame, summary DataFrame of means and SDs, null p-values dict).
    """
    from .progress import Progress
    rows, null_p = [], {}
    bar = Progress(len(scenarios) * n_sims * (len(methods) + bool(baseline)),
                   desc="Benchmark", enabled=verbose)
    for s_idx, (sc_name, sim_kwargs) in enumerate(scenarios.items()):
        for i in range(n_sims):
            sim_seed = seed + 1000 * s_idx + i
            data = simulate_dataset(seed=sim_seed, **sim_kwargs)
            Y = torch.tensor(data["Y"])
            X = torch.tensor(data["X"])
            b = torch.tensor(data["batch_ids"], dtype=torch.long)
            rep = torch.tensor(data["replicate_ids"], dtype=torch.long)
            null = ~data["is_de"]
            base = {"scenario": sc_name, "sim": i, "missing_rate": data["missing_rate"]}

            for m_name, m_kwargs in methods.items():
                t0 = time.perf_counter()
                try:
                    res, *_ = run_differential_abundance(
                        Y, X, b, contrast=(1, 0), replicate_ids=rep, seed=sim_seed, **m_kwargs)
                    pv = res["pvalue"].to_numpy()
                    ev = evaluate(pv, res["padj"].to_numpy(), res["estimate"].to_numpy(),
                                  data, alpha)
                    err = None
                except Exception as exc:                      # keep the study running
                    pv, ev = np.full(len(null), np.nan), {}
                    err = f"{type(exc).__name__}: {exc}"
                    if verbose:
                        traceback.print_exc()
                rows.append({**base, "method": m_name, **_components(m_kwargs),
                             "seconds": time.perf_counter() - t0,
                             "error": err, **ev})
                null_p.setdefault((sc_name, m_name), []).append(pv[null])
                bar.step(f"{sc_name} {i + 1}/{n_sims}: {m_name[:30]}")

            if baseline:
                y_n, _ = normalise(Y.double(), "median_ratio")
                y_c, x_c, *_ = collapse_replicates(y_n, X.double(), b, rep)
                est, pv = welch_test(y_c.numpy(), x_c.numpy())
                ev = evaluate(pv, benjamini_hochberg(pv), est, data, alpha)
                rows.append({**base, "method": BASELINE, "normalisation": "median_ratio",
                             "missing": "baseline (Welch)", "variance": "Welch",
                             "seconds": np.nan, "error": None, **ev})
                null_p.setdefault((sc_name, BASELINE), []).append(pv[null])
                bar.step(f"{sc_name} {i + 1}/{n_sims}: baseline")

            if checkpoint:
                pd.DataFrame(rows).to_csv(checkpoint, index=False)

    bar.close()
    per_sim = pd.DataFrame(rows)
    summary = summarise(per_sim)
    null_p = {k: np.concatenate(v) for k, v in null_p.items()}
    return per_sim, summary, null_p


def _components(kwargs: dict) -> dict:
    """Method components from pipeline kwargs, with the pipeline's defaults."""
    return {"normalisation": kwargs.get("normalisation", "median_ratio"),
            "missing": kwargs.get("missing", "selection"),
            "variance": "moderated" if kwargs.get("moderate", False) else "plain"}


def summarise(per_sim: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD of every metric per scenario and method."""
    metrics = [m for m in METRICS if m in per_sim]
    return (per_sim.groupby(["scenario", "method"], sort=False)[metrics]
            .agg(["mean", "std"]).round(4))


def rank_methods(per_sim: pd.DataFrame, scenario: str, max_type1: float = 0.06,
                 max_fdp: float = 0.06) -> pd.DataFrame:
    """
    Methods for one scenario sorted by power, with a 'calibrated' flag: mean per-test type I
    error <= max_type1 and mean FDP <= max_fdp. Power of uncalibrated methods is not comparable.
    """
    d = per_sim[per_sim["scenario"] == scenario]
    cols = [c for c in ["type1_per_test", "fdp", "power", "lfc_bias", "lfc_rmse", "n_tested",
                        "seconds"] if c in d]
    t = d.groupby("method")[cols].mean()
    t["calibrated"] = (t["type1_per_test"] <= max_type1) & (t["fdp"] <= max_fdp)
    sort_col = "power" if "power" in t and t["power"].notna().any() else "type1_per_test"
    return t.sort_values(["calibrated", sort_col], ascending=[False, sort_col == "type1_per_test"])


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_null_qq(null_p: dict, scenario: str, methods=None, path: str = "null_pvalue_qq.png"):
    """QQ plot of null p-values (-log10) for one scenario; points on the diagonal = calibrated."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    for (sc, m), p in null_p.items():
        if sc != scenario or (methods is not None and m not in methods):
            continue
        p = np.sort(p[~np.isnan(p)])
        if p.size == 0:
            continue
        exp = (np.arange(1, p.size + 1) - 0.5) / p.size
        ax.plot(-np.log10(exp), -np.log10(p), ".", ms=2, label=m)
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("expected -log10(p)")
    ax.set_ylabel("observed -log10(p)")
    ax.set_title(f"Null p-values: {scenario}")
    ax.legend(fontsize=6, markerscale=4, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_tradeoff(per_sim: pd.DataFrame, scenario: str, alpha: float = 0.05,
                  path: str = "power_vs_type1.png"):
    """Power against per-test type I error for every method in one DE scenario."""
    import matplotlib.pyplot as plt
    d = per_sim[per_sim["scenario"] == scenario].groupby("method")[
        ["type1_per_test", "power"]].mean().dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d["type1_per_test"], d["power"], s=15)
    for m, row in d.iterrows():
        ax.annotate(m, (row["type1_per_test"], row["power"]), fontsize=5)
    ax.axvline(alpha, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("per-test type I error (log scale)")
    ax.set_ylabel("power (padj < 0.05)")
    ax.set_title(scenario)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
DEFAULT_SCENARIOS = {
    "no DE": dict(prop_de=0.0),
    "no DE | trend variance": dict(prop_de=0.0, var_dist="trend"),
    "DE 0.5": dict(prop_de=0.2, logfc=0.5),
    "DE 1": dict(prop_de=0.2, logfc=1.0),
    "DE 1 | lognormal variance": dict(prop_de=0.2, logfc=1.0, var_dist="lognormal"),
    "DE 1 | trend variance": dict(prop_de=0.2, logfc=1.0, var_dist="trend"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sims", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quick", action="store_true",
                    help="median_ratio only, 3 missing-value strategies, 3 simulations")
    ap.add_argument("--out", default="benchmark_results")
    args = ap.parse_args(argv)

    if args.quick:
        methods = method_grid(normalisations=("median_ratio",),
                              missing=("selection", "none", "min_prob"))
        n_sims = min(args.n_sims, 3)
    else:
        methods = method_grid()
        n_sims = args.n_sims
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(methods)} methods x {len(DEFAULT_SCENARIOS)} scenarios x {n_sims} simulations")

    per_sim, summary, null_p = run_study(
        DEFAULT_SCENARIOS, methods, n_sims=n_sims, seed=args.seed,
        checkpoint=os.path.join(args.out, "per_simulation_checkpoint.csv"))
    per_sim.to_csv(os.path.join(args.out, "per_simulation.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    for sc in DEFAULT_SCENARIOS:
        print(f"\n=== {sc} ===")
        print(rank_methods(per_sim, sc).round(4).to_string())
    try:
        for sc in DEFAULT_SCENARIOS:
            slug = sc.replace(" ", "_").replace("|", "").replace(".", "")
            plot_null_qq(null_p, sc, path=os.path.join(args.out, f"qq_{slug}.png"))
            if DEFAULT_SCENARIOS[sc].get("prop_de", 0.2) > 0:
                plot_tradeoff(per_sim, sc, path=os.path.join(args.out, f"tradeoff_{slug}.png"))
        print(f"\nPlots written to {args.out}/")
    except ImportError:
        print("matplotlib not installed; skipping plots")
    if per_sim["error"].notna().any():
        print("\nMethods with errors:")
        print(per_sim.loc[per_sim["error"].notna(), ["scenario", "method", "error"]]
              .drop_duplicates("method").to_string())


if __name__ == "__main__":
    main()
