"""
Plots of benchmarking results by method component.

Each method is "normalisation | missing | variance". plot_error_rates_by_component shows the
type I and type II error of each level of each component, averaged over the levels of the other
two components (marginal effect), with one colour per scenario:

    rows    : type I error (per test, p < alpha), type II error (1 - power at padj < alpha)
    columns : normalisation, missing-value handling, variance (plain / moderated)

Error bars come from the variation across simulations, so per_simulation.csv is the best input.
A summary table also works but only gives approximate error bars (see table_from_summary).
"""
import numpy as np
import pandas as pd
from scipy import stats

COMPONENTS = {"normalisation": "Normalisation", "missing": "Missing-value handling",
              "variance": "Variance"}
METRIC_LABELS = {"type1_per_test": "Type I error (p < α)", "type2": "Type II error (1 − power)"}


def _split_method(df):
    """Component columns: used as-is if present (run_study adds them), else parsed from names
    of the form "normalisation | missing | variance" (as made by method_grid)."""
    if not {"normalisation", "missing", "variance"} <= set(df.columns):
        parts = df["method"].str.split(r" \| ", n=2, expand=True).reindex(columns=[0, 1, 2])
        if parts[2].isna().all():
            raise ValueError("No normalisation/missing/variance columns and method names are "
                             "not 'normalisation | missing | variance'; use per_simulation "
                             "results from run_study, which carry the columns.")
        df = df.assign(normalisation=parts[0], missing=parts[1], variance=parts[2])
    return df[~df["method"].str.startswith("baseline")].dropna(subset=["variance"])


def table_from_per_sim(per_sim: pd.DataFrame, metrics=tuple(METRIC_LABELS), err: str = "ci95"):
    """
    Marginal means and error bars from per-simulation results.
    For each component level, the metric is first averaged over the other components' levels
    within each simulation; mean and spread are then taken across simulations.
    err: "sd" (across simulations), "se", or "ci95" (t quantile with n_sims - 1 df times se).
    Returns long table: scenario, component, level, metric, mean, err, n_sims.
    """
    df = _split_method(per_sim)
    rows = []
    for comp in COMPONENTS:
        for metric in metrics:
            per = (df.groupby(["scenario", "sim", comp], sort=False)[metric].mean()
                   .dropna().reset_index())
            g = per.groupby(["scenario", comp], sort=False)[metric]
            t = g.agg(["mean", "std", "count"]).reset_index()
            sd = t["std"].fillna(0.0)
            se = sd / np.sqrt(t["count"])
            tq = stats.t.ppf(0.975, np.maximum(t["count"] - 1, 1))
            t["err"] = {"sd": sd, "se": se, "ci95": tq * se}[err]
            rows.append(t.rename(columns={comp: "level", "count": "n_sims"})
                        .assign(component=comp, metric=metric)
                        [["scenario", "component", "level", "metric", "mean", "err", "n_sims"]])
    return pd.concat(rows, ignore_index=True)


def table_from_summary(summary: pd.DataFrame, n_sims: int, metrics=tuple(METRIC_LABELS),
                       err: str = "ci95"):
    """
    Same table from the summary (MultiIndex columns metric x {mean, std}, index scenario x method).
    The SD of a marginal mean is not recoverable from per-method SDs, so the average of the
    per-method SDs is used: usually conservative (wider bars) because methods are positively
    correlated across simulations. Load a saved summary with
        pd.read_csv("summary.csv", header=[0, 1], index_col=[0, 1])
    """
    rows = []
    s = summary.reset_index()
    s.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in s.columns]
    s = _split_method(s)
    for comp in COMPONENTS:
        for metric in metrics:
            t = (s.groupby(["scenario", comp], sort=False)
                 .agg(mean=(f"{metric}_mean", "mean"), sd=(f"{metric}_std", "mean"))
                 .dropna(subset=["mean"]).reset_index())
            sd = t["sd"].fillna(0.0)
            se = sd / np.sqrt(n_sims)
            tq = stats.t.ppf(0.975, max(n_sims - 1, 1))
            t["err"] = {"sd": sd, "se": se, "ci95": tq * se}[err]
            rows.append(t.rename(columns={comp: "level"})
                        .assign(component=comp, metric=metric, n_sims=n_sims)
                        [["scenario", "component", "level", "metric", "mean", "err", "n_sims"]])
    return pd.concat(rows, ignore_index=True)


def plot_error_rates_by_component(results: pd.DataFrame, alpha: float = 0.05,
                                  err: str = "ci95", n_sims: int | None = None,
                                  exclude: dict | None = None, path: str | None = None):
    """
    results: per_simulation DataFrame (preferred) or the summary DataFrame (then give n_sims).
    exclude: drop component levels before averaging, e.g. {"missing": ["mean_group", "zero"]},
        so that one badly behaved level does not shift the marginal means of the others.
    path: save the figure there if given. Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    is_summary = isinstance(results.columns, pd.MultiIndex)
    if exclude:
        df = results.reset_index() if is_summary else results
        if is_summary:
            df.columns = pd.MultiIndex.from_tuples(
                [c if c[1] else (c[0], "") for c in df.columns])
            method = df[("method", "")]
        else:
            method = df["method"]
        if not is_summary and set(COMPONENTS) <= set(df.columns):
            comp_cols = {c: df[c] for c in COMPONENTS}
        else:
            parts = method.str.split(r" \| ", n=2, expand=True).reindex(columns=[0, 1, 2])
            comp_cols = {c: parts[k] for k, c in enumerate(COMPONENTS)}
        keep = pd.Series(True, index=df.index)
        for comp, levels in exclude.items():
            keep &= ~comp_cols[comp].isin(levels)
        df = df[keep.to_numpy()]
        results = df.set_index([("scenario", ""), ("method", "")]) if is_summary else df
        if is_summary:
            results.index.names = ["scenario", "method"]

    if is_summary:
        if n_sims is None:
            raise ValueError("n_sims is required when plotting from the summary table")
        tab = table_from_summary(results, n_sims=n_sims, err=err)
    else:
        tab = table_from_per_sim(results, err=err)

    scenarios = list(dict.fromkeys(tab["scenario"]))
    colours = plt.cm.tab10(np.linspace(0, 1, 10))[: len(scenarios)]
    metrics = list(METRIC_LABELS)
    fig, axes = plt.subplots(len(metrics), len(COMPONENTS), figsize=(16, 8.5),
                             gridspec_kw={"width_ratios": [4, 8, 2.5]}, squeeze=False)

    for i, metric in enumerate(metrics):
        for j, comp in enumerate(COMPONENTS):
            ax = axes[i, j]
            sub = tab[(tab["metric"] == metric) & (tab["component"] == comp)]
            levels = list(dict.fromkeys(sub["level"]))
            scs = [sc for sc in scenarios if (sub["scenario"] == sc).any()]
            width = 0.8 / max(len(scs), 1)
            for k, sc in enumerate(scenarios):
                d = sub[sub["scenario"] == sc].set_index("level").reindex(levels)
                if d["mean"].isna().all():
                    continue
                pos = np.arange(len(levels)) - 0.4 + width * (scs.index(sc) + 0.5)
                ax.errorbar(pos, d["mean"], yerr=d["err"], fmt="o", ms=4, capsize=2,
                            color=colours[k], label=sc)
            if metric == "type1_per_test":
                ax.axhline(alpha, color="k", ls="--", lw=1)
            ax.set_xticks(np.arange(len(levels)))
            ax.set_xticklabels(levels, rotation=30, ha="right")
            ax.grid(axis="y", alpha=0.3)
            if i == 0:
                ax.set_title(COMPONENTS[comp])
            if j == 0:
                ax.set_ylabel(METRIC_LABELS[metric])

    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=min(len(handles), 6),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    bar = {"sd": "± 1 SD", "se": "± 1 SE", "ci95": "95% CI"}[err]
    note = f"; excluded: {exclude}" if exclude else ""
    fig.suptitle(f"Error rates by method component (mean over other components, {bar} "
                 f"across simulations{note})", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
