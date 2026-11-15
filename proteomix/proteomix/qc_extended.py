"""
Extended quality control (complements qc.py).

General (protein or site table):
  1. identifications per run       proteins quantified per run; low runs flagged
  2. contaminants                  % of intensity from contaminant proteins (contam_) per run
  3. coefficients of variation     per protein within each group: biological (between
                                   samples) and technical (between injections of a sample)
  4. technical replicate pairs     correlation and median log2 difference of the injections
                                   of each sample; scatter plots of the least concordant pairs
  5. mean-variance relationship    pooled within-group SD against mean log2 intensity
  6. sample correlation            clustered correlation heatmap; samples that agree clearly
                                   better with another group than with their own are flagged
                                   (possible mislabels)
  7. run order                     median intensity, identifications and missingness against
                                   acquisition order; drift tested with Spearman correlation
                                   after removing group differences
  8. confounding                   conditions cross-tabulated with run-level factors (tray,
                                   batch, acquisition block); Cramer's V near 1 = confounded
IP experiments:
 15. bait recovery                 bait intensity per sample; IP samples with a missing or low
                                   bait flagged; IP / IgG enrichment of the bait
     IgG background                proteins quantified per IP / IgG sample; proteins detected
                                   in most IgG samples (frequent background binders)
Phospho experiments:
 16. enrichment efficiency         phosphopeptide share of peptide intensity / identifications
                                   per run (needs FragPipe's modified-peptide table)
     localisation quality          % of quantified sites passing the localisation threshold
     residue distribution          S / T / Y share of quantified sites per run

Flags use robust z-scores (group-centred where groups legitimately differ, global median
absolute deviation, minimum scales) as in qc.py; they are advisory.
"""
import logging
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .qc import _robust_z

_log = logging.getLogger("proteomix")


def _np(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


# --------------------------------------------------------------------------------------
# 1-2. identifications and contaminants
# --------------------------------------------------------------------------------------
def identifications_per_run(Y_runs, run_names, groups, threshold: float = 3.5) -> pd.DataFrame:
    Y = _np(Y_runs)
    n = (~np.isnan(Y)).sum(1)
    t = pd.DataFrame({"run": list(run_names), "group": list(groups), "n_quantified": n})
    t["z_identifications"] = _robust_z(np.log(np.maximum(n, 1)), groups, 0.02)
    t["low_identifications"] = t["z_identifications"] < -threshold
    return t


def contaminant_share(Y_runs, contaminants: pd.DataFrame | None, run_names, groups,
                      threshold: float = 3.5, max_pct: float = 10.0):
    """
    Per run: % of summed raw intensity from contaminant proteins, flagged when above max_pct
    or an outlier (robust z > threshold). Returns (run table, mean share per contaminant).
    """
    Y = _np(Y_runs)
    total_clean = np.nansum(np.exp2(Y), 1)
    t = pd.DataFrame({"run": list(run_names), "group": list(groups)})
    if contaminants is None or contaminants.empty:
        t["contaminant_pct"] = np.nan
        t["high_contaminants"] = False
        return t, pd.DataFrame(columns=["contaminant", "mean_pct", "max_pct"])
    C = contaminants[list(run_names)].to_numpy(float)
    csum = np.nansum(C, 0)
    pct = 100 * csum / (csum + total_clean)
    t["contaminant_pct"] = pct
    t["z_contaminants"] = _robust_z(pct, None, 1.0)
    t["high_contaminants"] = (pct > max_pct) | (t["z_contaminants"] > threshold)
    share = 100 * np.nan_to_num(C) / (csum + total_clean)[None, :]
    top = (pd.DataFrame({"contaminant": contaminants.index, "mean_pct": share.mean(1),
                         "max_pct": share.max(1)})
           .groupby("contaminant", as_index=False).sum()
           .sort_values("mean_pct", ascending=False).reset_index(drop=True))
    t["top_contaminant"] = [contaminants.index[np.nanargmax(np.nan_to_num(C[:, j]))]
                            if csum[j] > 0 else "" for j in range(len(t))]
    return t, top


# --------------------------------------------------------------------------------------
# 3-5. CVs, replicate pairs, mean-variance
# --------------------------------------------------------------------------------------
def coefficient_of_variation(Y_samples, sample_groups, Y_runs=None, run_sample=None,
                             min_values: int = 2) -> tuple:
    """
    CV (%) on the linear scale per protein: biological (samples of a group) and technical
    (injections of a sample, averaged over samples). Returns (long table, summary by group).
    """
    rows = []
    Ys = np.exp2(_np(Y_samples))
    g = np.asarray(sample_groups).astype(str)
    for lev in dict.fromkeys(g):
        V = Ys[g == lev]
        ok = (~np.isnan(V)).sum(0) >= min_values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cv = 100 * np.nanstd(V[:, ok], 0, ddof=1) / np.nanmean(V[:, ok], 0)
        rows.append(pd.DataFrame({"group": lev, "kind": "biological", "cv_pct": cv}))
    if Y_runs is not None and run_sample is not None:
        Yr = np.exp2(_np(Y_runs))
        rs = np.asarray(run_sample).astype(str)
        for s_ in np.unique(rs):
            V = Yr[rs == s_]
            if len(V) < 2:
                continue
            ok = (~np.isnan(V)).sum(0) >= 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cv = 100 * np.nanstd(V[:, ok], 0, ddof=1) / np.nanmean(V[:, ok], 0)
            rows.append(pd.DataFrame({"group": "technical", "kind": "technical",
                                      "cv_pct": cv}))
    long = pd.concat(rows, ignore_index=True)
    bio = long[long["kind"] == "biological"].groupby("group", sort=False)["cv_pct"]
    summary = pd.DataFrame({"median_cv_pct": bio.median(), "n_proteins": bio.size()})
    tech = long[long["kind"] == "technical"]
    if len(tech):
        summary.loc["technical (all samples)", ["median_cv_pct", "n_proteins"]] = \
            [tech["cv_pct"].median(), len(tech)]
    return long, summary


def replicate_pairs(Y_runs, run_sample, run_names) -> pd.DataFrame:
    """Every pair of injections of the same sample: Pearson r, median and SD of differences."""
    Y = _np(Y_runs)
    rs = np.asarray(run_sample).astype(str)
    names = np.asarray(run_names)
    rows = []
    for s_ in dict.fromkeys(rs):
        idx = np.flatnonzero(rs == s_)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                ok = ~np.isnan(Y[i]) & ~np.isnan(Y[j])
                if ok.sum() < 10:
                    continue
                d = Y[i, ok] - Y[j, ok]
                rows.append({"sample": s_, "run_a": names[i], "run_b": names[j],
                             "n_shared": int(ok.sum()),
                             "pearson_r": float(np.corrcoef(Y[i, ok], Y[j, ok])[0, 1]),
                             "median_log2_diff": float(np.median(d)),
                             "sd_log2_diff": float(np.std(d, ddof=1))})
    return pd.DataFrame(rows)


def mean_variance(Y_samples, sample_groups) -> pd.DataFrame:
    """Per protein: mean log2 intensity and pooled within-group SD (groups with >= 2 values)."""
    Y = _np(Y_samples)
    g = np.asarray(sample_groups).astype(str)
    ss = np.zeros(Y.shape[1])
    df = np.zeros(Y.shape[1])
    for lev in np.unique(g):
        V = Y[g == lev]
        n = (~np.isnan(V)).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(V, 0)
        dev = np.nansum((V - m) ** 2, 0)
        use = n >= 2
        ss[use] += dev[use]
        df[use] += n[use] - 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return pd.DataFrame({"mean_log2": np.nanmean(Y, 0),
                             "pooled_sd": np.where(df > 0, np.sqrt(ss / np.maximum(df, 1)),
                                                   np.nan),
                             "df": df})


# --------------------------------------------------------------------------------------
# 6. sample correlation
# --------------------------------------------------------------------------------------
def sample_correlation(Y_samples, sample_groups, sample_names, min_periods: int = 30,
                       threshold: float = 3.5):
    """
    Returns (correlation DataFrame, clustering order, per-sample table).
    A sample is flagged as a possible mislabel when it agrees clearly better with another
    group than with its own: the gap between its median correlation with the best other
    group and with its own group is an outlier among all samples (robust z > threshold).
    Groups that barely differ give small gaps and no flags.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    Y = _np(Y_samples)
    names = [str(s_) for s_ in sample_names]
    C = pd.DataFrame(Y.T).corr(min_periods=min_periods).to_numpy()
    C = np.where(np.isnan(C), 0.0, C)
    np.fill_diagonal(C, 1.0)
    D = np.clip(1 - C, 0, 2)
    np.fill_diagonal(D, 0)
    order = leaves_list(linkage(squareform(D, checks=False), "average"))
    g = np.asarray(sample_groups).astype(str)
    levels = list(dict.fromkeys(g))
    rows = []
    for i in range(len(names)):
        others = np.arange(len(names)) != i
        j = np.flatnonzero(others)[np.argmax(C[i, others])]
        med = {lev: float(np.median(C[i, (g == lev) & others]))
               for lev in levels if ((g == lev) & others).any()}
        own = med.get(g[i], np.nan)
        other = {k: v for k, v in med.items() if k != g[i]}
        best = max(other, key=other.get) if other else None
        rows.append({"sample": names[i], "group": g[i], "nearest_sample": names[j],
                     "nearest_group": g[j], "r_nearest": C[i, j], "median_r_own_group": own,
                     "best_other_group": best,
                     "median_r_best_other": other.get(best, np.nan) if best else np.nan})
    nn = pd.DataFrame(rows)
    nn["gap"] = nn["median_r_best_other"] - nn["median_r_own_group"]
    nn["z_gap"] = _robust_z(nn["gap"].to_numpy(float), None, 0.005)
    nn["possible_mislabel"] = (nn["gap"] > 0) & (nn["z_gap"] > threshold)
    return pd.DataFrame(C, index=names, columns=names), order, nn


# --------------------------------------------------------------------------------------
# 7-8. run order and confounding
# --------------------------------------------------------------------------------------
def run_order_effects(Y_runs_raw, runs: pd.DataFrame, groups, order_col: str = "acquisition"):
    """
    Per run metrics against acquisition order (file order when unavailable). Drift is tested
    with Spearman's correlation between order and each metric after subtracting the group
    median (so differences between groups acquired in blocks are not called drift).
    Returns (run table, test table).
    """
    Y = _np(Y_runs_raw)
    order = (runs[order_col].to_numpy(float) if order_col in runs and runs[order_col].notna().all()
             else np.arange(len(runs), dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t = pd.DataFrame({"run": runs["run"].to_numpy(), "group": list(groups),
                          "order": order, "median_log2": np.nanmedian(Y, 1),
                          "n_quantified": (~np.isnan(Y)).sum(1),
                          "missing_pct": 100 * np.isnan(Y).mean(1)})
    tests = []
    g = np.asarray(groups).astype(str)
    for metric in ("median_log2", "n_quantified", "missing_pct"):
        v = t[metric].to_numpy(float)
        res = v.copy()
        for lev in np.unique(g):
            m = g == lev
            res[m] = v[m] - np.median(v[m])
        rho, p = stats.spearmanr(order, res) if np.std(res) > 0 else (np.nan, 1.0)
        rho_raw, p_raw = stats.spearmanr(order, v) if np.std(v) > 0 else (np.nan, 1.0)
        tests.append({"metric": metric, "spearman_rho_within_group": rho,
                      "pvalue_within_group": p, "spearman_rho_raw": rho_raw,
                      "pvalue_raw": p_raw})
    tests = pd.DataFrame(tests)
    tests["drift"] = tests["pvalue_within_group"] < 0.01
    return t, tests


def confounding_check(runs: pd.DataFrame, condition_col: str = "condition", factors=None,
                      n_blocks: int = 4, order_col: str = "acquisition") -> tuple:
    """
    Cramer's V between the condition and each run-level factor (plus acquisition blocks:
    the acquisition order cut into n_blocks equal parts). V = 1: every level of the factor
    contains a single condition, so their effects cannot be separated.
    Returns (table, {factor: crosstab}).
    """
    t = runs.copy()
    if order_col in t and t[order_col].notna().all() and len(t) >= 2 * n_blocks:
        t["acquisition_block"] = pd.qcut(t[order_col].rank(method="first"), n_blocks,
                                         labels=[f"block {i + 1}" for i in range(n_blocks)])
    if factors is None:
        skip = {condition_col, "run", "sample", "sample_id", "sample_name", "acquisition",
                "well", "missing_pct", "injection", "sites_quantified", "sites_localised",
                "localised_pct", "pulldown", "treatment", "time_min"}
        factors = [c for c in t.columns if c not in skip
                   and 1 < t[c].nunique() < len(t) and t[c].dtype == object
                   or c == "acquisition_block"]
    rows, tabs = [], {}
    for f in factors:
        if f not in t or t[f].nunique() < 2:
            continue
        ct = pd.crosstab(t[condition_col], t[f])
        chi2, p, _, _ = stats.chi2_contingency(ct, correction=False)
        k = min(ct.shape) - 1
        v = float(np.sqrt(chi2 / (ct.to_numpy().sum() * k))) if k > 0 else np.nan
        tabs[f] = ct
        level_pure = bool(((ct > 0).sum(0) == 1).all())     # one condition per level
        cond_single = bool(((ct > 0).sum(1) == 1).all())    # each condition on one level
        rows.append({"factor": f, "levels": ct.shape[1], "cramers_v": v, "pvalue": p,
                     "each_level_single_condition": level_pure,
                     "each_condition_single_level": cond_single,
                     "confounded": level_pure or cond_single or v > 0.9})
    return pd.DataFrame(rows), tabs


# --------------------------------------------------------------------------------------
# 15. IP: bait recovery and IgG background
# --------------------------------------------------------------------------------------
def bait_recovery(Y_samples, genes, bait: str, sample_names, sample_groups, pulldown,
                  ip_label: str = "IP", control_label: str = "IgG", threshold: float = 3.5):
    """
    Bait log2 intensity per sample. IP samples where the bait is missing, or low compared
    with the other IP samples (robust z < -threshold), are flagged. Returns (table, summary).
    """
    genes = [str(x).upper() for x in genes]
    b = str(bait).upper()
    idx = [i for i, g_ in enumerate(genes) if g_ == b]
    if not idx:
        raise ValueError(f"bait {bait!r} not found among the quantified proteins")
    Y = _np(Y_samples)
    v = np.nanmax(Y[:, idx], 1) if len(idx) > 1 else Y[:, idx[0]]
    pd_ = np.asarray(pulldown).astype(str)
    t = pd.DataFrame({"sample": list(sample_names), "group": list(sample_groups),
                      "pulldown": pd_, "bait_log2": v})
    ip = pd_ == ip_label
    z = np.full(len(v), np.nan)
    ok = ip & ~np.isnan(v)
    if ok.sum() >= 3:
        z[ok] = _robust_z(v[ok], None, 0.25)
    t["z_bait_within_ip"] = z
    t["failed_pulldown"] = ip & (np.isnan(v) | (np.nan_to_num(z) < -threshold))
    ctrl = pd_ == control_label
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        summary = {"bait": bait, "ip_detected": int((ip & ~np.isnan(v)).sum()),
                   "ip_samples": int(ip.sum()),
                   "control_detected": int((ctrl & ~np.isnan(v)).sum()),
                   "control_samples": int(ctrl.sum()),
                   "median_ip_log2": float(np.nanmedian(v[ip])) if ip.any() else np.nan,
                   "median_control_log2": float(np.nanmedian(v[ctrl])) if (ctrl & ~np.isnan(v)).any()
                   else np.nan}
    summary["ip_vs_control_log2"] = summary["median_ip_log2"] - summary["median_control_log2"]
    return t, summary


def igg_background(Y_samples, genes, sample_names, pulldown, ip_label: str = "IP",
                   control_label: str = "IgG", min_fraction: float = 0.75):
    """
    Proteins quantified per sample (IP vs IgG) and the proteins detected in at least
    min_fraction of IgG samples (frequent background), with their IP / IgG log2 ratio.
    Returns (per-sample table, background table).
    """
    Y = _np(Y_samples)
    pd_ = np.asarray(pulldown).astype(str)
    per = pd.DataFrame({"sample": list(sample_names), "pulldown": pd_,
                        "n_quantified": (~np.isnan(Y)).sum(1)})
    ctrl, ip = pd_ == control_label, pd_ == ip_label
    if not ctrl.any():
        return per, pd.DataFrame()
    det = (~np.isnan(Y[ctrl])).mean(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mc = np.nanmedian(Y[ctrl], 0)
        mi = np.nanmedian(Y[ip], 0) if ip.any() else np.full(Y.shape[1], np.nan)
    bg = pd.DataFrame({"gene": list(genes), "igg_detection": det, "igg_median_log2": mc,
                       "ip_median_log2": mi, "ip_vs_igg_log2": mi - mc})
    bg = bg[bg["igg_detection"] >= min_fraction].sort_values("igg_median_log2",
                                                             ascending=False)
    return per, bg.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 16. phospho: residues, localisation, enrichment efficiency
# --------------------------------------------------------------------------------------
def site_residues(Y_runs, feature_names, run_names) -> pd.DataFrame:
    """S / T / Y share of the sites quantified in each run (residue from the site name)."""
    res = np.array([(re.search(r"_([STY])\d+(?:_|$)", str(f)) or [None, "other"])[1]
                    for f in feature_names])
    Y = _np(Y_runs)
    rows = []
    for i, r in enumerate(run_names):
        q = ~np.isnan(Y[i])
        n = q.sum()
        row = {"run": r, "sites_quantified": int(n)}
        for aa in ("S", "T", "Y", "other"):
            row[f"{aa}_pct"] = 100 * float((q & (res == aa)).sum()) / max(n, 1)
        rows.append(row)
    return pd.DataFrame(rows)


PHOSPHO_PATTERN = r"79\.96|Phospho|[STY]\[(?:167|181|243)"


def enrichment_efficiency(peptide_path: str, run_names, quantity_suffix: str = " Intensity"):
    """
    Phosphopeptide share per run from a FragPipe peptide-level table (e.g.
    combined_modified_peptide.tsv): % of peptide intensity and of quantified peptides that
    carry a phosphorylation. Phosphopeptides are recognised in the modified-sequence /
    modification column (79.966 Da, "Phospho", or S/T/Y with +80 Da residue masses).
    """
    sep = "\t" if open(peptide_path, encoding="utf-8-sig").readline().count("\t") else ","
    df = pd.read_csv(peptide_path, sep=sep, low_memory=False)
    mod_cols = [c for c in ("Modified Sequence", "Modified Peptide", "Assigned Modifications",
                            "Modifications") if c in df]
    if not mod_cols:
        raise ValueError("no modification column (e.g. 'Modified Sequence') in the peptide table")
    text = df[mod_cols].astype(str).agg(" ".join, axis=1)
    phos = text.str.contains(PHOSPHO_PATTERN, regex=True).to_numpy()
    rows = []
    for r in run_names:
        col = f"{r}{quantity_suffix}"
        if col not in df:
            rows.append({"run": r, "phospho_intensity_pct": np.nan,
                         "phospho_peptides_pct": np.nan})
            continue
        v = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
        q = v > 0
        rows.append({"run": r,
                     "phospho_intensity_pct": 100 * v[phos].sum() / max(v.sum(), 1e-300),
                     "phospho_peptides_pct": 100 * (q & phos).sum() / max(q.sum(), 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def run_extended_qc(inputs, layer: str, out_dir: str, normalisation: str = "median_ratio",
                    threshold: float = 3.5, experiment: str = "proteome",
                    bait: str | None = None, peptide_path: str | None = None,
                    genes=None, figures: bool = True, formats=("pdf", "png"),
                    dpi: int = 600, font_size: float = 16) -> tuple:
    """
    Runs every applicable check, writes <layer>_*.csv (and figures) to out_dir and returns
    (dict of tables, list of warning messages for the summary).
    """
    import matplotlib.pyplot as plt
    from . import qc_figures as qf
    from .figures import save_figure
    from .normalisation import normalise
    from .pipeline import _group_labels
    from .replicates import collapse_replicates

    os.makedirs(out_dir, exist_ok=True)
    msgs, tabs = [], {}
    Yraw = inputs.Y.double()
    Yn, _ = normalise(Yraw, method=normalisation)
    Ys, Xs, _, _, _ = collapse_replicates(Yn, inputs.X.double(), inputs.batch_ids,
                                          inputs.replicate_ids)
    run_groups = _group_labels(inputs.X.double(), inputs.group_names)
    sample_groups = _group_labels(Xs, inputs.group_names)
    run_sample = np.asarray(inputs.sample_names)[inputs.replicate_ids.numpy()]
    runs = inputs.runs.reset_index(drop=True)
    names = list(inputs.run_names)
    genes = genes if genes is not None else list(inputs.feature_names)

    def save(fig, name):
        if figures:
            save_figure(fig, os.path.join(out_dir, f"{layer}_{name}"), formats, dpi)
        plt.close(fig)

    def write(df, name, index=False):
        df.to_csv(os.path.join(out_dir, f"{layer}_{name}.csv"), index=index)
        tabs[name] = df

    ids = identifications_per_run(Yraw, names, run_groups, threshold)
    cont, top_cont = contaminant_share(Yraw, inputs.extra.get("contaminants"), names,
                                       run_groups, threshold)
    runtab = ids.merge(cont, on=["run", "group"])
    write(runtab, "identifications_contaminants")
    write(top_cont, "contaminant_proteins")
    for _, r in runtab[runtab["low_identifications"]].iterrows():
        msgs.append(f"run {r['run']}: only {r['n_quantified']} features quantified "
                    f"(z = {r['z_identifications']:.1f})")
    for _, r in runtab[runtab["high_contaminants"]].iterrows():
        msgs.append(f"run {r['run']}: {r['contaminant_pct']:.1f}% contaminant intensity "
                    f"(mostly {r['top_contaminant']})")

    cv_long, cv_sum = coefficient_of_variation(Ys, sample_groups, Yn, run_sample)
    write(cv_sum.reset_index().rename(columns={"index": "group"}), "cv_summary")
    pairs = replicate_pairs(Yn, run_sample, names)
    if len(pairs):
        write(pairs, "replicate_pairs")
    mv = mean_variance(Ys, sample_groups)
    corr, order, nn = sample_correlation(Ys, sample_groups, inputs.sample_names,
                                         threshold=threshold)
    write(corr, "sample_correlation", index=True)
    write(nn, "nearest_samples")
    for _, r in nn[nn["possible_mislabel"]].iterrows():
        msgs.append(f"sample {r['sample']} ({r['group']}) agrees better with "
                    f"{r['best_other_group']} (median r {r['median_r_best_other']:.3f} vs "
                    f"{r['median_r_own_group']:.3f} with its own group): possible mislabel")
    order_tab, drift = run_order_effects(Yraw, runs, run_groups)
    write(order_tab, "run_order")
    write(drift, "run_order_tests")
    for _, r in drift[drift["drift"]].iterrows():
        msgs.append(f"drift with acquisition order in {r['metric']} (Spearman rho "
                    f"{r['spearman_rho_within_group']:+.2f} within groups, "
                    f"p = {r['pvalue_within_group']:.2g})")
    conf, xtabs = confounding_check(runs.assign(condition=run_groups))
    if len(conf):
        write(conf, "confounding")
        for _, r in conf[conf["confounded"]].iterrows():
            if r["each_condition_single_level"] and not r["each_level_single_condition"]:
                msgs.append(f"each condition was run on a single {r['factor']} level: "
                            f"comparisons between conditions on different levels are "
                            f"confounded with {r['factor']} (Cramer's V {r['cramers_v']:.2f})")
            else:
                msgs.append(f"condition is confounded with {r['factor']} (Cramer's V "
                            f"{r['cramers_v']:.2f}): their effects cannot be separated")

    if figures:
        save(qf.plot_identifications_contaminants(runtab, font_size=font_size),
             "identifications_contaminants")
        save(qf.plot_cv(cv_long, font_size=font_size), "cv")
        if len(pairs):
            save(qf.plot_replicate_pairs(Yn, pairs, names, font_size=font_size),
                 "replicate_pairs")
        save(qf.plot_mean_variance(mv, font_size=font_size), "mean_variance")
        save(qf.plot_sample_correlation(corr, order, sample_groups, font_size=font_size),
             "sample_correlation")
        save(qf.plot_run_order(order_tab, drift, font_size=font_size), "run_order")
        if xtabs:
            save(qf.plot_confounding(xtabs, conf, font_size=font_size), "confounding")

    st = inputs.sample_table()
    if experiment == "ip" and "pulldown" in st:
        per, bg = igg_background(Ys, genes, inputs.sample_names, st["pulldown"].tolist())
        write(per, "pulldown_identifications")
        write(bg, "igg_background")
        if bait:
            bt, bsum = bait_recovery(Ys, genes, bait, inputs.sample_names, sample_groups,
                                     st["pulldown"].tolist(), threshold=threshold)
            write(bt, "bait_recovery")
            msgs.append(f"bait {bait}: detected in {bsum['ip_detected']}/{bsum['ip_samples']} IP "
                        f"and {bsum['control_detected']}/{bsum['control_samples']} IgG samples; "
                        f"IP vs IgG {bsum['ip_vs_control_log2']:+.1f} log2")
            for _, r in bt[bt["failed_pulldown"]].iterrows():
                msgs.append(f"sample {r['sample']}: bait missing or low; pull-down may have failed")
            if figures:
                save(qf.plot_bait(bt, bait, font_size=font_size), "bait_recovery")
        if figures:
            save(qf.plot_pulldown_counts(per, font_size=font_size), "pulldown_identifications")

    if experiment == "phospho":
        res = site_residues(Yraw, inputs.feature_names, names)
        if "localised_pct" in runs:
            res = res.merge(runs[["run", "sites_quantified", "sites_localised",
                                  "localised_pct"]].rename(columns={"sites_quantified":
                                                                    "sites_before_filter"}),
                            on="run", how="left")
        if peptide_path:
            res = res.merge(enrichment_efficiency(peptide_path, names), on="run", how="left")
        res.insert(1, "group", run_groups)
        write(res, "phospho_qc")
        for col, lab in (("phospho_intensity_pct", "phosphopeptide intensity"),
                         ("localised_pct", "localised sites")):
            if col in res and res[col].notna().sum() >= 4:
                z = _robust_z(res[col].to_numpy(float), None, 2.0)
                for i in np.flatnonzero(z < -threshold):
                    msgs.append(f"run {res.loc[i, 'run']}: low {lab} "
                                f"({res.loc[i, col]:.1f}%)")
        if figures:
            save(qf.plot_phospho_qc(res, font_size=font_size), "phospho_qc")
    return tabs, msgs
