"""
Outlier detection for samples and runs.

Per biological sample (after normalisation and collapsing technical replicates):
    missing_pct        % missing values                         (flag if high)
    group_correlation  median Pearson correlation with the other samples of its group
                       (pairwise-complete log2 values)          (flag if low)
    group_deviation    RMS log2 difference from the median profile of the other samples
                       of its group (leave-one-out, so the sample cannot mask itself)
                                                                (flag if high)
    median_shift       median of that difference: a large shift with good correlation
                       points to a loading / normalisation problem rather than bad data
    log2_size_factor   normalisation offset (mean over its runs) (flag if extreme)
Per run (only with technical replicates):
    replicate_correlation  median correlation with the other injections of the same sample

Each metric becomes a robust z-score. For the group-dependent metrics (missingness,
correlation, size factor) values are centred on their group's median, so a group that is
legitimately different (e.g. IgG vs IP) is not flagged as a whole; the scale is the median
absolute deviation (x 1.4826) of the metric over all samples, which stays stable with only
3-4 samples per group; minimum scales (MIN_SCALE) stop negligible differences from being
flagged in small studies. Group centring for the PCA distance also uses medians, so one bad
sample does not make its group-mates look shifted (group_deviation is computed on the log
scale, leave-one-out). A sample is flagged when a z-score passes
the threshold in the "bad" direction (default 3.5, the usual cut-off for robust z-scores).

Per run, agreement is measured against the group's typical profile (median over the other
samples' runs), so with two injections per sample the bad one can be told from the good one.

Flags are advisory: nothing is removed unless asked (CLI: --exclude or --remove-outliers).
"""
import numpy as np
import pandas as pd

MAD_SCALE = 1.4826


# Smallest scales used for the z-scores: differences below these are never "outlying",
# however tight the other samples are (protects small studies, where the MAD of a handful
# of values can be tiny).
MIN_SCALE = {"correlation": 0.005, "log_deviation": 0.05, "missing_pct": 1.5,
             "log2_size_factor": 0.1}


def _robust_z(values, groups=None, min_scale: float = 0.0):
    """(value - group median) / max(1.4826 * MAD of all values, min_scale)."""
    v = np.asarray(values, dtype=float)
    dev = v - np.nanmedian(v)
    if groups is not None:
        g = np.asarray(groups)
        for lev in np.unique(g):
            m = g == lev
            dev[m] = v[m] - np.nanmedian(v[m])
    scale = MAD_SCALE * np.nanmedian(np.abs(v - np.nanmedian(v)))
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(dev) or 1.0
    return dev / max(scale, min_scale)


def _corr_matrix(Y, min_periods=30):
    """Pairwise-complete Pearson correlations between rows of Y (samples x features)."""
    return pd.DataFrame(Y.T).corr(min_periods=min_periods).to_numpy()


def residual_pca(Y, groups, n_pcs: int = 3, max_missing: float = 0.2):
    """
    PCA of group-centred data. Features with at most max_missing missing values are used;
    remaining gaps are set to 0 (the group mean). Returns (scores S x k, explained %).
    """
    Y = np.asarray(Y, dtype=float)
    g = np.asarray(groups)
    R = np.full_like(Y, np.nan)
    import warnings
    for lev in np.unique(g):
        m = g == lev
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            R[m] = Y[m] - np.nanmedian(Y[m], axis=0)
    use = np.isnan(R).mean(0) <= max_missing
    R = np.nan_to_num(R[:, use], nan=0.0)
    R = R - R.mean(0)
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    k = int(min(n_pcs, len(S) - 1, R.shape[0] - 1))
    k = max(k, 1)
    scores = U[:, :k] * S[:k]
    return scores, (S[:k] ** 2 / np.sum(S ** 2) * 100), int(use.sum())


def sample_qc(y_samples, groups, sample_names, log_sf_samples=None, threshold: float = 3.5,
              n_pcs: int = 3) -> pd.DataFrame:
    """Per-sample QC table with metrics, robust z-scores and flags."""
    Y = np.asarray(y_samples, dtype=float)
    g = np.asarray(groups).astype(str)
    S = len(Y)
    missing = np.isnan(Y).mean(1) * 100

    C = _corr_matrix(Y)
    gc = np.full(S, np.nan)
    for i in range(S):
        mates = (g == g[i]) & (np.arange(S) != i)
        if mates.any():
            gc[i] = np.nanmedian(C[i, mates])

    import warnings
    dev = np.full(S, np.nan)
    shift = np.full(S, np.nan)
    for i in range(S):
        mates = (g == g[i]) & (np.arange(S) != i)
        if mates.sum() >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ref = np.nanmedian(Y[mates], axis=0)
            d = Y[i] - ref
            dev[i] = np.sqrt(np.nanmean(d ** 2))
            shift[i] = np.nanmedian(d)

    t = pd.DataFrame({"sample": list(sample_names), "group": g, "missing_pct": missing,
                      "group_correlation": gc, "group_deviation": dev,
                      "median_shift": shift})
    t["z_missing"] = _robust_z(missing, g, MIN_SCALE["missing_pct"])
    t["z_correlation"] = _robust_z(gc, g, MIN_SCALE["correlation"])
    t["z_distance"] = _robust_z(np.log(dev), g, MIN_SCALE["log_deviation"])
    flags = {"high missingness": t["z_missing"] > threshold,
             "low within-group correlation": t["z_correlation"] < -threshold,
             "far from its group": t["z_distance"] > threshold}
    if log_sf_samples is not None:
        t["log2_size_factor"] = np.asarray(log_sf_samples, dtype=float)
        t["z_size_factor"] = _robust_z(t["log2_size_factor"], g, MIN_SCALE["log2_size_factor"])
        flags["extreme size factor"] = t["z_size_factor"].abs() > threshold
    reasons = []
    for i in range(S):
        r = [k for k, f in flags.items() if f.iloc[i]]
        if "far from its group" in r and abs(shift[i]) > 0.1:
            r[r.index("far from its group")] = (f"far from its group (overall shift "
                                                f"{shift[i]:+.2f} log2: loading/normalisation?)")
        reasons.append(", ".join(r))
    t["flags"] = reasons
    t["outlier"] = t["flags"] != ""
    return t


def run_qc(y_runs, run_sample, run_names, run_groups=None, threshold: float = 3.5) -> pd.DataFrame:
    """
    Per run: correlation with its group's typical profile (median over runs of the other
    samples of the group), robust z-scored (group-centred, global MAD); and, for information,
    the correlation with the other injections of the same sample.
    """
    import warnings
    Y = np.asarray(y_runs, dtype=float)
    rs = np.asarray(run_sample).astype(str)
    rg = np.asarray(run_groups if run_groups is not None else ["all"] * len(Y)).astype(str)
    C = _corr_matrix(Y)
    n = len(Y)
    rep_c = np.full(n, np.nan)
    grp_c = np.full(n, np.nan)
    for i in range(n):
        mates = (rs == rs[i]) & (np.arange(n) != i)
        if mates.any():
            rep_c[i] = np.nanmedian(C[i, mates])
        others = (rg == rg[i]) & (rs != rs[i])
        if others.sum() >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ref = np.nanmedian(Y[others], axis=0)
            ok = ~np.isnan(Y[i]) & ~np.isnan(ref)
            if ok.sum() >= 30:
                grp_c[i] = np.corrcoef(Y[i, ok], ref[ok])[0, 1]
    t = pd.DataFrame({"run": list(run_names), "sample": rs, "group": rg,
                      "group_profile_correlation": grp_c, "replicate_correlation": rep_c,
                      "missing_pct": np.isnan(Y).mean(1) * 100})
    t["z_group_profile"] = _robust_z(grp_c, rg, MIN_SCALE["correlation"])
    t["outlier"] = np.nan_to_num(t["z_group_profile"].to_numpy(), nan=0.0) < -threshold
    return t


def run_quality_control(inputs, normalisation: str = "median_ratio", threshold: float = 3.5,
                        collapse: str = "mean"):
    """
    Normalises and collapses the inputs exactly as the pipeline does, then computes the
    sample and run QC tables. Returns (sample_table, run_table, y_samples, sample_groups).
    """
    from .normalisation import normalise
    from .pipeline import _group_labels
    from .replicates import collapse_replicates

    Yn, log_sf = normalise(inputs.Y.double(), method=normalisation)
    yc, xc, _, _, _ = collapse_replicates(Yn, inputs.X.double(), inputs.batch_ids,
                                          inputs.replicate_ids, method=collapse)
    groups = _group_labels(xc, inputs.group_names)
    sf_s = None
    if log_sf is not None:
        codes = inputs.replicate_ids.numpy()
        sf_s = np.array([log_sf.numpy()[codes == k].mean() for k in range(codes.max() + 1)])
    st = sample_qc(yc.numpy(), groups, inputs.sample_names, sf_s, threshold)
    run_names_sample = [inputs.sample_names[k] for k in inputs.replicate_ids.numpy()]
    rt = run_qc(Yn.numpy(), run_names_sample, inputs.run_names,
                _group_labels(inputs.X.double(), inputs.group_names), threshold)
    return st, rt, yc.numpy(), groups


def subset_inputs(inputs, keep_runs):
    """ProteomixInputs restricted to the runs where keep_runs is True (codes re-indexed)."""
    import torch
    from dataclasses import replace

    keep = np.asarray(keep_runs, dtype=bool)
    if keep.all():
        return inputs
    runs = inputs.runs[keep].reset_index(drop=True)
    rep_old = inputs.replicate_ids.numpy()[keep]
    names_old = np.asarray(inputs.sample_names)[rep_old]
    codes, levels = pd.factorize(pd.Series(names_old))
    b_old = inputs.batch_ids.numpy()[keep]
    b_codes, b_levels = pd.factorize(pd.Series(b_old), sort=True)
    X = inputs.X[torch.tensor(keep)]
    if (X.sum(0) == 0).any():
        empty = [g for g, n in zip(inputs.group_names, X.sum(0).tolist()) if n == 0]
        raise ValueError(f"excluding these runs would remove every sample of {empty}")
    return replace(inputs, Y=inputs.Y[torch.tensor(keep)], X=X,
                   batch_ids=torch.tensor(b_codes, dtype=torch.long),
                   replicate_ids=torch.tensor(codes, dtype=torch.long),
                   run_names=list(np.asarray(inputs.run_names)[keep]),
                   batch_names=[inputs.batch_names[i] for i in b_levels],
                   sample_names=list(levels), runs=runs)


def exclusion_mask(inputs, exclude) -> np.ndarray:
    """Runs to keep, given names to exclude (sample IDs remove all their runs)."""
    exclude = {str(e).strip() for e in exclude if str(e).strip()}
    run_names = np.asarray(inputs.run_names).astype(str)
    samples = np.asarray(inputs.sample_names)[inputs.replicate_ids.numpy()].astype(str)
    unknown = exclude - set(run_names) - set(samples)
    if unknown:
        raise ValueError(f"cannot exclude {sorted(unknown)}: not a run name or sample ID")
    return ~(np.isin(run_names, list(exclude)) | np.isin(samples, list(exclude)))
