"""
End-to-end differential abundance pipeline:
  1.  normalisation per run (median_ratio, total, quantile, none),
  1b. optional collapsing of technical replicates,
  2.  missing-value handling:
        "selection"  per-batch missingness parameters (fitted, then frozen) and the
                     selection-model regression with sigmoid missingness,
        "none"       missing values are left out (observed-only regression),
        imputation   zero, min_prob, mean, mean_noise, mean_group, knn, then regression on the
                     completed matrix (ordinary least squares),
  3.  Wald tests (optionally limma-style variance moderation) and Benjamini-Hochberg FDR.
With an imputation method and moderate=True the test is equivalent to limma on imputed data.
  4.  optional publication figures (figures_dir=...): PCA, missingness per sample, total
      intensity before/after normalisation, missingness curve (selection only), and a volcano
      plot and heatmap per contrast.
"""
import logging

import numpy as np
import torch

from .imputation import IMPUTATIONS, impute
from .metadata import contrast_vector
from .missingness import MissingParametersNR
from .model import DifferentialAbundanceNR
from .normalisation import normalise
from .replicates import collapse_replicates

MISSING_HANDLING = ("selection", "none") + IMPUTATIONS
_log = logging.getLogger("proteomix")


def n_pipeline_steps(n_contrasts: int) -> int:
    """Progress steps of one run_differential_abundance call (figures are added later)."""
    return 4 + n_contrasts


def run_differential_abundance(Y, X, batch_ids, contrast, replicate_ids=None,
                               collapse: str = "mean", min_present: int = 1,
                               normalisation: str = "median_ratio",
                               missing: str = "selection", log_scale: bool = True,
                               moderate: bool = False, trend: bool = True, robust: bool = False,
                               feature_names=None, include_sigma: bool = True,
                               beta_prior_var: float | None = 100.0, seed=None,
                               impute_kwargs: dict | None = None, verbose: bool = False,
                               min_valid: int | None = None, min_valid_per_group: int = 0,
                               min_groups: int = 1,
                               group_names=None, run_names=None, sample_names=None,
                               figures_dir: str | None = None,
                               figure_formats=("pdf", "png"), figure_dpi: int = 600,
                               font_size: float = 16, volcano_kwargs: dict | None = None,
                               heatmap_kwargs: dict | None = None,
                               heatmap_samples: str = "contrast", progress=None,
                               pca_factors: dict | None = None, pca_shape=None,
                               highlight=None, layer: str = ""):
    """
    Y: (N_runs, F) intensities (log2 if log_scale=True) with NaN for missing.
    X: (N_runs, p) design, batch_ids: (N_runs,).
    contrast: one contrast or a dict {label: contrast} to test several (results is then a dict
        {label: DataFrame}). A contrast is a (g1, g0) tuple of group indices or group names,
        a {group name: weight} dict, or a (p,) weight vector.
    replicate_ids: (N_runs,) biological-sample label per run; if given, technical replicates are
        collapsed after run-level normalisation. None = no collapsing.
    normalisation: "median_ratio", "total", "quantile" or "none".
    missing: "selection", "none", or an imputation method (see MISSING_HANDLING).
    moderate / trend / robust: empirical-Bayes variance moderation options for the test.
    seed: random seed for stochastic imputations (min_prob, mean_noise).
    Missingness filters (applied to biological samples, after collapsing, before imputation;
    filtered proteins appear in the results with NaN statistics and fitted=False):
        min_valid: minimum observed samples in total (default: number of groups + 2, the
            minimum for a residual variance).
        min_valid_per_group / min_groups: require at least min_valid_per_group observed
            samples in at least min_groups groups (default 0: off). E.g. 3 and 1 = "at least
            3 valid values in at least one group", a common proteomics filter.
        The selection model uses missing values as information, so keep filters lenient;
        the missingness parameters are always fitted on all proteins.
    group_names: names of the columns of X (one-hot designs) used in labels and legends.
    run_names: (N_runs,) run labels for run-level figures.
    sample_names: labels of the collapsed samples, in the order of sorted unique replicate_ids;
        default: the name of each sample's first run.
    figures_dir: if given, publication figures are written there (see figures.py) in
        figure_formats at figure_dpi; their paths are in model.figures.
    heatmap_samples: "contrast" shows only samples of groups in each contrast, "all" every sample.
    pca_factors / pca_shape / highlight: extra PCA colourings, marker shapes and highlighted
        samples (see figures.make_report_figures).
    progress: optional Progress bar, advanced at each step (see progress.py).
    layer: label used in log messages (e.g. "protein", "site").
    Returns (results DataFrame, fitted DifferentialAbundanceNR, frozen MissingParametersNR or
             None, log2 run offsets or None). The model also keeps the analysed matrix as
             model.y_input (before imputation), model.x_input, model.batch_input, model.sample_ids.
    """
    if missing not in MISSING_HANDLING:
        raise ValueError(f"unknown missing-value handling {missing!r}; "
                         f"choose from {MISSING_HANDLING}")
    from .progress import NullProgress
    progress = progress or NullProgress()
    tag = f"[{layer}] " if layer else ""
    X = X.double()
    batch_ids = batch_ids.long()
    _log.debug(f"{tag}input: {Y.shape[0]} runs x {Y.shape[1]} features, "
               f"{torch.isnan(Y).double().mean().item():.1%} missing")

    # 1. Normalise each run
    Y_norm, log_sf = normalise(Y.double(), method=normalisation, log_scale=log_scale)
    Y_norm_runs, X_runs = Y_norm, X
    if log_sf is not None:
        _log.debug(f"{tag}normalisation {normalisation}: log2 offsets "
                   f"{log_sf.min().item():+.3f} to {log_sf.max().item():+.3f}")
    progress.step(f"{tag}normalised ({normalisation})")

    # 1b. Collapse technical replicates
    sample_ids = None
    if replicate_ids is not None:
        Y_norm, X, batch_ids, sample_ids, _ = collapse_replicates(
            Y_norm, X, batch_ids, replicate_ids, method=collapse, min_present=min_present)
        _log.debug(f"{tag}collapsed {Y_norm_runs.shape[0]} runs into {Y_norm.shape[0]} samples "
                   f"({collapse}, min_present={min_present})")
    progress.step(f"{tag}replicates collapsed")

    # 2. Missing values
    miss = None
    if missing == "selection":
        num_batches = int(batch_ids.max().item()) + 1
        miss = MissingParametersNR(num_batches=num_batches).double()
        miss.fit(Y_norm, X, batch_ids, verbose=verbose)
        miss.requires_grad_(False)
        miss.eval()
        for b, (w_, r_) in enumerate(zip(miss.w.squeeze(1).tolist(), miss.r.squeeze(1).tolist())):
            _log.debug(f"{tag}missingness batch {b}: w = {w_:.3f}, r = {r_:.3f} "
                       f"({len(miss.ll_history) - 1} Newton iterations)")
        model = DifferentialAbundanceNR(miss.w.squeeze(1), miss.r.squeeze(1),
                                        include_sigma=include_sigma,
                                        beta_prior_var=beta_prior_var)
        y_fit = Y_norm
    else:
        model = DifferentialAbundanceNR(missingness="ignore", beta_prior_var=beta_prior_var)
        y_fit = Y_norm if missing == "none" else impute(Y_norm, missing, x=X, seed=seed,
                                                        **(impute_kwargs or {}))
        if missing != "none":
            _log.debug(f"{tag}imputed {int(torch.isnan(Y_norm).sum())} values ({missing})")
    progress.step(f"{tag}missing values: {missing}")
    keep = missingness_filter(Y_norm, X, min_valid_per_group, min_groups)
    model.fit(y_fit, X, batch_ids, verbose=verbose, min_obs=min_valid, keep=keep)
    model.keep_filter = keep
    _log.debug(f"{tag}model fitted: {int(model.fitted.sum())} of {len(model.fitted)} features "
               f"(filters: min_valid={min_valid}, min_valid_per_group={min_valid_per_group}, "
               f"min_groups={min_groups}; {int((~keep).sum())} removed by the per-group "
               f"filter); {len(model.ll_history) - 1} Newton iterations, "
               f"max |gradient| {np.nanmax(model.grad_norm.numpy()) if model.fitted.any() else 0:.2e}")
    progress.step(f"{tag}model fitted")
    model.y_input, model.x_input, model.batch_input = Y_norm, X, batch_ids
    model.sample_ids = sample_ids

    # 3. Tests + FDR
    p = X.shape[1]
    gnames = list(group_names) if group_names is not None else [f"Group {k}" for k in range(p)]
    contrasts = contrast if isinstance(contrast, dict) else {
        _contrast_label(contrast, gnames): contrast}
    results = {}
    for label, c in contrasts.items():
        results[label] = model.test(_resolve(c, gnames), feature_names=feature_names,
                                    moderate=moderate, trend=trend, robust=robust)
        n_sig = int((results[label]["padj"] < 0.05).sum())
        prior = getattr(model, "var_prior", None)
        _log.debug(f"{tag}contrast '{label}': {n_sig} at FDR 5%"
                   + (f"; moderation d0 = {prior['d0_global']:.1f}, "
                      f"{prior['n_outliers']} variance outliers" if moderate and prior else ""))
        progress.step(f"{tag}tested {label[:40]}")

    # 4. Figures
    model.figures = {}
    if figures_dir is not None:
        from .figures import make_report_figures
        n_runs = Y.shape[0]
        rnames = (np.asarray(run_names).astype(str) if run_names is not None
                  else np.array([f"R{i + 1}" for i in range(n_runs)]))
        if replicate_ids is not None and sample_names is not None:
            snames = np.asarray(sample_names).astype(str)
        elif replicate_ids is not None:
            inv = torch.unique(torch.as_tensor(replicate_ids), sorted=True,
                               return_inverse=True)[1].numpy()
            snames = np.array([rnames[np.where(inv == s)[0][0]] for s in range(inv.max() + 1)])
        else:
            snames = rnames
        model.figures = make_report_figures(
            figures_dir, Y.double(), Y_norm_runs, model, results,
            run_groups=_group_labels(X_runs, gnames), run_names=rnames,
            sample_groups=_group_labels(X, gnames), sample_names=snames,
            sample_batches=batch_ids.numpy(), miss=miss, log2_input=log_scale,
            formats=figure_formats, dpi=figure_dpi, font_size=font_size,
            volcano_kwargs=volcano_kwargs, heatmap_kwargs=heatmap_kwargs,
            contrast_vectors={k: _as_vector(c, gnames) for k, c in contrasts.items()},
            heatmap_samples=heatmap_samples, pca_factors=pca_factors, pca_shape=pca_shape,
            highlight=highlight, progress=progress)

    if not isinstance(contrast, dict):
        results = next(iter(results.values()))
    return results, model, miss, log_sf


def missingness_filter(y, X, min_valid_per_group: int = 0, min_groups: int = 1):
    """(F,) bool: at least min_valid_per_group observed samples in at least min_groups groups."""
    if min_valid_per_group <= 0:
        return torch.ones(y.shape[1], dtype=torch.bool)
    obs = (~torch.isnan(y)).double()
    _, grp = torch.unique(X, dim=0, return_inverse=True)
    G = torch.nn.functional.one_hot(grp, int(grp.max()) + 1).double()   # (N, groups)
    counts = G.T @ obs                                                  # (groups, F)
    return (counts >= min_valid_per_group).sum(0) >= min_groups


def _group_labels(X, names):
    """Group label per row: the active column for one-hot designs, else the unique row index."""
    Xn = X.detach().cpu().numpy()
    if np.all((Xn == 0) | (Xn == 1)) and np.all(Xn.sum(1) == 1):
        return np.array([names[k] for k in Xn.argmax(1)])
    _, idx = np.unique(Xn, axis=0, return_inverse=True)
    return np.array([f"Design {k + 1}" for k in np.ravel(idx)])


def _resolve(c, names):
    """Group-name tuples and {name: weight} dicts become weight vectors; index tuples pass."""
    if isinstance(c, dict) or (isinstance(c, tuple) and any(isinstance(g, str) for g in c)):
        return contrast_vector(c, names)
    return c


def _as_vector(c, names):
    if isinstance(c, tuple) and len(c) == 2 and not any(isinstance(g, str) for g in c):
        v = np.zeros(len(names))
        v[c[0]], v[c[1]] = 1.0, -1.0
        return v
    return np.asarray(_resolve(c, names), dtype=float)


def _contrast_label(c, names):
    if isinstance(c, tuple) and len(c) == 2:
        a, b = (g if isinstance(g, str) else names[g] for g in c)
        return f"{a} vs {b}"
    if isinstance(c, dict):
        c = contrast_vector(c, names)
    v = np.asarray(c, dtype=float)
    order = sorted((k for k in range(len(v)) if v[k] != 0), key=lambda k: -v[k])
    terms = [f"{'+' if v[k] > 0 else '−'} {'' if abs(v[k]) == 1 else f'{abs(v[k]):g}·'}{names[k]}"
             for k in order]
    return " ".join(terms).lstrip("+ ")
