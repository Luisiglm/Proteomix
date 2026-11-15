"""
Protein-adjusted PTM (e.g. phosphosite) analysis with matched whole-proteome samples.

Paired model (default). For site f of protein j and biological sample s:

    log2 site[s, f] = log2 protein[s, j] + x_s beta_f + eps

The protein's log2 abundance in the same sample enters as a known offset (coefficient 1), so
beta_f is the change in the site / protein ratio (stoichiometry), and the pairing cancels
sample-to-sample variation shared by a protein and its sites. Missingness is evaluated on
the site's own intensity scale (offset + x beta), because detection depends on how much
modified peptide reaches the instrument. Where the protein is missing in a sample, the
offset is the protein model's fitted value for that sample's group (x_s beta_protein), which
already accounts for abundance-dependent missingness; the number of such fills is reported.
Sites whose protein is not quantified (or not fitted) get unadjusted results only.

combine_independent gives the alternative ratio-of-contrasts test (MSstatsPTM-style):
estimate_site - estimate_protein with SE sqrt(se_s^2 + se_p^2) and Welch-Satterthwaite df.
It assumes the two estimates are independent; with matched samples their positive
covariance makes it conservative. It is provided for benchmarking.
"""
import os
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy import stats

from .fragpipe import DEFAULT_RUN_PATTERN, FragPipeData, _info, parse_run_names
from .model import DifferentialAbundanceNR
from .pipeline import _resolve, run_differential_abundance
from .testing import benjamini_hochberg


# --------------------------------------------------------------------------------------
# Reading site tables
# --------------------------------------------------------------------------------------
def accession(s) -> str:
    """'sp|P12345|PD1_HUMAN;sp|...' -> 'P12345'; plain IDs pass through (first of a group)."""
    s = str(s).split(";")[0].strip()
    parts = s.split("|")
    return parts[1] if len(parts) >= 3 else s


def _first(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def read_site_table(path: str, quantity_suffix: str = " Intensity", site_col: str | None = None,
                    protein_col: str | None = None, gene_col: str | None = "Gene",
                    probability_col: str | None = None, min_probability: float | None = 0.75,
                    run_pattern: str = DEFAULT_RUN_PATTERN, min_observed: int = 1,
                    order_by_acquisition: bool = True) -> FragPipeData:
    """
    Reads a site-level (or modified-peptide) table with one row per site and one quantity
    column per run ("<run><quantity_suffix>"). Column names are detected where possible and
    can be given explicitly:
        site_col        site identifier, e.g. "Index" (P12345_S123), "Site", "Modified Peptide"
        protein_col     protein identifier used to match the whole-proteome table
        probability_col localisation probability; rows below min_probability are dropped
    Only quantity columns whose run name matches run_pattern are used when any match (so
    summary columns such as "Max Intensity" are ignored).
    """
    with open(path, encoding="utf-8-sig") as fh:
        header = fh.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
    cols = list(df.columns)

    site_col = site_col or _first(cols, ["Index", "Site", "Site ID", "Modified Peptide",
                                         "Modified Sequence", "Peptide"])
    protein_col = protein_col or _first(cols, ["Protein ID", "Protein", "Proteins"])
    if site_col is None or protein_col is None:
        raise ValueError(f"could not find site / protein columns; pass site_col= and "
                         f"protein_col= (columns: {cols[:15]} ...)")
    if probability_col is None:
        cand = [c for c in cols if re.search(r"locali[sz]ation prob", c, re.I)
                and not any(r in c for r in ("Spectral", "Intensity"))]
        probability_col = cand[0] if cand else None

    runs = [c[: -len(quantity_suffix)] for c in cols if c.endswith(quantity_suffix)]
    runs = [r for r in runs if not r.endswith(("MaxLFQ", "Max", "Combined", "Total"))
            and r.strip()]
    rx = re.compile(run_pattern)
    matched = [r for r in runs if rx.search(r)]
    runs = matched or runs
    if not runs:
        raise ValueError(f"no columns ending with {quantity_suffix!r}")

    keep = pd.Series(True, index=df.index)
    # localisation QC per run, before filtering: sites quantified vs sites localised
    loc_stats = None
    if probability_col is not None:
        q = df[[f"{r}{quantity_suffix}" for r in runs]].apply(pd.to_numeric, errors="coerce") > 0
        q = q[~df[protein_col].astype(str).str.startswith(("contam_", "rev_"))]
        prob = pd.to_numeric(df.loc[q.index, probability_col], errors="coerce")
        good = (prob >= (min_probability if min_probability is not None else 0.75)).to_numpy()
        loc_stats = pd.DataFrame({"sites_quantified": q.sum(0).to_numpy(),
                                  "sites_localised": q[good].sum(0).to_numpy()}, index=runs)
    if probability_col is not None and min_probability is not None:
        keep &= pd.to_numeric(df[probability_col], errors="coerce") >= min_probability
    elif min_probability is not None:
        warnings.warn("no localisation-probability column found; no localisation filter "
                      "applied (pass probability_col=)")
    prot = df[protein_col].astype(str)
    keep &= ~prot.str.startswith(("contam_", "rev_"))

    mat = df.loc[keep, [f"{r}{quantity_suffix}" for r in runs]].apply(pd.to_numeric,
                                                                    errors="coerce")
    mat = mat.where(mat > 0)
    mat = mat[mat.notna().sum(1) >= min_observed]

    ann = pd.DataFrame(index=mat.index)
    ann["site"] = df.loc[mat.index, site_col].astype(str)
    ann["protein_key"] = df.loc[mat.index, protein_col].map(accession)
    if gene_col and gene_col in df:
        ann["Gene"] = df.loc[mat.index, gene_col]
    if probability_col is not None:
        ann["localisation_probability"] = df.loc[mat.index, probability_col]
    gene = ann["Gene"].astype(str) if "Gene" in ann else ann["protein_key"]
    gene = gene.where(gene.notna() & (gene.str.strip() != "") & (gene != "nan"),
                      ann["protein_key"])
    pos = ann["site"].str.extract(r"([STYKRC]\d+)$")[0]          # "P12345_S123" -> "S123"
    name = np.where(pos.notna(), gene + "_" + pos.fillna(""), gene + "_" + ann["site"])
    name = pd.Series(name, index=ann.index)
    dup = name.duplicated(keep=False)
    name[dup] = name[dup] + "_" + ann.loc[dup, "site"]
    dup = name.duplicated(keep=False)
    name[dup] = name[dup] + "_" + pd.Series(np.arange(len(name)), index=name.index)[dup].astype(str)
    ann["feature_name"] = name
    ann["n_observed"] = mat.notna().sum(1)

    mat.columns = runs
    mat.index = ann["feature_name"].to_numpy()
    ann.index = mat.index
    table = parse_run_names(runs, run_pattern)
    if order_by_acquisition and "acquisition" in table and table["acquisition"].notna().all():
        table = table.sort_values("acquisition").reset_index(drop=True)
        mat = mat[table["run"].tolist()]
    table["missing_pct"] = mat.isna().mean(0).to_numpy() * 100
    if loc_stats is not None:
        table = table.join(loc_stats, on="run")
        table["localised_pct"] = 100 * table["sites_localised"] / table["sites_quantified"]
    _info(f"Read {len(runs)} runs, {mat.shape[0]} sites "
          f"({int((~keep).sum())} removed by localisation / contaminant filters); "
          f"{mat.isna().to_numpy().mean():.1%} missing.")
    return FragPipeData(intensities=mat, proteins=ann, runs=table, quantity=quantity_suffix)


# --------------------------------------------------------------------------------------
# Offsets
# --------------------------------------------------------------------------------------
def build_offsets(site_keys, site_samples, site_X, site_groups, protein_keys, protein_samples,
                  protein_model, protein_groups, min_protein_per_group: int = 2):
    """
    Offset matrix (site samples x sites) = protein log2 abundance in the matched sample, or
    the protein model's fitted group mean where the protein is missing in that sample.
    A site is only adjusted if its protein is measured in at least min_protein_per_group
    matched samples of every group: a fitted value for a group where the protein is (almost)
    never seen is very uncertain, and using it as a known offset creates false positives.
    Returns (offset with NaN for unadjustable sites, status per site, n_filled per site).
    """
    y_p = protein_model.y_input.numpy()                       # (S_p, P), normalised, collapsed
    beta = protein_model.beta.numpy()                         # (P, groups_p)
    fitted = protein_model.fitted.numpy()
    missing_groups = [g for g in site_groups if g not in protein_groups]
    if missing_groups:
        raise ValueError(f"groups {missing_groups} of the site data are not in the protein data")
    beta = beta[:, [list(protein_groups).index(g) for g in site_groups]]
    fit_vals = site_X @ beta.T                                # (S_s, P) fitted per site sample

    p_index = {k: i for i, k in enumerate(protein_keys)}
    s_index = {str(s): i for i, s in enumerate(protein_samples)}
    rows = np.array([s_index.get(str(s), -1) for s in site_samples])
    if (rows < 0).all():
        raise ValueError("no site sample matches a protein sample; check the sample IDs "
                         f"(site e.g. {site_samples[0]!r}, protein e.g. {protein_samples[0]!r})")
    n_unmatched = int((rows < 0).sum())
    if n_unmatched:
        warnings.warn(f"{n_unmatched} site samples have no matched protein sample; their "
                      "offsets use the protein model's fitted values")

    S, Fs = len(site_samples), len(site_keys)
    _, grp = np.unique(site_X, axis=0, return_inverse=True)
    grp = np.ravel(grp)
    offset = np.full((S, Fs), np.nan)
    status = np.empty(Fs, dtype=object)
    n_filled = np.zeros(Fs, dtype=int)
    for f, key in enumerate(site_keys):
        j = p_index.get(key)
        if j is None:
            status[f] = "protein not quantified"
            continue
        if not fitted[j]:
            status[f] = "protein not fitted"
            continue
        measured = np.full(S, np.nan)
        ok = rows >= 0
        measured[ok] = y_p[rows[ok], j]
        fill = np.isnan(measured)
        per_group = np.bincount(grp, weights=~fill, minlength=grp.max() + 1)
        if per_group.min() < min_protein_per_group:
            status[f] = "protein missing in a group"
            continue
        offset[:, f] = np.where(fill, fit_vals[:, j], measured)
        n_filled[f] = int(fill.sum())
        status[f] = "paired"
    return offset, status, n_filled


# --------------------------------------------------------------------------------------
# Independent combination (for comparison / benchmarking)
# --------------------------------------------------------------------------------------
def combine_independent(site_res: pd.DataFrame, protein_res: pd.DataFrame, site_keys,
                        protein_keys) -> pd.DataFrame:
    """estimate_site - estimate_protein, SE sqrt(se_s^2 + se_p^2), Welch-Satterthwaite df."""
    p_index = pd.Series(np.arange(len(protein_keys)), index=list(protein_keys))
    p_index = p_index[~p_index.index.duplicated()]
    j = p_index.reindex(list(site_keys)).to_numpy()
    ok = ~np.isnan(j)
    jj = np.where(ok, j, 0).astype(int)
    ep, sp, dp = (protein_res[c].to_numpy(float)[jj] for c in ("estimate", "se", "df"))
    ep, sp, dp = (np.where(ok, v, np.nan) for v in (ep, sp, dp))
    es, ss, ds = (site_res[c].to_numpy(float) for c in ("estimate", "se", "df"))
    est = es - ep
    v = ss ** 2 + sp ** 2
    df = v ** 2 / (ss ** 4 / ds + sp ** 4 / dp)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = est / np.sqrt(v)
    p = 2 * stats.t.sf(np.abs(t), df)
    return pd.DataFrame({"estimate": est, "se": np.sqrt(v), "df": df, "pvalue": p,
                         "padj": benjamini_hochberg(p)}, index=site_res.index)


# --------------------------------------------------------------------------------------
# Paired analysis
# --------------------------------------------------------------------------------------
@dataclass
class PTMResults:
    tables: dict                 # {contrast: combined site table}
    protein_results: dict        # {contrast: protein table}
    site_results: dict           # {contrast: unadjusted site table (full statistics)}
    site_model: object           # unadjusted site model
    adjusted_model: object       # paired (offset) site model
    protein_model: object
    site_missingness: object
    offset: np.ndarray
    status: np.ndarray


def run_paired_ptm(site_inputs, protein_inputs, contrast, site_normalisation="median_ratio",
                   protein_normalisation="median_ratio", missing: str = "selection",
                   moderate: bool = True, trend: bool = True, robust: bool = False,
                   include_sigma: bool = True, min_valid: int | None = None,
                   min_valid_per_group: int = 0, min_groups: int = 1,
                   figures_dir: str | None = None, figure_formats=("pdf", "png"),
                   figure_dpi: int = 600, font_size: float = 16, alpha: float = 0.05,
                   lfc_threshold: float = 1.0, min_protein_per_group: int = 2,
                   verbose: bool = False, progress=None, site_figure_kwargs: dict | None = None,
                   protein_figure_kwargs: dict | None = None) -> PTMResults:
    """
    site_inputs / protein_inputs: ProteomixInputs (from build_inputs) for the PTM and the
        matched whole-proteome data; their sample_names must refer to the same biological
        samples and their group names must agree. site_inputs.proteins needs "protein_key".
    contrast: as in run_differential_abundance (single, or dict {label: contrast}).
    missing: "selection" or "none" (imputation is not supported with offsets).
    min_protein_per_group: sites are adjusted only if their protein is measured in at least
        this many matched samples of every group (otherwise unadjusted only).
    progress: optional Progress bar; *_figure_kwargs: extra PCA options per layer
        (pca_factors, pca_shape, highlight).
    Returns PTMResults; tables[label] has, per site, the unadjusted and protein-adjusted
    statistics, the protein's own change, the adjustment status and the number of offsets
    filled from fitted protein values.
    """
    if missing not in ("selection", "none"):
        raise ValueError("paired PTM analysis supports missing='selection' or 'none'")
    contrasts = contrast if isinstance(contrast, dict) else {"contrast": contrast}
    common = dict(missing=missing, moderate=moderate, trend=trend, robust=robust,
                  include_sigma=include_sigma, min_valid=min_valid,
                  min_valid_per_group=min_valid_per_group, min_groups=min_groups,
                  figure_formats=figure_formats, figure_dpi=figure_dpi, font_size=font_size,
                  volcano_kwargs=dict(alpha=alpha, lfc_threshold=lfc_threshold),
                  verbose=verbose)
    sub = (lambda name: os.path.join(figures_dir, name)) if figures_dir else (lambda name: None)

    import logging
    log = logging.getLogger("proteomix")
    from .pipeline import n_pipeline_steps
    from .progress import NullProgress
    progress = progress or NullProgress()
    progress.add(2 * n_pipeline_steps(len(contrasts)) + 1 + len(contrasts)
                 + (2 * len(contrasts) if figures_dir else 0))
    prot_res, prot_model, _, _ = run_differential_abundance(
        **protein_inputs.pipeline_kwargs(), contrast=contrasts,
        normalisation=protein_normalisation, figures_dir=sub("protein"), progress=progress,
        layer="protein", **(protein_figure_kwargs or {}), **common)
    site_res, site_model, site_miss, _ = run_differential_abundance(
        **site_inputs.pipeline_kwargs(), contrast=contrasts,
        normalisation=site_normalisation, figures_dir=sub("site_unadjusted"),
        progress=progress, layer="site", **(site_figure_kwargs or {}), **common)

    site_keys = site_inputs.proteins.loc[site_inputs.feature_names, "protein_key"].to_numpy()
    protein_keys = protein_inputs.proteins.loc[protein_inputs.feature_names].get(
        "Protein ID", pd.Series(protein_inputs.feature_names,
                                index=protein_inputs.feature_names)).map(accession).to_numpy()
    offset, status, n_filled = build_offsets(
        site_keys, site_inputs.sample_names, site_model.x_input.numpy(),
        site_inputs.group_names, protein_keys, protein_inputs.sample_names, prot_model,
        protein_inputs.group_names, min_protein_per_group=min_protein_per_group)

    adjustable = torch.tensor(status == "paired")
    keep = adjustable & site_model.keep_filter
    if missing == "selection":
        adj_model = DifferentialAbundanceNR(site_miss.w.squeeze(1), site_miss.r.squeeze(1),
                                            include_sigma=include_sigma)
    else:
        adj_model = DifferentialAbundanceNR(missingness="ignore")
    adj_model.fit(site_model.y_input, site_model.x_input, site_model.batch_input,
                  offset=torch.tensor(np.nan_to_num(offset, nan=0.0)), keep=keep,
                  min_obs=min_valid, verbose=verbose)
    adj_model.y_input, adj_model.x_input = site_model.y_input, site_model.x_input
    adj_model.batch_input = site_model.batch_input
    st = pd.Series(status).value_counts().to_dict()
    log.debug(f"[ptm] offsets: {st}; filled protein values used for "
              f"{int(n_filled[status == 'paired'].sum())} site-sample cells "
              f"(min_protein_per_group={min_protein_per_group})")
    log.debug(f"[ptm] paired model fitted: {int(adj_model.fitted.sum())} sites")
    progress.step("[ptm] paired model fitted")

    p_index = pd.Series(np.arange(len(protein_keys)), index=list(protein_keys))
    p_index = p_index[~p_index.index.duplicated()]
    pj = p_index.reindex(list(site_keys)).to_numpy()
    has_p = ~np.isnan(pj)
    pj = np.where(has_p, pj, 0).astype(int)

    groups = site_inputs.group_names
    tables = {}
    names = site_inputs.feature_names
    for label, c in contrasts.items():
        adj = adj_model.test(_resolve(c, groups), feature_names=names, moderate=moderate,
                             trend=trend, robust=robust)
        un = site_res[label]
        pr = prot_res[label]
        t = site_inputs.proteins.loc[names, [c_ for c_ in ["site", "protein_key", "Gene",
                                                            "localisation_probability"]
                                              if c_ in site_inputs.proteins]].copy()
        t["adjustment"] = status
        t["n_offset_filled"] = np.where(status == "paired", n_filled, np.nan)
        for col in ("estimate", "se", "df", "pvalue", "padj"):
            t[f"{col}_adjusted"] = adj[col].to_numpy()
        for col in ("estimate", "pvalue", "padj"):
            t[f"{col}_unadjusted"] = un[col].to_numpy()
        for col in ("estimate", "padj"):
            t[f"protein_{col}"] = np.where(has_p, pr[col].to_numpy()[pj], np.nan)
        t["n_obs_site"] = un["n_obs"].to_numpy()
        t["amean_site"] = un["amean"].to_numpy()
        tables[label] = t.sort_values("pvalue_adjusted")
        log.debug(f"[ptm] contrast '{label}': {int((t['padj_adjusted'] < 0.05).sum())} "
                  f"adjusted, {int((t['padj_unadjusted'] < 0.05).sum())} unadjusted at FDR 5%")
        progress.step(f"[ptm] tested {label[:40]}")

    out = PTMResults(tables=tables, protein_results=prot_res, site_results=site_res,
                     site_model=site_model,
                     adjusted_model=adj_model, protein_model=prot_model,
                     site_missingness=site_miss, offset=offset, status=status)
    if figures_dir:
        from .figures import plot_site_vs_protein, plot_volcano, save_figure
        import matplotlib.pyplot as plt
        d = os.path.join(figures_dir, "ptm")
        for label, t in tables.items():
            slug = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")
            v = t.rename(columns={"estimate_adjusted": "estimate", "pvalue_adjusted": "pvalue",
                                  "padj_adjusted": "padj"})
            fig = plot_volcano(v, alpha=alpha, lfc_threshold=lfc_threshold,
                               contrast_label=f"{label} (protein-adjusted)",
                               font_size=font_size)
            save_figure(fig, os.path.join(d, f"volcano_adjusted_{slug}"), figure_formats,
                        figure_dpi)
            plt.close(fig)
            progress.step("[ptm] adjusted volcano")
            fig = plot_site_vs_protein(t, alpha=alpha, title=label, font_size=font_size)
            save_figure(fig, os.path.join(d, f"site_vs_protein_{slug}"), figure_formats,
                        figure_dpi)
            plt.close(fig)
            progress.step("[ptm] site vs protein")
    return out


# --------------------------------------------------------------------------------------
# Benchmarking: unadjusted vs independent combination vs paired offset
# --------------------------------------------------------------------------------------
def evaluate_ptm_methods(res: PTMResults, sim: dict, site_inputs, protein_inputs,
                         label: str, alpha: float = 0.05) -> list:
    """
    Metrics on sites whose protein could be adjusted (identical site set for every method).
    The target is the stoichiometry change dS; protein-driven sites are therefore nulls.
        fpr_protein_driven  p < alpha among protein-driven sites (should be ~ alpha)
        fpr_null            p < alpha among sites where nothing changes
        fdp                 false discoveries / discoveries at padj < alpha (target dS = 0)
        power               discoveries among sites with dS != 0
        lfc_bias / lfc_rmse error of the estimate against dS, sites with dS != 0
    """
    t = res.tables[label].reindex(site_inputs.feature_names)
    adjustable = (t["adjustment"] == "paired").to_numpy()
    truth = sim["truth"]
    cat, dS = truth["category"], truth["site_logfc"]
    site_keys = t["protein_key"].to_numpy()
    prot_keys = protein_inputs.proteins.loc[protein_inputs.feature_names, "Protein ID"].map(
        accession).to_numpy()
    un = res.site_results[label]
    ind = combine_independent(un, res.protein_results[label], site_keys, prot_keys)
    methods = {
        "unadjusted": (un["estimate"].to_numpy(), un["pvalue"].to_numpy()),
        "independent (Welch-Satterthwaite)": (ind["estimate"].to_numpy(), ind["pvalue"].to_numpy()),
        "paired (protein offset)": (t["estimate_adjusted"].to_numpy(),
                                    t["pvalue_adjusted"].to_numpy()),
    }
    rows = []
    for name, (est, p) in methods.items():
        sel = adjustable & np.isfinite(p)
        padj = np.full(len(p), np.nan)
        padj[sel] = benjamini_hochberg(p[sel])
        disc = np.nan_to_num(padj, nan=1.0) < alpha
        pd_ = sel & (cat == "protein-driven")
        nl = sel & (cat == "null")
        de = adjustable & (dS != 0)
        err = (est - dS)[sel & (dS != 0)]
        rows.append({
            "method": name, "n_sites": int(sel.sum()),
            "fpr_protein_driven": float(np.mean(p[pd_] < alpha)) if pd_.any() else np.nan,
            "fpr_null": float(np.mean(p[nl] < alpha)) if nl.any() else np.nan,
            "fdp": float((disc & (dS == 0)).sum() / max(disc.sum(), 1)),
            "power": float((disc & de).sum() / de.sum()) if de.any() else np.nan,
            "lfc_bias": float(np.mean(err * np.sign(dS[sel & (dS != 0)]))) if err.size else np.nan,
            "lfc_rmse": float(np.sqrt(np.mean(err ** 2))) if err.size else np.nan,
        })
    return rows


def run_ptm_study(scenarios: dict | None = None, n_sims: int = 10, alpha: float = 0.05,
                  seed: int = 0, verbose: bool = True, **ptm_kwargs):
    """
    Benchmarks the three site tests on simulate_paired_ptm data.
    scenarios: {name: kwargs for simulate_paired_ptm}; default varies shared_sd (the
    sample-level variation shared by a protein and its sites) and includes a no-change null.
    Returns (per-simulation DataFrame, summary of means and SDs).
    """
    from .simulation import paired_ptm_inputs, simulate_paired_ptm
    scenarios = scenarios or {
        "no site change": dict(prop_site_change=0.0),
        "shared sd 0.1": dict(shared_sd=0.1),
        "shared sd 0.3": dict(shared_sd=0.3),
        "shared sd 0.6": dict(shared_sd=0.6),
    }
    from .progress import Progress
    label = "Treatment vs Control"
    rows = []
    bar = Progress(len(scenarios) * n_sims, desc="PTM benchmark", enabled=verbose)
    for s_idx, (name, kw) in enumerate(scenarios.items()):
        for i in range(n_sims):
            sim = simulate_paired_ptm(seed=seed + 1000 * s_idx + i, **kw)
            si, pi = paired_ptm_inputs(sim)
            res = run_paired_ptm(si, pi, {label: ("Treatment", "Control")}, **ptm_kwargs)
            for r in evaluate_ptm_methods(res, sim, si, pi, label, alpha):
                rows.append({"scenario": name, "sim": i, **r})
            bar.step(f"{name} {i + 1}/{n_sims}")
    bar.close()
    per_sim = pd.DataFrame(rows)
    metrics = ["fpr_protein_driven", "fpr_null", "fdp", "power", "lfc_bias", "lfc_rmse"]
    summary = (per_sim.groupby(["scenario", "method"], sort=False)[metrics]
               .agg(["mean", "std"]).round(4))
    return per_sim, summary
