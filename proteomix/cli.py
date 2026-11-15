"""
Command-line pipeline: FragPipe protein table + run metadata -> tables, summary and figures.

    proteomix combined_protein.tsv -m metadata.tsv -o results --type ip
    proteomix combined_protein.tsv -m metadata.tsv -o results --type proteome --no-figures
    python -m proteomix.cli ...                      (same, without the installed command)

--type ip        IP / IgG pull-downs. Conditions are labelled like "IP TCS PDL1 5 min" and
                 "IgG unstimulated"; contrasts: IP vs IgG per condition, treatment vs control
                 within IP, IgG-corrected treatment effects, time vs unstimulated.
                 Default normalisation: none (IP vs IgG is too asymmetric for median scaling).
--type proteome  Whole-proteome comparison of arbitrary conditions; contrasts: every group vs
                 the reference group (first group, or --reference), or all pairs.
                 Default normalisation: median_ratio.
--type phospho   PTM sites (data = site table) with matched whole proteome (--protein-data):
                 paired protein-adjusted analysis (protein level as a per-sample offset) plus
                 unadjusted site and protein results. Contrasts as for proteome.
                 Samples are matched between the two files by sample ID: parsed from run
                 names, or taken from a metadata column (--sample-column).

The metadata file has two columns: run (or column) name and condition, e.g.
    230801kw_..._PD1_A1_S3_A1_1_15972 MaxLFQ Intensity<TAB>IP unstimulated
A header row is optional. Sample IDs (e.g. "A1") instead of run names also work.
Custom contrasts: --contrasts file.tsv with columns label, group1, group2 (group1 vs group2).

Outputs in the output folder:
    tables/<contrast>.csv       one table per contrast, sorted by p-value
    all_contrasts.csv           wide table: estimate / p-value / padj for every contrast
    runs.csv                    run table with condition, sample, % missing, size factor
    summary.txt                 design, settings and number of hits per contrast
    run.log                     detailed, timestamped log: parameters, software versions,
                                input-file checksums, every step and its duration, warnings
    qc/                         sample and run outlier tables (+ QC figure)
    enrichment/                 gene-set enrichment per contrast (with --gene-sets)
    ppi/                        interaction network and PPI enrichment (with --ppi)
    export/                     complete matrix for downstream analyses (with --export)
    timecourse/                 temporal profile clusters (with --timecourse)
    settings.json               all settings, for reproducibility
    figures/                    QC, missingness, volcano and heatmap figures (unless --no-figures)
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

TYPES = ("ip", "proteome", "phospho")


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="proteomix", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="FragPipe combined_protein file (.tsv or .csv); for --type "
                                 "phospho the site table")
    ap.add_argument("-m", "--metadata", required=True, help="run -> condition file")
    ap.add_argument("-o", "--out", required=True, help="output folder (created if needed)")
    ap.add_argument("-t", "--type", required=True, choices=TYPES, type=str.lower,
                    help="ip (pull-down vs IgG), proteome (whole-proteome comparison) or "
                         "phospho (PTM sites adjusted for matched whole proteome)")
    ap.add_argument("--figures", action=argparse.BooleanOptionalAction, default=True,
                    help="produce figures (default: yes; --no-figures to skip)")

    f0 = ap.add_argument_group("missingness filters (biological samples, after collapsing)")
    f0.add_argument("--min-valid", type=int, default=None,
                    help="minimum observed samples per protein in total "
                         "(default: number of groups + 2)")
    f0.add_argument("--min-valid-per-group", type=int, default=0,
                    help="require this many observed samples in at least --min-groups groups "
                         "(default 0 = off; e.g. 3)")
    f0.add_argument("--min-groups", type=int, default=1,
                    help="number of groups that must pass --min-valid-per-group (default 1)")

    ph = ap.add_argument_group("phospho / PTM options (--type phospho)")
    ph.add_argument("--protein-data", default=None,
                    help="matched whole-proteome combined_protein file")
    ph.add_argument("--protein-metadata", default=None,
                    help="metadata for the whole-proteome runs (default: --metadata)")
    ph.add_argument("--sample-column", type=int, default=None,
                    help="1-based metadata column with biological-sample IDs shared by both "
                         "files (default: sample IDs parsed from run names)")
    ph.add_argument("--site-quantity", default=" Intensity",
                    help="suffix of the per-run columns in the site table (default: ' Intensity')")
    ph.add_argument("--site-probability", type=float, default=0.75,
                    help="minimum localisation probability (default 0.75)")
    ph.add_argument("--protein-normalisation", default="median_ratio",
                    choices=["median_ratio", "total", "quantile", "none"])
    ph.add_argument("--min-protein-per-group", type=int, default=2,
                    help="adjust a site only if its protein is measured in at least this many "
                         "samples of every group (default 2)")

    g = ap.add_argument_group("analysis options")
    g.add_argument("--quantity", default="MaxLFQ Intensity",
                   help="FragPipe quantity column suffix (default: %(default)s)")
    g.add_argument("--normalisation", default="auto",
                   choices=["auto", "median_ratio", "total", "quantile", "none"],
                   help="auto: none for ip, median_ratio for proteome")
    g.add_argument("--missing", default="selection",
                   help="selection (default), none, or an imputation method "
                        "(zero, min_prob, mean, mean_noise, mean_group, knn)")
    g.add_argument("--no-moderation", action="store_true",
                   help="disable limma-style variance moderation")
    g.add_argument("--contrasts", default="auto",
                   help="auto, vs-reference, all-pairs, or a .tsv/.csv file "
                        "(columns: label, group1, group2)")
    g.add_argument("--reference", default=None, help="reference group for vs-reference")
    g.add_argument("--group-order", default=None,
                   help="comma-separated group order (default: ip -> IP before IgG, then "
                        "treatment and time; proteome -> order in the metadata)")
    g.add_argument("--min-observed", type=int, default=2,
                   help="keep proteins quantified in at least this many runs")
    g.add_argument("--fdr", type=float, default=0.05, help="FDR threshold for hits")
    g.add_argument("--lfc", type=float, default=1.0,
                   help="|log2 FC| threshold for coloured points in volcano plots")
    g.add_argument("--batch-column", default=None,
                   help="run-table column to use as missingness batch (e.g. tray); "
                        "default: one batch")
    g.add_argument("--run-pattern", default=None, help="regex for parsing run names")
    g.add_argument("--seed", type=int, default=0, help="seed for stochastic imputation")

    q = ap.add_argument_group("quality control and outliers")
    q.add_argument("--qc", action=argparse.BooleanOptionalAction, default=True,
                   help="sample / run outlier detection (default: yes)")
    q.add_argument("--outlier-threshold", type=float, default=3.5,
                   help="robust z-score threshold for flagging (default 3.5)")
    q.add_argument("--remove-outliers", action="store_true",
                   help="exclude flagged samples from the analysis (default: flag only)")
    q.add_argument("--exclude", default="",
                   help="comma-separated sample IDs or run names to exclude")
    q.add_argument("--bait", default=None,
                   help="IP experiments: bait gene symbol or protein ID (e.g. PDCD1) for the "
                        "bait-recovery check")
    q.add_argument("--peptide-data", default=None,
                   help="phospho: FragPipe modified-peptide table for enrichment efficiency")

    d = ap.add_argument_group("pathways, interactions and export")
    d.add_argument("--gene-sets", default=None,
                   help="comma-separated GMT files (gene symbols), e.g. ReactomePathways.gmt,"
                        "h.all.v2024.1.Hs.symbols.gmt")
    d.add_argument("--enrichment-methods", default="camera,ora",
                   help="camera (competitive, all proteins) and/or ora (hits only)")
    d.add_argument("--set-size", default="5,500", help="min,max genes per set (default 5,500)")
    d.add_argument("--ppi", default=None,
                   help="STRING protein.links file or an edge list (gene_a, gene_b[, score])")
    d.add_argument("--ppi-info", default=None,
                   help="STRING protein.info file (maps STRING IDs to gene symbols)")
    d.add_argument("--ppi-score", type=float, default=700,
                   help="minimum interaction score (STRING: 700 = high confidence)")
    d.add_argument("--export", default="none",
                   help="write a complete matrix: none (default), normalised, model, "
                        "model_draw, or an imputation method (min_prob, knn, ...)")
    d.add_argument("--export-design", default="none", choices=["none", "groups"],
                   help="model export: none = no group information in imputed values "
                        "(safe for classifiers), groups = group-specific")
    d.add_argument("--export-level", default="sample", choices=["sample", "run"])

    tc = ap.add_argument_group("time course (temporal profile clustering)")
    tc.add_argument("--timecourse", action="store_true",
                    help="cluster temporal profiles of responsive proteins")
    tc.add_argument("--timecourse-groups", default="auto",
                    help="groups to use: auto (ip: the IP groups; proteome: all), or "
                         "field=value (e.g. pulldown=IP), or comma-separated group names")
    tc.add_argument("--baseline", default="auto",
                    help="time-0 group shared by all series (auto: the unstimulated group)")
    tc.add_argument("--clusters", default="auto", help="number of clusters, or auto "
                                                       "(mean silhouette over 2-10)")
    tc.add_argument("--cluster-method", default="kmeans", choices=["kmeans", "fuzzy"])

    kn = ap.add_argument_group("kinases and motifs (--type phospho)")
    kn.add_argument("--kinase-predictions", default=None,
                    help="per-site kinase predictions (e.g. Kinase Library export; long or "
                         "wide format) for kinase activity inference")
    kn.add_argument("--kinase-method", default="both", choices=["both", "ksea", "camera"])
    kn.add_argument("--kinase-percentile", type=float, default=90,
                    help="a site is in a kinase's set if its percentile is at least this")
    kn.add_argument("--kinase-top", type=int, default=15,
                    help="... and the kinase is among the site's top N predictions")
    kn.add_argument("--kinase-min-size", type=int, default=3)
    kn.add_argument("--fasta", default=None,
                    help="protein sequences (UniProt FASTA) for site windows, residue checks, "
                         "sequence-based matching of predictions and motif enrichment")

    f = ap.add_argument_group("figure options")
    f.add_argument("--pca-colour-by", default="auto",
                   help="comma-separated factors for extra PCAs, e.g. pulldown,treatment,"
                        "time_min,tray (metadata or run-name fields); auto: pulldown, "
                        "treatment and time for --type ip, none otherwise; 'none' to skip")
    f.add_argument("--pca-shape", default=None,
                   help="factor used as marker shape in the PCAs (e.g. pulldown)")
    f.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                   help="show a progress bar (default: yes)")
    f.add_argument("--formats", default="pdf,png", help="comma-separated (pdf,png,tiff,svg)")
    f.add_argument("--dpi", type=int, default=600)
    f.add_argument("--font-size", type=float, default=16)
    f.add_argument("--heatmap-top", type=int, default=50, help="proteins per heatmap")
    return ap


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
class Log:
    """Messages for summary.txt; each also goes to the console and run.log."""
    def __init__(self, path, logger):
        self.path = path
        self.lines = []
        self.logger = logger

    def __call__(self, msg=""):
        for line in str(msg).split("\n"):
            self.logger.info(line)
        self.lines.append(str(msg))

    def save(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")


def _slug(label):
    s = "".join(ch if ch.isalnum() else "_" for ch in label)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _match_key(meta, data):
    """Metadata keyed by run names or by parsed sample IDs."""
    keys = set(meta["run"])
    if keys & set(data.runs["run"]):
        return "run"
    if keys & set(data.runs["sample"].astype(str)):
        return "sample"
    raise SystemExit(
        "Metadata names match neither run names nor sample IDs.\n"
        f"  metadata example: {meta['run'].iloc[0]!r}\n"
        f"  run name example: {data.runs['run'].iloc[0]!r}")


def _custom_contrasts(path, groups):
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    t = pd.read_csv(path, sep=sep, dtype=str)
    if t.shape[1] < 3:
        raise SystemExit("contrast file needs three columns: label, group1, group2")
    from .metadata import clean_condition
    out = {}
    for _, row in t.iterrows():
        a, b = clean_condition(row.iloc[1]), clean_condition(row.iloc[2])
        for g_ in (a, b):
            if g_ not in groups:
                raise SystemExit(f"contrast {row.iloc[0]!r}: group {g_!r} not in {groups}")
        out[str(row.iloc[0])] = (a, b)
    return out


def _choose_contrasts(args, meta, groups):
    from .metadata import ap_ms_contrasts
    mode = args.contrasts
    if mode not in ("auto", "vs-reference", "all-pairs"):
        return _custom_contrasts(mode, groups)
    if mode == "auto":
        mode = "apms" if args.type == "ip" else "vs-reference"
    if mode == "apms":
        c = ap_ms_contrasts(meta)
        if not c:
            raise SystemExit("No IP/IgG contrasts could be formed: conditions must start with "
                             "'IP' or 'IgG' (e.g. 'IP TCS PDL1 5 min'). Use --type proteome "
                             "or a contrast file.")
        return c
    if mode == "vs-reference":
        ref = args.reference or groups[0]
        if ref not in groups:
            raise SystemExit(f"reference {ref!r} not in groups {groups}")
        return {f"{g} vs {ref}": (g, ref) for g in groups if g != ref}
    return {f"{groups[j]} vs {groups[i]}": (groups[j], groups[i])
            for i in range(len(groups)) for j in range(i + 1, len(groups))}


def _observed_counts(model, vec):
    """Number of observed samples per protein on the + and - side of a contrast."""
    obs = ~np.isnan(model.y_input.numpy())
    X = model.x_input.numpy()
    plus = (X[:, vec > 0] != 0).any(1)
    minus = (X[:, vec < 0] != 0).any(1)
    return obs[plus].sum(0), obs[minus].sum(0), plus.sum(), minus.sum()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def _load_layer(args, log, data, metadata_path, name):
    """Metadata + build_inputs for one data layer; returns (inputs, meta, groups, key)."""
    from .fragpipe import build_inputs
    from .metadata import condition_order, read_run_metadata

    sample_col = None if args.sample_column is None else args.sample_column - 1
    meta = read_run_metadata(metadata_path, parse_factors=(args.type == "ip"),
                             sample_col=sample_col)
    key = _match_key(meta, data)
    if key == "sample":
        meta = meta.rename(columns={"run": "sample"})
    if args.group_order:
        groups = [s_.strip() for s_ in args.group_order.split(",")]
    elif args.type == "ip":
        groups = condition_order(meta)
    else:
        groups = list(dict.fromkeys(meta["condition"]))
    if args.batch_column and args.batch_column not in data.runs:
        raise SystemExit(f"--batch-column {args.batch_column!r} not in run table columns "
                         f"{list(data.runs.columns)}")
    rep_col = "sample_id" if sample_col is not None else "sample"
    inputs = build_inputs(data, meta, key=key, group_col="condition", group_order=groups,
                          batch_col=args.batch_column, replicate_col=rep_col,
                          drop_unmatched=True)
    log(f"\n{name} design (biological samples x technical replicates):")
    design = inputs.runs.groupby("condition", sort=False).agg(
        samples=(rep_col, "nunique"), runs=("run", "size")).reindex(groups)
    for g_, row in design.iterrows():
        log(f"  {g_:30s} {row['samples']:3d} samples, {row['runs']:3d} runs")
    if (design["samples"] < 2).any():
        log("  WARNING: groups with fewer than 2 biological samples cannot be tested well.")
    return inputs, meta, groups, key


def _filters(args):
    return dict(min_valid=args.min_valid, min_valid_per_group=args.min_valid_per_group,
                min_groups=args.min_groups)


def run(args) -> dict:
    from .fragpipe import DEFAULT_RUN_PATTERN, read_fragpipe
    from .metadata import contrast_vector
    from .pipeline import MISSING_HANDLING, run_differential_abundance

    from .progress import NullProgress, Progress
    from .runlog import log_environment, log_inputs, log_parameters, setup_run_logging

    t0 = time.time()
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)
    logger = setup_run_logging(args.out)
    log_environment(logger)
    log_parameters(logger, vars(args))
    args.input_files = log_inputs(logger, [args.data, args.metadata,
                                           getattr(args, "protein_data", None),
                                           getattr(args, "protein_metadata", None)])
    logger.debug("=" * 78)
    log = Log(os.path.join(args.out, "summary.txt"), logger)
    args.progress_bar = (Progress(3, desc="proteomix") if args.progress else NullProgress())
    log(f"proteomix run {datetime.now():%Y-%m-%d %H:%M}  |  type: {args.type}")
    log(f"data: {args.data}\nmetadata: {args.metadata}\noutput: {args.out}")
    if args.type == "phospho":
        log(f"protein data: {args.protein_data}\n"
            f"protein metadata: {args.protein_metadata or args.metadata}")
    log("")

    if args.missing not in MISSING_HANDLING:
        raise SystemExit(f"--missing must be one of {MISSING_HANDLING}")
    if args.type == "phospho":
        if not args.protein_data:
            raise SystemExit("--type phospho needs --protein-data (matched whole proteome)")
        if args.missing not in ("selection", "none"):
            raise SystemExit("--type phospho supports --missing selection or none")
    normalisation = args.normalisation
    if normalisation == "auto":
        normalisation = "none" if args.type == "ip" else "median_ratio"
    pattern = args.run_pattern or DEFAULT_RUN_PATTERN
    formats = tuple(s_.strip().lower() for s_ in args.formats.split(",") if s_.strip())
    fig_dir = os.path.join(args.out, "figures") if args.figures else None

    if args.type == "phospho":
        return _run_phospho(args, log, t0, normalisation, pattern, formats, fig_dir)

    # ---- data and design -------------------------------------------------------------
    data = read_fragpipe(args.data, quantity=args.quantity, min_observed=args.min_observed,
                         run_pattern=pattern)
    log(f"{len(data.runs)} runs, {data.runs['sample'].nunique()} samples, "
        f"{data.intensities.shape[0]} proteins, "
        f"{data.intensities.isna().to_numpy().mean():.1%} missing values")
    bar = args.progress_bar
    bar.step("data read")
    inputs, meta, groups, key = _load_layer(args, log, data, args.metadata, "Protein")
    bar.step("metadata matched")
    inputs, qc_tab = _qc_and_exclusions(args, log, inputs, normalisation, "protein", fig_dir,
                                        formats)
    bar.step("quality control")

    contrasts = _choose_contrasts(args, meta, groups)
    vectors = {k: contrast_vector(v, groups) for k, v in contrasts.items()}
    log(f"\n{len(contrasts)} contrasts; normalisation: {normalisation}; missing values: "
        f"{args.missing}; variance moderation: {not args.no_moderation}")

    # ---- analysis ---------------------------------------------------------------------
    from .pipeline import n_pipeline_steps
    bar.add(n_pipeline_steps(len(contrasts)))
    results, model, miss, log_sf = run_differential_abundance(
        progress=bar, layer="protein", **_pca_options(args, inputs, qc_tab),
        **inputs.pipeline_kwargs(), contrast=contrasts, normalisation=normalisation,
        missing=args.missing, moderate=not args.no_moderation, seed=args.seed,
        figures_dir=fig_dir, figure_formats=formats, figure_dpi=args.dpi,
        font_size=args.font_size, volcano_kwargs=dict(alpha=args.fdr, lfc_threshold=args.lfc),
        heatmap_kwargs=dict(top_n=args.heatmap_top), **_filters(args))
    log(f"Proteins tested: {int(model.fitted.sum())} of {len(model.fitted)} "
        f"(others removed by missingness filters)")

    # ---- outputs ----------------------------------------------------------------------
    ann_cols = [c for c in ["Protein ID", "Entry Name", "Gene", "Description"]
                if c in inputs.proteins]
    annotation = inputs.proteins[ann_cols]
    wide = annotation.copy()
    log(f"\nHits at FDR {args.fdr:g}:")
    counts = []
    for label, res in results.items():
        n_plus, n_minus, _, _ = _observed_counts(model, vectors[label])
        res = res.assign(n_obs_group1=n_plus, n_obs_group2=n_minus)
        if args.type == "ip":
            res["only_in_group1"] = (n_plus > 0) & (n_minus == 0)
        annotation.join(res).sort_values("pvalue").to_csv(
            os.path.join(args.out, "tables", f"{_slug(label)}.csv"))
        for col in ("estimate", "pvalue", "padj"):
            wide[f"{label} | {col}"] = res[col]
        counts.append(_count(log, label, res["estimate"], res["padj"], args.fdr))
    wide.to_csv(os.path.join(args.out, "all_contrasts.csv"))
    pd.DataFrame(counts).to_csv(os.path.join(args.out, "hits_per_contrast.csv"), index=False)

    genes = _gene_symbols(inputs.proteins, inputs.feature_names)
    for label, res in results.items():
        _downstream(args, log, label, res, genes, model, inputs.feature_names, fig_dir, formats)
    _export(args, log, inputs, normalisation, "protein")
    if args.timecourse:
        _timecourse(args, log, inputs, meta, groups, results, vectors, model, genes, fig_dir,
                    formats)

    runs = inputs.runs.copy()
    if log_sf is not None:
        runs["log2_size_factor"] = log_sf.numpy()
    runs.to_csv(os.path.join(args.out, "runs.csv"), index=False)
    _log_models(log, miss, model, inputs)
    if args.figures:
        log(f"\n{len(model.figures)} figures written to {fig_dir}")
    _finish(args, log, t0, normalisation, key, groups, contrasts)
    return {"results": results, "model": model, "inputs": inputs, "missingness": miss,
            "qc": qc_tab}


def _count(log, label, est, padj, fdr):
    sig = padj < fdr
    up, down = int((sig & (est > 0)).sum()), int((sig & (est < 0)).sum())
    log(f"  {label:48s} {int(sig.sum()):5d}  (up {up}, down {down})")
    return {"contrast": label, "hits": int(sig.sum()), "up": up, "down": down}


def _log_models(log, miss, model, inputs, name=""):
    if miss is not None:
        log(f"\n{name}Missingness model, P(missing) = sigmoid(w (log2 intensity - r)):")
        for b, (w, r) in enumerate(zip(miss.w.squeeze(1).tolist(), miss.r.squeeze(1).tolist())):
            bn = inputs.batch_names[b] if b < len(inputs.batch_names) else b
            log(f"  batch {bn}: w = {w:.2f}, r = {r:.2f}")
    if getattr(model, "var_prior", None):
        log(f"{name}Variance moderation: prior df d0 = {model.var_prior['d0_global']:.1f}")


def _finish(args, log, t0, normalisation, key, groups, contrasts):
    args.progress_bar.close()
    settings = {k: v for k, v in vars(args).items() if k != "progress_bar"}
    settings.update(normalisation_used=normalisation, matched_metadata_on=key, groups=groups,
                    contrasts={k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in contrasts.items()},
                    proteomix_version=_version(), date=datetime.now().isoformat())
    with open(os.path.join(args.out, "settings.json"), "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2, default=str)
    log(f"\nDone in {time.time() - t0:.0f} s. Detailed log: "
        f"{os.path.join(args.out, 'run.log')}")
    log.save()


def _run_phospho(args, log, t0, normalisation, pattern, formats, fig_dir):
    from .fragpipe import read_fragpipe
    from .ptm import read_site_table, run_paired_ptm

    sites = read_site_table(args.data, quantity_suffix=args.site_quantity,
                            min_probability=args.site_probability, run_pattern=pattern,
                            min_observed=args.min_observed)
    prots = read_fragpipe(args.protein_data, quantity=args.quantity,
                          min_observed=args.min_observed, run_pattern=pattern)
    log(f"Sites: {len(sites.runs)} runs, {sites.intensities.shape[0]} sites, "
        f"{sites.intensities.isna().to_numpy().mean():.1%} missing")
    log(f"Proteins: {len(prots.runs)} runs, {prots.intensities.shape[0]} proteins, "
        f"{prots.intensities.isna().to_numpy().mean():.1%} missing")
    bar = args.progress_bar
    bar.step("data read")
    site_in, meta, groups, key = _load_layer(args, log, sites, args.metadata, "Site")
    prot_in, _, pgroups, _ = _load_layer(args, log, prots,
                                         args.protein_metadata or args.metadata, "Protein")
    bar.step("metadata matched")
    site_in, site_qc = _qc_and_exclusions(args, log, site_in, normalisation, "site", fig_dir,
                                          formats)
    prot_in, prot_qc = _qc_and_exclusions(args, log, prot_in, args.protein_normalisation,
                                          "protein", fig_dir, formats)
    if args.remove_outliers:                      # a sample flagged in one layer leaves both
        drop = (set(site_qc.loc[site_qc["outlier"], "sample"])
                | set(prot_qc.loc[prot_qc["outlier"], "sample"]))
        from .qc import exclusion_mask, subset_inputs
        for name in ("site", "protein"):
            inp = site_in if name == "site" else prot_in
            present = drop & set(inp.sample_names)
            if present:
                inp = subset_inputs(inp, exclusion_mask(inp, present))
                log(f"  --remove-outliers: {sorted(present)} removed from the {name} layer "
                    f"(flagged in either layer)")
            if name == "site":
                site_in = inp
            else:
                prot_in = inp
    bar.step("quality control")
    if set(groups) != set(pgroups):
        raise SystemExit(f"site and protein groups differ: {groups} vs {pgroups}")
    shared = set(site_in.sample_names) & set(prot_in.sample_names)
    log(f"\nSamples matched between site and protein data: {len(shared)} of "
        f"{len(site_in.sample_names)} site samples")
    if not shared:
        raise SystemExit("no sample IDs are shared between the site and protein data; use "
                         "--sample-column with a metadata column of shared sample IDs")

    contrasts = _choose_contrasts(args, meta, groups)
    log(f"{len(contrasts)} contrasts; site normalisation: {normalisation}; protein "
        f"normalisation: {args.protein_normalisation}; missing values: {args.missing}; "
        f"variance moderation: {not args.no_moderation}")

    res = run_paired_ptm(
        site_in, prot_in, contrasts, progress=bar,
        site_figure_kwargs=_pca_options(args, site_in, site_qc),
        protein_figure_kwargs=_pca_options(args, prot_in, prot_qc),
        site_normalisation=normalisation,
        protein_normalisation=args.protein_normalisation, missing=args.missing,
        moderate=not args.no_moderation, figures_dir=fig_dir, figure_formats=formats,
        figure_dpi=args.dpi, font_size=args.font_size, alpha=args.fdr,
        lfc_threshold=args.lfc, min_protein_per_group=args.min_protein_per_group,
        **_filters(args))

    status = pd.Series(res.status).value_counts()
    log("\nProtein adjustment of sites:")
    for k, v in status.items():
        log(f"  {k:30s} {v:6d}")

    os.makedirs(os.path.join(args.out, "tables_protein"), exist_ok=True)
    wide = site_in.proteins[[c for c in ["site", "protein_key", "Gene"]
                             if c in site_in.proteins]].copy()
    counts = []
    log(f"\nSite hits at FDR {args.fdr:g} (protein-adjusted | unadjusted):")
    for label, t in res.tables.items():
        t.to_csv(os.path.join(args.out, "tables", f"{_slug(label)}.csv"))
        pr = res.protein_results[label]
        prot_in.proteins[[c for c in ["Protein ID", "Gene", "Description"]
                          if c in prot_in.proteins]].join(pr).sort_values("pvalue").to_csv(
            os.path.join(args.out, "tables_protein", f"{_slug(label)}.csv"))
        for col in ("estimate_adjusted", "pvalue_adjusted", "padj_adjusted",
                    "estimate_unadjusted", "padj_unadjusted"):
            wide[f"{label} | {col}"] = t[col]
        a = _count(log, f"{label} [adjusted]", t["estimate_adjusted"], t["padj_adjusted"],
                   args.fdr)
        u = _count(log, f"{label} [unadjusted]", t["estimate_unadjusted"],
                   t["padj_unadjusted"], args.fdr)
        p_ = _count(log, f"{label} [protein]", pr["estimate"], pr["padj"], args.fdr)
        counts.append({"contrast": label, "adjusted_hits": a["hits"],
                       "adjusted_up": a["up"], "adjusted_down": a["down"],
                       "unadjusted_hits": u["hits"], "protein_hits": p_["hits"]})
    wide.to_csv(os.path.join(args.out, "all_contrasts.csv"))
    pd.DataFrame(counts).to_csv(os.path.join(args.out, "hits_per_contrast.csv"), index=False)

    site_genes = _gene_symbols(site_in.proteins, site_in.feature_names)
    prot_genes = _gene_symbols(prot_in.proteins, prot_in.feature_names)
    for label, t in res.tables.items():
        adj = t.reindex(site_in.feature_names).rename(columns={
            "estimate_adjusted": "estimate", "pvalue_adjusted": "pvalue",
            "padj_adjusted": "padj"})
        _downstream(args, log, f"{label} sites adjusted", adj, site_genes, res.adjusted_model,
                    site_in.feature_names, fig_dir, formats)
        _downstream(args, log, f"{label} protein", res.protein_results[label], prot_genes,
                    res.protein_model, prot_in.feature_names, fig_dir, formats)
    _kinases_and_sites(args, log, site_in, meta, groups, contrasts, res, site_genes,
                       fig_dir, formats)
    _export(args, log, site_in, normalisation, "site")
    _export(args, log, prot_in, args.protein_normalisation, "protein")
    pd.concat([site_in.runs.assign(layer="site"), prot_in.runs.assign(layer="protein")],
              ignore_index=True).to_csv(os.path.join(args.out, "runs.csv"), index=False)
    _log_models(log, res.site_missingness, res.adjusted_model, site_in, "Site ")
    if args.figures:
        log(f"\nFigures written to {fig_dir} (protein/, site_unadjusted/, ptm/)")
    _finish(args, log, t0, normalisation, key, groups, contrasts)
    return {"ptm": res, "site_inputs": site_in, "protein_inputs": prot_in}


def _gene_symbols(proteins, features):
    """Gene symbol per feature; protein ID (or feature name) where the symbol is missing."""
    t = proteins.loc[list(features)]
    fallback = t["protein_key"] if "protein_key" in t else (
        t["Protein ID"] if "Protein ID" in t else pd.Series(list(features), index=t.index))
    if "Gene" not in t:
        return fallback.astype(str).tolist()
    g = t["Gene"].astype(str).str.strip()
    ok = t["Gene"].notna() & g.ne("") & g.ne("nan")
    return g.where(ok, fallback.astype(str)).tolist()


_CACHE = {}


def _downstream(args, log, label, res, genes, model, features, fig_dir, formats):
    """Gene-set enrichment and PPI analysis for one contrast (when requested)."""
    if not args.gene_sets and not args.ppi:
        return
    import matplotlib.pyplot as plt
    from .figures import plot_enrichment, plot_network, save_figure
    slug = _slug(label)
    if args.gene_sets:
        from .enrichment import collection_name, read_gmt, run_enrichment
        if "sets" not in _CACHE:
            _CACHE["sets"] = {collection_name(p): read_gmt(p.strip())
                              for p in args.gene_sets.split(",") if p.strip()}
            for k, v in _CACHE["sets"].items():
                log.logger.debug(f"gene sets {k}: {len(v)} sets")
        lo, hi = (int(v) for v in args.set_size.split(","))
        methods = tuple(m.strip() for m in args.enrichment_methods.split(","))
        enr = run_enrichment(res, genes, _CACHE["sets"], model=model, features=features,
                             methods=methods, alpha=args.fdr, lfc=0.0, min_size=lo,
                             max_size=hi)
        d = os.path.join(args.out, "enrichment", slug)
        os.makedirs(d, exist_ok=True)
        parts = []
        for (coll, test), tab in enr.items():
            tab.to_csv(os.path.join(d, f"{_slug(coll)}_{test}.csv"), index=False)
            n_sig = int((tab["padj"] < args.fdr).sum()) if len(tab) else 0
            parts.append(f"{coll} {test}: {n_sig}")
            if fig_dir and n_sig and test in ("camera", "ora_all"):
                fig = plot_enrichment(tab, alpha=args.fdr, font_size=args.font_size,
                                      title=f"{label}: {coll} ({test})")
                save_figure(fig, os.path.join(fig_dir, "enrichment", f"{slug}_{_slug(coll)}_"
                                              f"{test}"), formats, args.dpi)
                plt.close(fig)
        log(f"  enrichment [{label}] sets at FDR {args.fdr:g}: " + "; ".join(parts))
    if args.ppi:
        from .ppi import ppi_analysis, read_network, write_network
        if "net" not in _CACHE:
            _CACHE["net"] = read_network(args.ppi, args.ppi_info, args.ppi_score)
            log.logger.debug(f"PPI network: {len(_CACHE['net'])} interactions "
                             f"(score >= {args.ppi_score:g})")
        pr = ppi_analysis(res, genes, _CACHE["net"], alpha=args.fdr, seed=args.seed)
        write_network(pr, os.path.join(args.out, "ppi", slug))
        s_ = pr["summary"].iloc[0]
        log(f"  PPI [{label}]: {int(s_['observed_edges'])} interactions among "
            f"{int(s_['hits_in_network'])} hits (expected {s_['expected_edges']:.1f}, "
            f"p = {s_['pvalue']:.3g}); {len(pr['modules'])} modules")
        if fig_dir and len(pr["edges"]):
            fig = plot_network(pr, title=f"{label}: interactions among hits",
                               font_size=args.font_size)
            save_figure(fig, os.path.join(fig_dir, "ppi", f"{slug}_network"), formats, args.dpi)
            plt.close(fig)


def _timecourse(args, log, inputs, meta, groups, results, vectors, model, genes, fig_dir,
                formats, subdir="timecourse", what="proteins", cluster_features=None):
    """
    Temporal profile clustering of features (proteins or sites) responsive in the time-course
    contrasts. cluster_features: optional boolean table (feature x motif / kinase set) tested
    for enrichment in each cluster.
    """
    import matplotlib.pyplot as plt
    from .timecourse import (build_profiles, cluster_profiles, responsive_proteins,
                             time_design)
    first = meta.drop_duplicates("condition").set_index("condition")
    spec = args.timecourse_groups
    if spec == "auto":
        subset = [g for g in groups if args.type != "ip" or
                  first.loc[g].get("pulldown", "IP") == "IP"]
    elif "=" in spec:
        field, value = (x.strip() for x in spec.split("=", 1))
        if field not in first:
            raise SystemExit(f"--timecourse-groups: {field!r} not in metadata")
        subset = [g for g in groups if str(first.loc[g, field]) == value]
    else:
        subset = [g.strip() for g in spec.split(",")]
    if args.type == "ip" and {"treatment", "time_min"} <= set(first.columns):
        sub = first.loc[subset]
        design = time_design(subset, series=sub["treatment"].tolist(),
                             times=sub["time_min"].tolist())
        base = [g for g in subset if sub.loc[g, "treatment"] == "unstimulated"]
        design["baseline"] = design["group"].isin(base if args.baseline == "auto"
                                                  else [args.baseline])
    else:
        design = time_design(subset, baseline=None if args.baseline == "auto" else args.baseline)
    n_series = design.loc[~design["baseline"], "series"].nunique()
    if design["time"].nunique() < 3:
        log("\nTime course: fewer than 3 time points in the selected groups; skipped")
        return
    proteins = responsive_proteins(results, vectors, groups, args.fdr, subset)
    if len(proteins) < 10:
        log(f"\nTime course: only {len(proteins)} responsive proteins; skipped")
        return
    profiles = build_profiles(model, inputs.feature_names, design, groups, proteins)
    k = "auto" if args.clusters == "auto" else int(args.clusters)
    tc = cluster_profiles(profiles, k=k, method=args.cluster_method, seed=args.seed)
    d = os.path.join(args.out, subdir)
    os.makedirs(d, exist_ok=True)
    gene_of = dict(zip(inputs.feature_names, genes))
    flat = tc.profiles.copy()
    flat.columns = [f"{s_} | {t:g} min" for s_, t in flat.columns]
    table = tc.assignments.join(flat)
    table.insert(0, "gene", [gene_of.get(p_, p_) for p_ in table.index])
    table.sort_values(["cluster", "membership"], ascending=[True, False]).to_csv(
        os.path.join(d, "clusters.csv"))
    tc.centroids.to_csv(os.path.join(d, "centroids_zscore.csv"))
    design.to_csv(os.path.join(d, "design.csv"), index=False)
    if tc.silhouette:
        pd.Series(tc.silhouette, name="mean_silhouette").rename_axis("k").to_csv(
            os.path.join(d, "silhouette.csv"))
    log(f"\nTime course: {len(tc.profiles)} responsive {what}, {n_series} series x "
        f"{design['time'].nunique()} time points; {tc.k} clusters ({args.cluster_method}"
        + (f", k by silhouette {tc.silhouette[tc.k]:.2f}" if tc.silhouette else "") + ")")
    for c, n_ in tc.assignments["cluster"].value_counts().sort_index().items():
        top = table[table["cluster"] == c].sort_values("membership", ascending=False)
        log(f"  cluster {c}: {n_} proteins (e.g. {', '.join(map(str, top['gene'].head(5)))})")
    if args.gene_sets and "sets" in _CACHE:
        from .enrichment import gene_level_stats, ora
        any_res = next(iter(results.values()))
        universe_genes = set(gene_level_stats(any_res, genes).index)
        for coll, sets in _CACHE["sets"].items():
            uni = {g_ for g_ in universe_genes if any(g_ in s2 for s2 in sets.values())}
            for c in range(1, tc.k + 1):
                hits = {str(gene_of.get(p_, p_)).upper()
                        for p_ in tc.assignments.index[tc.assignments["cluster"] == c]}
                ora(hits, uni, sets).to_csv(os.path.join(d, f"enrichment_cluster{c}_"
                                                          f"{_slug(coll)}.csv"), index=False)
    if fig_dir:
        from .figures import plot_timecourse_clusters, save_figure
        fig = plot_timecourse_clusters(tc, font_size=args.font_size)
        save_figure(fig, os.path.join(fig_dir, f"{subdir}_clusters"), formats, args.dpi)
        plt.close(fig)
    if cluster_features is not None and len(cluster_features.columns):
        from .kinases import cluster_enrichment
        ce = cluster_enrichment(tc.assignments["cluster"], cluster_features,
                                list(inputs.feature_names))
        ce.to_csv(os.path.join(d, "cluster_enrichment.csv"), index=False)
        for c in sorted(tc.assignments["cluster"].unique()):
            sig = ce[(ce["cluster"] == c) & (ce["padj"] < args.fdr)] if len(ce) else ce
            if len(sig):
                log(f"  cluster {c} enriched for: " + ", ".join(
                    f"{f} ({100 * fr:.0f}% vs {100 * bg:.0f}%)" for f, fr, bg in
                    zip(sig["feature"].head(4), sig["fraction_in_cluster"].head(4),
                        sig["fraction_background"].head(4))))


def _kinases_and_sites(args, log, site_in, meta, groups, contrasts, res, genes, fig_dir,
                       formats):
    """Site windows (FASTA), kinase activity inference and site time-course clustering."""
    if not (args.fasta or args.kinase_predictions or args.timecourse):
        return
    import matplotlib.pyplot as plt
    from .kinases import (kinase_activity, kinase_sets, motif_table, plot_kinase_heatmap,
                          read_fasta, read_kinase_predictions, site_positions, site_windows)
    from .metadata import contrast_vector
    keys = site_in.proteins.loc[site_in.feature_names].get(
        "protein_key", pd.Series([None] * len(site_in.feature_names))).tolist()
    sites = site_positions(site_in.feature_names, genes, keys)
    d = os.path.join(args.out, "kinases")
    features = None
    if args.fasta:
        sites = site_windows(sites, read_fasta(args.fasta))
        os.makedirs(d, exist_ok=True)
        sites.to_csv(os.path.join(d, "site_windows.csv"), index=False)
        have = sites["window"].notna()
        bad = have & ~sites["residue_matches"]
        log(f"\nSequence windows: {int(have.sum())} of {len(sites)} sites found in the FASTA"
            + (f"; {int(bad.sum())} residue mismatches (other isoform / numbering?)"
               if bad.any() else ""))
        ok = (have & ~bad).to_numpy()
        motifs = motif_table(sites.loc[ok].set_index("site")["window"])
        features = motifs
    tables = {label: t.reindex(site_in.feature_names).rename(columns={
        "estimate_adjusted": "estimate", "pvalue_adjusted": "pvalue", "padj_adjusted": "padj"})
        for label, t in res.tables.items()}
    if args.kinase_predictions:
        os.makedirs(d, exist_ok=True)
        pred = read_kinase_predictions(args.kinase_predictions, sites)
        sets = kinase_sets(pred, args.kinase_percentile, args.kinase_top, args.kinase_min_size)
        pd.DataFrame([{"kinase": k, "n_sites": len(v), "sites": ";".join(sorted(v))}
                      for k, v in sets.items()]).sort_values("n_sites", ascending=False) \
            .to_csv(os.path.join(d, "kinase_sets.csv"), index=False)
        log(f"\nKinase predictions: {pred['site'].nunique()} sites matched; {len(sets)} "
            f"kinases with >= {args.kinase_min_size} predicted substrates (percentile >= "
            f"{args.kinase_percentile:g}, top {args.kinase_top})")
        methods = ("ksea", "camera") if args.kinase_method == "both" else (args.kinase_method,)
        per_method = {m: {} for m in methods}
        for label, t in tables.items():
            act = kinase_activity(t, sets, methods, model=res.adjusted_model,
                                  features=site_in.feature_names,
                                  min_size=args.kinase_min_size)
            for m, tab in act.items():
                tab.to_csv(os.path.join(d, f"{_slug(label)}_{m}.csv"), index=False)
                per_method[m][label] = tab
                sig = tab[tab["padj"] < args.fdr] if len(tab) else tab
                log(f"  {m.upper()} [{label}]: {len(sig)} kinases at FDR {args.fdr:g}"
                    + (": " + ", ".join(f"{k} ({dr})" for k, dr in
                                         zip(sig["kinase"].head(6), sig["direction"].head(6)))
                       if len(sig) else ""))
        if fig_dir:
            from .figures import save_figure
            for m, tabs in per_method.items():
                try:
                    fig = plot_kinase_heatmap(tabs, m, args.fdr, font_size=args.font_size)
                    save_figure(fig, os.path.join(fig_dir, "kinases", f"kinase_activity_{m}"),
                                formats, args.dpi)
                    plt.close(fig)
                except ValueError as exc:
                    log(f"  {m}: {exc}")
        ks = pd.DataFrame({k: [s_ in v for s_ in site_in.feature_names] for k, v in sets.items()},
                          index=site_in.feature_names)
        features = ks if features is None else features.join(ks, how="outer").fillna(False)
    if args.timecourse:
        vectors = {k: contrast_vector(v, groups) for k, v in contrasts.items()}
        _timecourse(args, log, site_in, meta, groups, tables, vectors, res.adjusted_model,
                    genes, fig_dir, formats, subdir="timecourse_sites", what="sites",
                    cluster_features=features)


def _export(args, log, inputs, normalisation, layer):
    if args.export == "none":
        return
    from .export import export_matrix, write_export
    r = export_matrix(inputs, method=args.export, normalisation=normalisation,
                      design=args.export_design, level=args.export_level, seed=args.seed)
    paths = write_export(r, os.path.join(args.out, "export"), prefix=f"{layer}_matrix")
    log(f"\nExport ({layer}): {r.info['n_rows']} {args.export_level}s x "
        f"{r.info['n_features']} features, {r.info['imputed_fraction']:.1%} imputed "
        f"({args.export}, design {args.export_design}) -> {paths[0]}")
    if args.export_design == "groups":
        log("  note: imputed values use group labels; do not use this matrix to train a "
            "classifier of those groups")


def _qc_and_exclusions(args, log, inputs, normalisation, layer, fig_dir, formats):
    """Sample / run QC, then manual (--exclude) and automatic (--remove-outliers) removal."""
    from .qc import exclusion_mask, run_quality_control, subset_inputs
    qc_dir = os.path.join(args.out, "qc")
    os.makedirs(qc_dir, exist_ok=True)
    qc_tab = None
    if args.qc:
        qc_tab, run_tab, _, _ = run_quality_control(inputs, normalisation=normalisation,
                                                    threshold=args.outlier_threshold)
        qc_tab.to_csv(os.path.join(qc_dir, f"{layer}_sample_qc.csv"), index=False)
        run_tab.to_csv(os.path.join(qc_dir, f"{layer}_run_qc.csv"), index=False)
        flagged = qc_tab[qc_tab["outlier"]]
        bad_runs = run_tab[run_tab["outlier"]]
        log(f"\nQC ({layer}): {len(flagged)} of {len(qc_tab)} samples flagged, "
            f"{len(bad_runs)} runs disagreeing with their group "
            f"(robust |z| > {args.outlier_threshold:g})")
        for _, r in flagged.iterrows():
            log(f"  sample {r['sample']} ({r['group']}): {r['flags']}")
        for _, r in bad_runs.iterrows():
            log(f"  run {r['run']}: correlation with group profile "
                f"{r['group_profile_correlation']:.3f} (replicate {r['replicate_correlation']:.3f})")
        if fig_dir:
            from .figures import plot_sample_qc, save_figure
            import matplotlib.pyplot as plt
            fig = plot_sample_qc(qc_tab, args.outlier_threshold, font_size=args.font_size,
                                 title=f"Sample QC ({layer}): {len(flagged)} flagged")
            save_figure(fig, os.path.join(qc_dir, f"{layer}_sample_qc"), formats, args.dpi)
            plt.close(fig)
        from .qc_extended import run_extended_qc
        experiment = "phospho" if layer == "site" else ("proteome" if args.type == "phospho"
                                                        else args.type)
        try:
            _, msgs = run_extended_qc(
                inputs, layer, qc_dir, normalisation=normalisation,
                threshold=args.outlier_threshold, experiment=experiment,
                bait=args.bait if experiment == "ip" else None,
                peptide_path=args.peptide_data if layer == "site" else None,
                genes=_gene_symbols(inputs.proteins, inputs.feature_names),
                figures=bool(fig_dir), formats=formats, dpi=args.dpi,
                font_size=args.font_size)
        except ValueError as exc:                    # e.g. bait not found
            msgs = [f"extended QC: {exc}"]
        if msgs:
            log(f"Further QC notes ({layer}):")
            for m in msgs:
                log(f"  {m}")

    exclude = [e for e in args.exclude.split(",") if e.strip()]
    if args.remove_outliers and qc_tab is not None and args.type != "phospho":
        auto = qc_tab.loc[qc_tab["outlier"], "sample"].tolist()
        if auto:
            log(f"  --remove-outliers: excluding {auto}")
        exclude += auto
    present = [e for e in exclude
               if e in set(inputs.sample_names) | set(map(str, inputs.run_names))]
    if args.type == "phospho":
        exclude = present                    # IDs may belong to the other layer only
    if exclude:
        keep = exclusion_mask(inputs, exclude)
        inputs = subset_inputs(inputs, keep)
        log(f"  excluded {int((~keep).sum())} runs ({layer}); "
            f"{len(inputs.sample_names)} samples remain")
        if qc_tab is not None:
            qc_tab = qc_tab[qc_tab["sample"].isin(inputs.sample_names)].reset_index(drop=True)
    return inputs, qc_tab


def _pca_options(args, inputs, qc_tab) -> dict:
    """pca_factors / pca_shape / highlight for the figures from --pca-colour-by etc."""
    table = inputs.sample_table()
    spec = args.pca_colour_by
    if spec == "auto":
        names = [c for c in ("pulldown", "treatment", "time_min") if c in table] \
            if args.type == "ip" else []
    elif spec.lower() == "none":
        names = []
    else:
        names = [c.strip() for c in spec.split(",") if c.strip()]
    missing = [n for n in names if n not in table]
    if missing:
        raise SystemExit(f"--pca-colour-by: {missing} not found; available: "
                         f"{[c for c in table.columns if c not in ('run',)]}")
    factors = {n: table[n].astype(str).tolist() for n in names
               if table[n].nunique() > 1}
    shape = None
    if args.pca_shape:
        if args.pca_shape not in table:
            raise SystemExit(f"--pca-shape {args.pca_shape!r} not found")
        shape = (args.pca_shape.replace("_", " ").capitalize(),
                 table[args.pca_shape].astype(str).tolist())
    highlight = None
    if qc_tab is not None and qc_tab["outlier"].any():
        flagged = set(qc_tab.loc[qc_tab["outlier"], "sample"])
        highlight = [s_ in flagged for s_ in inputs.sample_names]
    return {"pca_factors": factors, "pca_shape": shape, "highlight": highlight}


def _version():
    from . import __version__
    return __version__


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except SystemExit:
        raise
    except Exception as exc:                                      # readable error, then exit
        print(f"\nError: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
