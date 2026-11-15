"""
Planning and interpretation on simulated data that resembles your experiment.

  proteomix-plan power -o plan --effect 1 --target-power 0.8 \\
        --data combined_protein.tsv -m metadata.tsv
      -> biological samples per group needed to detect a 2-fold change (log2 FC = 1) with
         80% power at 5% FDR, in data with your variance and missingness

  proteomix-plan interpret -o check --data combined_protein.tsv -m metadata.tsv \\
        --normalisation none --min-valid-per-group 3 --prop-de 0.1 --effect-dist uniform --effect 2
      -> how your analysis settings perform on simulated data of your design: how often a hit
         is real, how often real changes are missed (by abundance and effect size), bias

-t/--type chooses both the simulated data and the analysis, matching the proteomix command:
  proteome  two groups; contrast group 1 vs group 0
  ip        IgG and IP, each Ctrl / Treatment; interactors enriched in the IP (--prop-
            interactors, --enrichment), some responding to the treatment in the IP only
            (--prop-responsive, effect from --effect / --effect-dist); evaluated contrasts:
            IP vs IgG (Ctrl), treatment in IP, IgG-corrected treatment
  phospho   sites with matched proteins, paired protein-offset analysis; the effect is the
            change in site stoichiometry (--prop-de of sites); --prop-protein-change of
            proteins change, and sites that only follow their protein count as false hits
Any other simulation setting: --set name=value (see settings.json of a run for the names).

Settings, in increasing priority: built-in defaults < calibration from --data (if given) <
--config file (JSON with "simulation" and "analysis" sections) < command-line flags.
Without --data the defaults describe a typical label-free experiment (4 samples per group,
~25% missing values); set --n-per-group, --missing-midpoint, --bio-sd etc. to match yours.
Outputs: tables (.csv), a figure, settings.json and summary.txt in the output folder.
"""
import argparse
import json
import os
import sys
import time


SIM_FLAGS = {"n_per_group": int, "tech_reps": int, "n_features": int, "prop_de": float,
             "effect_dist": str, "frac_up": float, "loading_sd": float,
             "group_loading_shift": float, "mcar": float, "missing_midpoint": float,
             "missing_slope": float, "bio_sd": float, "tech_sd": float}
SIM_NAMES = {"tech_reps": "n_tech_reps", "loading_sd": "sf_sd", "missing_midpoint": "r",
             "missing_slope": "w", "bio_sd": "bio_s0"}


def build_parser():
    ap = argparse.ArgumentParser(prog="proteomix-plan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("-o", "--out", required=True)
        p.add_argument("-t", "--type", default="proteome", type=str.lower,
                       choices=["ip", "proteome", "phospho"],
                       help="experiment type, giving the same default analysis settings as "
                            "the proteomix command (ip: no normalisation; proteome / phospho: "
                            "median-of-ratios)")
        d = p.add_argument_group("calibration from your data (optional)")
        d.add_argument("--data", default=None, help="FragPipe combined_protein file")
        d.add_argument("-m", "--metadata", default=None)
        d.add_argument("--quantity", default="MaxLFQ Intensity")
        d.add_argument("--sample-column", type=int, default=None)
        d.add_argument("--config", default=None, help="JSON with 'simulation' / 'analysis'")
        s = p.add_argument_group("simulated experiment")
        for k, t in SIM_FLAGS.items():
            s.add_argument("--" + k.replace("_", "-"), type=t, default=None)
        for k in ("prop_interactors", "enrichment", "prop_responsive", "prop_protein_change",
                  "shared_sd"):
            s.add_argument("--" + k.replace("_", "-"), type=float, default=None)
        s.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                       help="any simulation setting, e.g. --set stoich_sd=1.2 (repeatable)")
        s.add_argument("--effect", type=float, default=None,
                       help="log2 fold change of changed proteins: power: the fixed effect; "
                            "interpret: the mean of the fold-change distribution ('gamma', "
                            "default), the value ('fixed'), the maximum ('uniform') or the SD "
                            "('normal')")
        s.add_argument("--fold-change", type=float, default=None,
                       help="alternative to --effect, e.g. 2 for a 2-fold change")
        a = p.add_argument_group("analysis settings to evaluate")
        a.add_argument("--normalisation", default=None,
                       choices=["median_ratio", "total", "quantile", "none"])
        a.add_argument("--missing", default=None)
        a.add_argument("--no-moderation", action="store_true")
        a.add_argument("--min-valid", type=int, default=None)
        a.add_argument("--min-valid-per-group", type=int, default=None)
        a.add_argument("--min-groups", type=int, default=None)
        o = p.add_argument_group("run")
        o.add_argument("--fdr", type=float, default=0.05)
        o.add_argument("--n-sims", type=int, default=5)
        o.add_argument("--seed", type=int, default=0)
        o.add_argument("--formats", default="pdf,png")
        o.add_argument("--dpi", type=int, default=600)
        o.add_argument("--font-size", type=float, default=16)
        o.add_argument("--figures", action=argparse.BooleanOptionalAction, default=True)
        o.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)

    pw = sub.add_parser("power", help="samples per group needed for a chosen effect size",
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    common(pw)
    pw.add_argument("--target-power", type=float, default=0.8)
    pw.add_argument("--n-range", default="2,3,4,5,6,8,10,12")
    pw.add_argument("--contrast", default=None,
                    help="contrast whose power is computed (ip: 'Treatment (IP - IgG)' by "
                         "default; also 'Treatment in IP', 'IP vs IgG (Ctrl)')")
    it = sub.add_parser("interpret", help="how your settings perform on simulated data",
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    common(it)
    it.add_argument("--lfc-threshold", type=float, default=0.0,
                    help="hits also need |log2 FC| >= this (as in your own analysis)")
    return ap


def _settings(args, say):
    import math
    from .planning import SimulationSettings, calibrate
    st = SimulationSettings()
    default_norm = "none" if args.type == "ip" else "median_ratio"   # as the proteomix command
    analysis = {"normalisation": default_norm, "missing": "selection", "moderate": True}
    if args.data:
        if not args.metadata:
            raise SystemExit("--data needs -m/--metadata")
        from .fragpipe import build_inputs, read_fragpipe
        from .metadata import read_run_metadata
        data = read_fragpipe(args.data, quantity=args.quantity, min_observed=1)
        sc = None if args.sample_column is None else args.sample_column - 1
        meta = read_run_metadata(args.metadata, parse_factors=True, sample_col=sc)
        key = "run" if set(meta["run"]) & set(data.runs["run"]) else "sample"
        if key == "sample":
            meta = meta.rename(columns={"run": "sample"})
        inputs = build_inputs(data, meta, key=key, group_col="condition",
                              replicate_col="sample_id" if sc is not None else "sample",
                              drop_unmatched=True)
        norm = args.normalisation or default_norm
        st, rep = calibrate(inputs, normalisation=norm)
        say("Calibrated from your data:")
        say(f"  {rep['proteins_in_data']} proteins, {st.n_per_group} samples per group, "
            f"{st.n_tech_reps} injection(s) per sample")
        say(f"  biological SD {st.bio_s0:.2f} log2 (prior df {rep['prior_df_d0']:.1f}), "
            f"technical SD {st.tech_sd:.2f}, loading SD {st.sf_sd:.2f}")
        say(f"  missingness: midpoint {st.r:.1f} log2, slope {st.w:.2f}; missing values "
            f"{rep['observed_missing_pct']:.1f}% observed vs {rep['simulated_missing_pct']:.1f}% "
            f"simulated")
    if args.config:
        cfg = json.load(open(args.config, encoding="utf-8"))
        st.update(**cfg.get("simulation", {}))
        analysis.update(cfg.get("analysis", {}))
    flags = {SIM_NAMES.get(k, k): getattr(args, k) for k in SIM_FLAGS
             if getattr(args, k) is not None}
    for k in ("prop_interactors", "enrichment", "prop_responsive", "prop_protein_change",
              "shared_sd"):
        if getattr(args, k) is not None:
            flags[k] = getattr(args, k)
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects NAME=VALUE, got {item!r}")
        k, v = (x.strip() for x in item.split("=", 1))
        flags[k] = v
    st.update(**flags)
    effect = args.effect
    if args.fold_change is not None:
        if args.fold_change <= 0:
            raise SystemExit("--fold-change must be positive (e.g. 2 for a doubling, "
                             "0.5 for a halving)")
        effect = abs(math.log2(args.fold_change))      # 0.5 and 2 are the same size of change
    if effect is not None:
        effect = abs(effect)
        if effect < 0.05:
            raise SystemExit(f"the effect size is {effect:g} log2 (a fold change of "
                             f"{2 ** effect:.2f}), i.e. essentially no change, so there is "
                             "nothing to detect. Use e.g. --fold-change 2 (= --effect 1) or "
                             "--fold-change 1.5.")
    if effect is not None:
        st.logfc = float(effect)
    # interpret: a realistic spread of fold changes unless a distribution was chosen
    cfg_sim = json.load(open(args.config, encoding="utf-8")).get("simulation", {}) \
        if args.config else {}
    if args.command == "interpret" and args.effect_dist is None and \
            "effect_dist" not in cfg_sim:
        st.effect_dist = "gamma"
    if args.normalisation:
        analysis["normalisation"] = args.normalisation
    if args.missing:
        analysis["missing"] = args.missing
    if args.no_moderation:
        analysis["moderate"] = False
    for k in ("min_valid", "min_valid_per_group", "min_groups"):
        if getattr(args, k) is not None:
            analysis[k] = getattr(args, k)
    return st, analysis


def main(argv=None):
    from dataclasses import asdict
    import matplotlib.pyplot as plt
    from .figures import save_figure
    from .planning import interpret, plot_interpretation, plot_power_curve, power_analysis
    from .progress import NullProgress, Progress
    from .runlog import log_environment, log_parameters, setup_run_logging

    args = build_parser().parse_args(argv)
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    log = setup_run_logging(args.out)
    log_environment(log)
    log_parameters(log, vars(args))
    lines = []

    def say(m=""):
        log.info(m)
        lines.append(m)

    try:
        st, analysis = _settings(args, say)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    say(f"Experiment type: {args.type}; analysis settings evaluated: {analysis}")
    formats = tuple(s.strip() for s in args.formats.split(",") if s.strip())

    if args.command == "power":
        if args.effect is None and args.fold_change is None:
            raise SystemExit("power needs --effect (log2 FC) or --fold-change: choose the "
                             "smallest change that matters biologically")
        grid = [int(n) for n in args.n_range.split(",")]
        bar = Progress(len(grid) * args.n_sims, "power") if args.progress else NullProgress()
        tab, tiers, req = power_analysis(st, st.logfc, args.target_power, args.fdr, grid,
                                         args.n_sims, analysis, args.seed, bar,
                                         experiment=args.type, contrast=args.contrast)
        bar.close()
        tab.to_csv(os.path.join(args.out, "power_by_n.csv"), index=False)
        tiers.to_csv(os.path.join(args.out, "power_by_n_and_abundance.csv"), index=False)
        from .planning import DEFAULT_CONTRAST
        what = {"proteome": f"{100 * st.prop_de:g}% of proteins changed",
                "ip": f"treatment effect of {100 * st.prop_responsive:g}% of interactors",
                "phospho": f"stoichiometry change of {100 * st.prop_de:g}% of sites"}[args.type]
        say(f"\nPower to detect |log2 FC| = {st.logfc:g} ({2 ** st.logfc:.2g}-fold) at "
            f"FDR {args.fdr:g} ({args.type}: {what}; contrast "
            f"'{args.contrast or DEFAULT_CONTRAST[args.type]}'):")
        for _, r in tab.iterrows():
            say(f"  n = {int(r['n_per_group']):2d} per group: power {r['power']:.2f}")
        if req:
            say(f"=> {req} biological samples per group reach {args.target_power:.0%} power.")
            low = tiers[(tiers["n_per_group"] == req) & (tiers["abundance_tier"] == "low")]
            if len(low) and low["power"].iloc[0] < args.target_power:
                say(f"   For the least abundant third of proteins power is "
                    f"{low['power'].iloc[0]:.2f} at that size (missing values).")
        else:
            say(f"=> {args.target_power:.0%} power is not reached with up to {max(grid)} "
                f"samples per group for this effect; consider a larger effect of interest, "
                f"more samples, or less variable / less missing data.")
        if args.figures:
            fig = plot_power_curve(tab, tiers, st.logfc, args.target_power, req,
                                   font_size=args.font_size)
            save_figure(fig, os.path.join(args.out, "power_curve"), formats, args.dpi)
            plt.close(fig)
    else:
        bar = Progress(args.n_sims, "interpret") if args.progress else NullProgress()
        res = interpret(st, analysis, args.n_sims, args.fdr, args.lfc_threshold, args.seed,
                        bar, experiment=args.type)
        bar.close()
        say(f"\nSimulated {args.type} experiment: {st.n_per_group} samples per group, "
            f"{args.n_sims} simulations; changes drawn from a {st.effect_dist} distribution "
            f"(mean |log2 FC| {st.logfc:g})")
        for contrast, out in res.items():
            slug = "".join(ch if ch.isalnum() else "_" for ch in contrast).strip("_")
            for k, v in out.items():
                if k != "proteins":
                    v.to_csv(os.path.join(args.out, f"{slug}__{k}.csv"), index=False)
            o = out["overall"].iloc[0]
            eff = out["proteins"].loc[out["proteins"]["is_de"], "true_logfc"].abs()
            say(f"\n[{contrast}]  simulated changes: median |log2 FC| {eff.median():.2f}, "
                f"10-90% range {eff.quantile(0.1):.2f}-{eff.quantile(0.9):.2f}")
            say(f"  hits per analysis: {o['mean_hits']:.0f}, of which "
                f"{100 * o['precision_share_of_hits_real']:.0f}% are real changes")
            say(f"  real changes found: {100 * o['detection_rate_real_changes_found']:.0f}% "
                f"(missed per analysis: {o['missed_real_per_analysis']:.0f})")
            if "protein_driven_sites_called" in o:
                say(f"  sites that only follow their protein, called anyway: "
                    f"{100 * o['protein_driven_sites_called']:.1f}%")
            be = out["by_effect"].assign(found=lambda t: t["detection_rate"] * t["proteins"])
            be = be.groupby("effect", observed=True)[["found", "proteins"]].sum()
            be = be[be["proteins"] >= 10]                         # skip near-empty bins
            say("  detection by true |log2 FC|: " + ", ".join(
                f"{k}: {r['found'] / r['proteins']:.0%}" for k, r in be.iterrows()))
            say("  by abundance (precision | detection | non-hits that are real | bias):")
            for _, r in out["by_abundance"].iterrows():
                say(f"    {r['abundance_tier']:4s}  {r['precision']:.2f} | "
                    f"{r['detection_rate']:.2f} | {r['non_hits_that_are_real_changes']:.3f} | "
                    f"{r['fold_change_bias']:+.2f}")
            say("  share of results that are real changes, by adjusted p-value:")
            for _, r in out["by_padj"].iterrows():
                say(f"    padj {str(r['band']):11s} {r['share_real']:.2f}  "
                    f"({int(r['proteins'])})")
            if args.figures:
                fig = plot_interpretation(out, args.fdr, font_size=args.font_size,
                                          title=f"{contrast}: what your settings detect "
                                                f"(simulated {args.type} data)")
                save_figure(fig, os.path.join(args.out, f"interpretation_{slug}"), formats,
                            args.dpi)
                plt.close(fig)
    json.dump({"simulation": asdict(st), "analysis": analysis},
              open(os.path.join(args.out, "settings.json"), "w"), indent=2, default=str)
    say(f"\nDone in {time.time() - t0:.0f} s. Settings: {os.path.join(args.out, 'settings.json')}"
        " (reusable with --config)")
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
