"""
Command line for supervised learning on FragPipe proteomics data.

  proteomix-ml train combined_protein.tsv -m metadata.tsv -o ml_results \\
        --task classification --target condition --models lasso,elasticnet,knn,mlp
  proteomix-ml train combined_protein.tsv -m metadata.tsv --outcomes clinical.csv \\
        --task survival --time os_months --event death -o ml_survival
  proteomix-ml predict ml_results/models/lasso.pkl new_protein.tsv -m new_metadata.tsv \\
        -o predictions.csv

Data preparation (identical for train and predict): FragPipe intensities -> log2 ->
run-level normalisation -> technical replicates collapsed -> samples x proteins (NaN kept).
Inside every cross-validation fold, filtering, imputation and standardisation are fitted on
the training samples only, and every model tunes its hyperparameters by an inner CV, so
the reported performance is an honest (nested) estimate.

Outcomes: --target names a metadata field (e.g. condition, pulldown) or a column of
--outcomes (a .csv / .tsv with one row per sample; the first column, or --id-column, holds
sample IDs or run names). Survival needs --time and --event (1 = event, 0 = censored).

Outputs: cv_summary.csv, cv_folds.csv, cv_pooled.csv, oof_predictions.csv, tuned_params.csv,
selection_<model>.csv, models/<model>.pkl (trained on all samples), coefficients_<lasso>.csv,
figures/, summary.txt and run.log.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd


def build_parser():
    ap = argparse.ArgumentParser(prog="proteomix-ml", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def data_args(p):
        p.add_argument("data", help="FragPipe combined_protein file")
        p.add_argument("-m", "--metadata", required=True, help="run -> condition file")
        p.add_argument("--sample-column", type=int, default=None,
                       help="1-based metadata column with sample IDs (default: parsed)")
        p.add_argument("--quantity", default="MaxLFQ Intensity")
        p.add_argument("--normalisation", default="median_ratio",
                       choices=["median_ratio", "total", "quantile", "none"])
        p.add_argument("--run-pattern", default=None)

    t = sub.add_parser("train", help="nested cross-validation benchmark + final models",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    data_args(t)
    t.add_argument("-o", "--out", required=True)
    t.add_argument("--task", required=True, choices=["regression", "classification", "survival"])
    t.add_argument("--target", default=None, help="outcome column (regression/classification)")
    t.add_argument("--time", default=None, help="survival time column")
    t.add_argument("--event", default=None, help="survival event column (1 = event)")
    t.add_argument("--outcomes", default=None, help="outcome table (.csv/.tsv)")
    t.add_argument("--id-column", default=None, help="sample-ID column of --outcomes")
    t.add_argument("--classes", default=None,
                   help="comma-separated classes to keep (others are dropped), e.g. IP,IgG")
    t.add_argument("--models", default="lasso,elasticnet,knn,mlp")
    t.add_argument("--folds", type=int, default=5, help="outer CV folds")
    t.add_argument("--repeats", type=int, default=1, help="repeats of the outer CV")
    t.add_argument("--inner-folds", type=int, default=5, help="inner CV folds (tuning)")
    t.add_argument("--lambda-rule", default="1se", choices=["1se", "min"])
    t.add_argument("--knn-k", default="1,3,5,7,9,15", help="k values tried by inner CV")
    t.add_argument("--mlp-hidden", default="32;64,32",
                   help="architectures tried by inner CV, ';'-separated, e.g. '32;64,32'")
    t.add_argument("--mlp-epochs", type=int, default=500)
    t.add_argument("--impute", default="model", choices=["model", "min_prob", "mean", "zero"])
    t.add_argument("--min-valid", type=float, default=0.5,
                   help="keep proteins observed in at least this fraction of training samples")
    t.add_argument("--max-features", type=int, default=None,
                   help="keep the most variable proteins (training folds) after filtering")
    t.add_argument("--no-final", action="store_true", help="skip training final models")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--figures", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--formats", default="pdf,png")
    t.add_argument("--dpi", type=int, default=600)
    t.add_argument("--font-size", type=float, default=16)
    t.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)

    pr = sub.add_parser("predict", help="apply a trained model to new data")
    pr.add_argument("model", help="models/<name>.pkl from proteomix-ml train")
    data_args(pr)
    pr.add_argument("-o", "--out", required=True, help="output .csv")
    return ap


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_samples(args, logger=None):
    """FragPipe + metadata -> (samples x proteins DataFrame, sample table, inputs)."""
    from ..export import export_matrix
    from ..fragpipe import DEFAULT_RUN_PATTERN, build_inputs, read_fragpipe
    from ..metadata import read_run_metadata
    data = read_fragpipe(args.data, quantity=args.quantity, min_observed=1,
                         run_pattern=args.run_pattern or DEFAULT_RUN_PATTERN)
    sc = None if args.sample_column is None else args.sample_column - 1
    meta = read_run_metadata(args.metadata, parse_factors=True, sample_col=sc)
    key = "run" if set(meta["run"]) & set(data.runs["run"]) else "sample"
    if key == "sample":
        meta = meta.rename(columns={"run": "sample"})
    inputs = build_inputs(data, meta, key=key, group_col="condition",
                          replicate_col="sample_id" if sc is not None else "sample",
                          drop_unmatched=True)
    ex = export_matrix(inputs, method="normalised", normalisation=args.normalisation)
    return ex.values, inputs.sample_table().set_index("sample_name"), inputs


def outcomes_for(args, samples: pd.DataFrame, sample_table: pd.DataFrame, inputs):
    """Outcome vector(s) aligned to the samples; rows without an outcome are dropped."""
    cols = [args.target] if args.task != "survival" else [args.time, args.event]
    if any(c is None for c in cols):
        raise SystemExit("survival needs --time and --event; other tasks need --target")
    if args.outcomes:
        sep = "\t" if args.outcomes.lower().endswith((".tsv", ".txt")) else ","
        out = pd.read_csv(args.outcomes, sep=sep)
        idc = args.id_column or out.columns[0]
        out[idc] = out[idc].astype(str).str.strip()
        run2sample = dict(zip(inputs.run_names,
                              np.asarray(inputs.sample_names)[inputs.replicate_ids.numpy()]))
        out["_sample"] = out[idc].map(lambda v: run2sample.get(v, v))
        out = out.drop_duplicates("_sample").set_index("_sample")
        missing = [c for c in cols if c not in out]
        if missing:
            raise SystemExit(f"columns {missing} not in {args.outcomes}")
        tab = out.reindex(samples.index)[cols]
    else:
        missing = [c for c in cols if c not in sample_table]
        if missing:
            raise SystemExit(f"{missing} not in the metadata; available: "
                             f"{list(sample_table.columns)} (or use --outcomes)")
        tab = sample_table.reindex(samples.index)[cols]
    if args.task == "classification":
        tab[cols[0]] = tab[cols[0]].astype(str).str.strip()
        tab.loc[tab[cols[0]].isin(["nan", "", "None"]), cols[0]] = np.nan
        if args.classes:
            keep = [c.strip() for c in args.classes.split(",")]
            tab.loc[~tab[cols[0]].isin(keep), cols[0]] = np.nan
    else:
        for c in cols:
            tab[c] = pd.to_numeric(tab[c], errors="coerce")
    ok = tab.notna().all(1)
    return tab[ok], ok


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------
def train(args):
    from ..progress import NullProgress, Progress
    from ..runlog import log_environment, log_inputs, log_parameters, setup_run_logging
    from ..figures import save_figure
    from . import plots
    from .cv import cross_validate, train_final
    from .metrics import PRIMARY
    from .models import make_learner
    import matplotlib.pyplot as plt

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    log = setup_run_logging(args.out)
    log_environment(log)
    log_parameters(log, vars(args))
    log_inputs(log, [args.data, args.metadata, args.outcomes])
    lines = []

    def say(msg=""):
        log.info(msg)
        lines.append(msg)

    Y, stab, inputs = load_samples(args, log)
    tab, ok = outcomes_for(args, Y, stab, inputs)
    Y = Y.loc[tab.index]
    if args.task == "survival":
        y, event = tab[args.time].to_numpy(float), tab[args.event].to_numpy(int)
    else:
        y, event = tab[args.target].to_numpy(), None
        if args.task == "regression":
            y = y.astype(float)
    say(f"proteomix-ml train | task: {args.task}")
    say(f"{len(Y)} samples with outcome ({int((~ok).sum())} without), {Y.shape[1]} proteins, "
        f"{np.isnan(Y.to_numpy()).mean():.1%} missing")
    if args.task == "classification":
        vc = pd.Series(y).value_counts()
        say("Classes: " + ", ".join(f"{k} ({v})" for k, v in vc.items()))
        if vc.min() < args.folds:
            raise SystemExit(f"class {vc.idxmin()!r} has {vc.min()} samples, fewer than "
                             f"--folds {args.folds}")
    if args.task == "survival":
        say(f"Events: {int(event.sum())} of {len(event)}")

    names = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.task == "survival" and "knn" in names:
        say("knn does not support survival; skipped")
        names.remove("knn")
    kw = {"lasso": dict(rule=args.lambda_rule, inner_folds=args.inner_folds),
          "elasticnet": dict(rule=args.lambda_rule, inner_folds=args.inner_folds),
          "knn": dict(k_grid=tuple(int(k) for k in args.knn_k.split(",")),
                      inner_folds=args.inner_folds),
          "mlp": dict(grid=[{"hidden": tuple(int(h) for h in arch.split(",")), "dropout": 0.3,
                             "weight_decay": 1e-3} for arch in args.mlp_hidden.split(";")],
                      epochs=args.mlp_epochs, inner_folds=min(args.inner_folds, 3))}
    learners = {n: make_learner(n, args.task, seed=args.seed, **kw.get(n, {})) for n in names}
    pp = dict(min_valid=args.min_valid, impute=args.impute, max_features=args.max_features)
    say(f"Models: {', '.join(names)}; outer CV {args.folds} folds x {args.repeats} repeats; "
        f"inner CV {args.inner_folds} folds; imputation {args.impute}; "
        f"min_valid {args.min_valid}")

    n_steps = args.folds * args.repeats * len(names) + (0 if args.no_final else len(names))
    bar = Progress(n_steps, desc="proteomix-ml") if args.progress else NullProgress()
    cv = cross_validate(Y, y, args.task, learners, event=event, n_folds=args.folds,
                        repeats=args.repeats, preprocess=pp, seed=args.seed, progress=bar)

    summ = cv.summary()
    summ.to_csv(os.path.join(args.out, "cv_summary.csv"))
    cv.fold_metrics.to_csv(os.path.join(args.out, "cv_folds.csv"), index=False)
    cv.pooled_metrics.to_csv(os.path.join(args.out, "cv_pooled.csv"), index=False)
    cv.params.to_csv(os.path.join(args.out, "tuned_params.csv"), index=False)
    oof = pd.DataFrame(index=Y.index)
    oof["observed"] = y
    if event is not None:
        oof["event"] = event
    for m, outs in cv.oof.items():
        o = outs[0]
        if cv.classes is not None:
            for j, c in enumerate(cv.classes):
                oof[f"{m} | p_{c}"] = o[:, j]
        else:
            oof[f"{m} | {'risk' if args.task == 'survival' else 'predicted'}"] = o
    oof.to_csv(os.path.join(args.out, "oof_predictions.csv"))
    for m, s in cv.selection.items():
        s.rename("selection_frequency").to_csv(os.path.join(args.out, f"selection_{m}.csv"))

    key = PRIMARY[args.task]
    say(f"\nNested CV performance ({key}; mean ± SD over folds | pooled out-of-fold):")
    for m, r in summ.iterrows():
        say(f"  {m:12s} {r[f'{key}_mean']:.3f} ± {r[f'{key}_std']:.3f} | "
            f"{r[f'{key}_pooled']:.3f}")
    from .metrics import HIGHER_IS_BETTER
    best = (summ[f"{key}_pooled"].idxmax() if HIGHER_IS_BETTER[key]
            else summ[f"{key}_pooled"].idxmin())
    say(f"Best (pooled {key}): {best}")

    fig_dir = os.path.join(args.out, "figures") if args.figures else None
    formats = tuple(s.strip() for s in args.formats.split(",") if s.strip())

    def save(fig, name):
        save_figure(fig, os.path.join(fig_dir, name), formats, args.dpi)
        plt.close(fig)

    if fig_dir:
        fs = args.font_size
        save(plots.plot_model_comparison(cv, font_size=fs), "model_comparison")
        if args.task == "classification":
            if len(cv.classes) == 2:
                save(plots.plot_roc(cv, font_size=fs), "roc")
            save(plots.plot_confusion(cv, best, font_size=fs), f"confusion_{best}")
        elif args.task == "regression":
            save(plots.plot_pred_vs_obs(cv, best, font_size=fs, outcome=args.target),
                 f"predicted_vs_observed_{best}")
        else:
            for m in cv.oof:
                save(plots.plot_km_risk(cv, m, font_size=fs, time_label=args.time),
                     f"km_risk_groups_{m}")
        for m in cv.selection:
            save(plots.plot_selection(cv, m, font_size=fs), f"selection_{m}")

    if not args.no_final:
        os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
        for n_ in names:
            tm = train_final(Y, y, args.task, make_learner(n_, args.task, seed=args.seed,
                                                           **kw.get(n_, {})),
                             event=event, preprocess=pp, seed=args.seed)
            tm.info.update(quantity=args.quantity, normalisation=args.normalisation,
                           task=args.task, target=args.target, time=args.time,
                           event=args.event)
            tm.save(os.path.join(args.out, "models", f"{n_}.pkl"))
            bar.step(f"final {n_}")
            if n_ in ("lasso", "elasticnet"):
                coef = tm.learner.coefficients()
                names_f = tm.preprocessor.feature_names_
                cdf = pd.DataFrame(coef.T, index=names_f,
                                   columns=[f"coef_{c}" for c in cv.classes]
                                   if coef.shape[0] > 1 else ["coefficient"])
                cdf = cdf[(cdf != 0).any(axis=1)]
                cdf.to_csv(os.path.join(args.out, f"coefficients_{n_}.csv"))
                say(f"Final {n_}: {len(cdf)} proteins selected "
                    f"(lambda {tm.learner.model_.lambda_:.4g})")
                if fig_dir:
                    save(plots.plot_lasso_cv(tm.learner.model_, font_size=args.font_size,
                                             title=f"{n_}: choice of λ (all samples)"),
                         f"lambda_cv_{n_}")
        say(f"Final models trained on all {len(y)} samples: {os.path.join(args.out, 'models')}")
    bar.close()
    say(f"\nDone in {time.time() - t0:.0f} s. Detailed log: {os.path.join(args.out, 'run.log')}")
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return cv


def predict(args):
    from .cv import TrainedModel
    tm = TrainedModel.load(args.model)
    for k in ("quantity", "normalisation"):
        if getattr(args, k) != tm.info.get(k, getattr(args, k)):
            print(f"note: model was trained with {k}={tm.info[k]!r}; using {getattr(args, k)!r}")
    Y, _, _ = load_samples(args)
    missing = set(tm.preprocessor.feature_names_) - set(Y.columns)
    if missing:
        print(f"note: {len(missing)} of the model's proteins are absent from the new data; "
              "they are imputed")
    pred = tm.predict(Y)
    pred.to_csv(args.out)
    print(f"Predictions for {len(pred)} samples written to {args.out}")
    return pred


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        train(args) if args.command == "train" else predict(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"\nError: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
