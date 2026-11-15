"""
Export of a normalised, complete matrix for downstream analyses (machine learning,
clustering, correlation networks ...).

method
    normalised   normalised log2 values, missing values left as NaN
    model        model-based (selection model) values for the missing entries: the expected
                 log2 intensity given that the value was NOT detected,
                     E[y | missing] = integral y N(y | mu, sigma^2) sigmoid(w (y - r)) dy
                                      / integral N(y | mu, sigma^2) sigmoid(w (y - r)) dy
                 with mu and sigma from the protein's regression fit and (w, r) the fitted
                 missingness curve of the sample's batch (Gauss-Hermite quadrature, exact for
                 the sigmoid). This sits below mu by an amount that depends on how likely
                 detection was, consistent with the model used for testing.
    model_draw   a random draw from the same conditional distribution (rejection sampling).
                 Keeps the spread of the data, which conditional means shrink; use several
                 seeds for multiple imputation.
    zero, min_prob, mean, mean_noise, mean_group, knn
                 the imputation methods of imputation.py

design (model methods only)
    none    mu is each protein's overall level (intercept-only regression): imputed values
            carry no information about the sample's group. Use this when the downstream model
            predicts the group (classification), otherwise labels leak into the features.
    groups  mu is the protein's level in the sample's group (the testing model). Closer to
            the truth, but only appropriate when group labels are not the prediction target.
The missingness curve (two numbers per batch) is always fitted with the group design; this
leaks a negligible amount of label information.
mean_group imputation uses group labels in the same way as design="groups".

level
    sample  one row per biological sample (technical replicates collapsed), the default
    run     one row per run; missing entries get the value of their sample (technical
            variation is not modelled)

The imputed mask (True = value was imputed) is returned and written alongside the matrix;
keep it, since many downstream methods benefit from knowing which values were measured.
"""
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .imputation import IMPUTATIONS, impute
from .missingness import MissingParametersNR
from .model import DifferentialAbundanceNR
from .normalisation import normalise
from .replicates import collapse_replicates

EXPORT_METHODS = ("normalised", "model", "model_draw") + IMPUTATIONS


@dataclass
class ExportResult:
    values: pd.DataFrame        # rows: samples (or runs), columns: features; log2
    imputed: pd.DataFrame       # same shape, True where the value was filled in
    samples: pd.DataFrame       # row annotation (sample, group, batch, ...)
    info: dict


def conditional_missing_mean(mu, sigma, w, r, n_nodes: int = 40):
    """E[y | missing] for y ~ N(mu, sigma^2), P(missing | y) = sigmoid(w (y - r)), elementwise."""
    x, wt = np.polynomial.hermite.hermgauss(n_nodes)
    mu, sigma, w, r = (np.asarray(v, dtype=float)[..., None] for v in (mu, sigma, w, r))
    y = mu + math.sqrt(2.0) * sigma * x
    lp = -np.logaddexp(0.0, -w * (y - r)) + np.log(wt)      # log weight * sigmoid
    lp -= lp.max(-1, keepdims=True)
    q = np.exp(lp)
    return (q * y).sum(-1) / q.sum(-1)


def draw_missing(mu, sigma, w, r, rng, max_rounds: int = 200):
    """Draws from N(mu, sigma^2) tilted by sigmoid(w (y - r)) (rejection sampling)."""
    mu, sigma, w, r = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (mu, sigma, w, r)))
    out = np.full(mu.shape, np.nan)
    todo = np.ones(mu.shape, dtype=bool)
    for _ in range(max_rounds):
        if not todo.any():
            break
        y = rng.normal(mu[todo], sigma[todo])
        acc = rng.random(y.shape) < 1.0 / (1.0 + np.exp(-w[todo] * (y - r[todo])))
        idx = np.flatnonzero(todo)
        out.flat[idx[acc]] = y[acc]
        todo.flat[idx[acc]] = False
    if todo.any():                                  # extremely unlikely detections: use mean
        out[todo] = conditional_missing_mean(mu[todo], sigma[todo], w[todo], r[todo])
    return out


def export_matrix(inputs, method: str = "model", normalisation: str = "median_ratio",
                  design: str = "none", level: str = "sample", seed: int = 0,
                  collapse: str = "mean", include_sigma: bool = True,
                  drop_unfitted: bool = True) -> ExportResult:
    """
    inputs: ProteomixInputs. Returns an ExportResult (see module docstring for the options).
    drop_unfitted: proteins the model could not fit (too few observed values) are dropped
        for the model methods (otherwise they stay NaN).
    """
    if method not in EXPORT_METHODS:
        raise ValueError(f"method must be one of {EXPORT_METHODS}")
    if design not in ("none", "groups"):
        raise ValueError("design must be 'none' or 'groups'")
    if level not in ("sample", "run"):
        raise ValueError("level must be 'sample' or 'run'")
    from .pipeline import _group_labels

    Y = inputs.Y.double()
    X = inputs.X.double()
    b = inputs.batch_ids.long()
    Yn, _ = normalise(Y, method=normalisation)
    Ys, Xs, bs, _, _ = collapse_replicates(Yn, X, b, inputs.replicate_ids, method=collapse)
    run_to_sample = inputs.replicate_ids.numpy()
    features = np.asarray(inputs.feature_names)
    info = {"method": method, "normalisation": normalisation, "design": design,
            "level": level, "seed": seed}

    if method == "normalised":
        vals_s = Ys.numpy()
        keep = np.ones(len(features), dtype=bool)
    elif method in IMPUTATIONS:
        if method == "mean_group":
            warnings.warn("mean_group imputation uses group labels (label leakage for "
                          "classification)")
        vals_s = impute(Ys, method, x=Xs, seed=seed).numpy()
        keep = np.ones(len(features), dtype=bool)
    else:
        miss = MissingParametersNR(int(bs.max()) + 1).double()
        miss.fit(Ys, Xs, bs)
        w = miss.w.detach().squeeze(1).numpy()
        r = miss.r.detach().squeeze(1).numpy()
        Xd = Xs if design == "groups" else torch.ones(Ys.shape[0], 1, dtype=torch.float64)
        model = DifferentialAbundanceNR(torch.tensor(w), torch.tensor(r),
                                        include_sigma=include_sigma)
        model.fit(Ys, Xd, bs)
        keep = model.fitted.numpy()
        mu = Xd.numpy() @ model.beta.numpy().T                       # (S, F)
        sd = np.sqrt(model.s2.numpy())[None, :] * np.ones_like(mu)
        wb, rb = w[bs.numpy()][:, None] * np.ones_like(mu), r[bs.numpy()][:, None] * np.ones_like(mu)
        vals_s = Ys.numpy().copy()
        m = np.isnan(vals_s) & keep[None, :]
        if method == "model":
            vals_s[m] = conditional_missing_mean(mu[m], sd[m], wb[m], rb[m])
        else:
            vals_s[m] = draw_missing(mu[m], sd[m], wb[m], rb[m], np.random.default_rng(seed))
        info.update(missingness_w=w.tolist(), missingness_r=r.tolist(),
                    proteins_not_fitted=int((~keep).sum()))

    if level == "sample":
        measured = ~np.isnan(Ys.numpy())
        values = vals_s
        rows = inputs.sample_table()
        index = list(inputs.sample_names)
        groups = _group_labels(Xs, inputs.group_names)
    else:
        runs = Yn.numpy().copy()
        measured = ~np.isnan(runs)
        fill = vals_s[run_to_sample]
        runs[~measured] = fill[~measured]
        values = runs
        rows = inputs.runs.copy()
        index = list(inputs.run_names)
        groups = _group_labels(X, inputs.group_names)

    cols = keep if (method in ("model", "model_draw") and drop_unfitted) else \
        np.ones(len(features), dtype=bool)
    vals = pd.DataFrame(values[:, cols], index=index, columns=features[cols])
    mask = pd.DataFrame(~measured[:, cols] & ~np.isnan(values[:, cols]), index=index,
                        columns=features[cols])
    rows = rows.reset_index(drop=True)
    rows.insert(0, "group", groups)
    rows.index = index
    info.update(n_rows=vals.shape[0], n_features=vals.shape[1],
                imputed_fraction=float(mask.to_numpy().mean()),
                remaining_missing=int(vals.isna().to_numpy().sum()))
    return ExportResult(values=vals, imputed=mask, samples=rows, info=info)


def write_export(res: ExportResult, out_dir: str, prefix: str = "matrix") -> list:
    """matrix.csv (rows x features), matrix_imputed_mask.csv, matrix_samples.csv, README."""
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, f"{prefix}.csv"),
             os.path.join(out_dir, f"{prefix}_imputed_mask.csv"),
             os.path.join(out_dir, f"{prefix}_samples.csv"),
             os.path.join(out_dir, f"{prefix}_README.txt")]
    res.values.to_csv(paths[0])
    res.imputed.astype(int).to_csv(paths[1])
    res.samples.to_csv(paths[2])
    with open(paths[3], "w", encoding="utf-8") as fh:
        fh.write(__doc__.strip() + "\n\nSettings of this export:\n")
        fh.write(json.dumps(res.info, indent=2))
    return paths
