"""
Imputation of missing log2 intensities (applied to the collapsed, normalised matrix).

All functions take y (N, F) with NaN for missing and return a complete matrix of the same shape.
    zero        replace by 0 on the log2 scale (an extreme left-shift; mainly a reference point)
    min_prob    per sample, draws from N(q-quantile of that sample, tune_sigma * median feature
                SD), as in imputeLCMD::impute.MinProb
    mean        feature mean of the observed values
    mean_noise  feature mean + N(0, feature SD) noise
    mean_group  feature mean within the sample's group (falls back to the feature mean)
    knn         k nearest features (NaN-Euclidean distance over co-observed samples), averaged,
                as in impute::impute.knn; features with > row_max missing get the feature mean
Fallbacks: a feature with no observed value is imputed with the sample's q-quantile.
"""
import warnings

import numpy as np
import torch

IMPUTATIONS = ("zero", "min_prob", "mean", "mean_noise", "mean_group", "knn")


def _feature_stats(y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(y, 0)
        sd = np.nanstd(y, 0, ddof=1)
    return mean, sd


def _fill_empty(y, out, q=0.01):
    """Entries still missing (features with no observed value): each sample's q-quantile."""
    left = np.isnan(out)
    if left.any():
        sample_q = np.nanquantile(y, q, axis=1)
        out = np.where(left, sample_q[:, None], out)
    return out


def impute_zero(y, **_):
    return np.where(np.isnan(y), 0.0, y)


def impute_min_prob(y, q: float = 0.01, tune_sigma: float = 1.0, rng=None, **_):
    rng = np.random.default_rng(rng)
    mean, sd = _feature_stats(y)
    s = tune_sigma * np.nanmedian(sd)
    mu = np.nanquantile(y, q, axis=1)                   # per sample
    draws = rng.normal(mu[:, None], s, size=y.shape)
    return np.where(np.isnan(y), draws, y)


def impute_mean(y, **_):
    mean, _ = _feature_stats(y)
    return _fill_empty(y, np.where(np.isnan(y), mean[None, :], y))


def impute_mean_noise(y, rng=None, **_):
    rng = np.random.default_rng(rng)
    mean, sd = _feature_stats(y)
    sd = np.where(np.isfinite(sd), sd, np.nanmedian(sd))
    draws = mean[None, :] + rng.normal(size=y.shape) * sd[None, :]
    return _fill_empty(y, np.where(np.isnan(y), draws, y))


def impute_mean_group(y, x=None, **_):
    if x is None:
        raise ValueError("mean_group imputation needs the design matrix x")
    out = y.copy()
    groups = np.unique(x, axis=0)
    mean, _ = _feature_stats(y)
    for g in groups:
        rows = np.all(x == g, axis=1)
        gm, _ = _feature_stats(y[rows])
        gm = np.where(np.isfinite(gm), gm, mean)
        sub = out[rows]
        out[rows] = np.where(np.isnan(sub), gm[None, :], sub)
    return _fill_empty(y, out)


def impute_knn(y, k: int = 10, row_max: float = 0.5, block: int = 500, **_):
    """
    kNN over features. Distance between features i, j: mean squared difference over samples
    observed in both (needs >= 2 shared samples). Each missing y[s, i] is the mean of the k
    nearest features of i that are observed in sample s.
    """
    N, Fn = y.shape
    obs = ~np.isnan(y)
    A = np.where(obs, y, 0.0).T                           # (F, N)
    M = obs.T.astype(float)
    A2 = A ** 2
    out = y.copy()
    mean, _ = _feature_stats(y)
    frac_missing = 1 - M.mean(1)
    todo = np.where((frac_missing > 0) & (frac_missing <= row_max))[0]

    for start in range(0, len(todo), block):
        rows = todo[start:start + block]
        Ar, Mr, A2r = A[rows], M[rows], A2[rows]
        ss = A2r @ M.T + Mr @ A2.T - 2 * Ar @ A.T          # sum of squared diffs, co-observed
        cnt = Mr @ M.T
        with np.errstate(invalid="ignore", divide="ignore"):
            D = np.where(cnt >= 2, ss / cnt, np.inf)
        D[np.arange(len(rows)), rows] = np.inf              # exclude self
        for s in range(N):
            need = ~obs[s, rows]
            if not need.any():
                continue
            cand = np.where(obs[s])[0]
            Ds = D[need][:, cand]
            kk = min(k, len(cand))
            nn = np.argpartition(Ds, kk - 1, axis=1)[:, :kk]
            dsel = np.take_along_axis(Ds, nn, axis=1)
            vals = y[s, cand][nn]
            vals = np.where(np.isfinite(dsel), vals, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                est = np.nanmean(vals, axis=1)
            out[s, rows[need]] = est

    still = np.isnan(out)
    out = np.where(still, mean[None, :], out)               # high-missing rows / no neighbours
    return _fill_empty(y, out)


_DISPATCH = {
    "zero": impute_zero,
    "min_prob": impute_min_prob,
    "mean": impute_mean,
    "mean_noise": impute_mean_noise,
    "mean_group": impute_mean_group,
    "knn": impute_knn,
}


def impute(y: torch.Tensor, method: str, x: torch.Tensor | None = None, seed=None, **kwargs):
    """Impute an (N, F) log2 tensor; returns a tensor of the same dtype with no NaNs."""
    if method not in _DISPATCH:
        raise ValueError(f"unknown imputation {method!r}; choose from {IMPUTATIONS}")
    yn = y.detach().cpu().numpy().astype(float)
    xn = None if x is None else x.detach().cpu().numpy()
    out = _DISPATCH[method](yn, x=xn, rng=seed, **kwargs)
    return torch.as_tensor(out, dtype=y.dtype)
