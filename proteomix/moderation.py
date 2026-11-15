"""
limma-style empirical-Bayes moderation of per-feature variances.

Model (Smyth 2004): given sigma_g^2, s_g^2 ~ sigma_g^2 chi^2_{d_g} / d_g, and the true variances
follow a scaled inverse chi-square prior, 1 / sigma_g^2 ~ chi^2_{d0} / (d0 s0^2).
The hyperparameters (d0, s0^2) are estimated by the method of moments on log s_g^2 (limma's
fitFDist). The posterior variance is

    s_tilde^2 = (d0 s0^2 + d_g s_g^2) / (d0 + d_g),   with df_total = d_g + d0.

trend: s0^2 depends smoothly on mean abundance (a least-squares cubic B-spline of the adjusted
log-variances on the covariate), like limma's trend=TRUE.
robust: a simplified version of limma's robust=TRUE. Features whose variance is an outlier
under the first prior (BH-adjusted upper-tail F test < 0.05, either tail if two_sided) are
excluded, the prior is re-estimated on the rest, and the outliers get d0 = 0 (no shrinkage).
limma instead winsorises and assigns outliers a reduced, not zero, prior df.
"""
import numpy as np
from scipy import stats
from scipy.interpolate import make_lsq_spline
from scipy.special import digamma, polygamma

from .testing import benjamini_hochberg


def trigamma_inverse(x, tol: float = 1e-8, max_iter: int = 50):
    """Solve trigamma(y) = x for y > 0 (Newton iteration as in limma's trigammaInverse)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.empty_like(x)
    big, small = x > 1e7, x < 1e-6
    mid = ~(big | small)
    y[big] = 1.0 / np.sqrt(x[big])
    y[small] = 1.0 / x[small]
    ym = 0.5 + 1.0 / x[mid]
    for _ in range(max_iter):
        tri = polygamma(1, ym)
        dif = tri * (1.0 - tri / x[mid]) / polygamma(2, ym)
        ym = ym + dif
        if np.all(-dif / ym < tol):
            break
    y[mid] = ym
    return y


def _spline_fit(xv, yv, n_interior: int = 3):
    """Least-squares cubic B-spline of yv on xv, knots at quantiles; constant beyond the data."""
    order = np.argsort(xv)
    xs, ys = xv[order], yv[order]
    lo, hi = xs[0], xs[-1]
    inner = np.unique(np.quantile(xs, np.linspace(0, 1, n_interior + 2)[1:-1]))
    inner = inner[(inner > lo) & (inner < hi)]
    t = np.r_[[lo] * 4, inner, [hi] * 4]
    try:
        spl = make_lsq_spline(xs, ys, t, k=3)
        fitted = np.empty_like(yv)
        fitted[order] = spl(xs)
        n_coef = len(t) - 4
        return fitted, n_coef
    except (ValueError, np.linalg.LinAlgError):
        return np.full_like(yv, ys.mean()), 1


def fit_f_dist(s2, df, covariate=None):
    """
    limma fitFDist: estimate (d0, s0^2) from variances s2 with degrees of freedom df.
    With a covariate, s0^2 is a smooth function of it (returned per feature).
    Returns dict(d0=float, s0_2=array per feature).
    """
    s2 = np.asarray(s2, dtype=float)
    df = np.asarray(df, dtype=float)
    n = len(s2)
    s2 = np.maximum(s2, 1e-5 * np.median(s2))          # guard against zero variances
    z = np.log(s2)
    e = z - digamma(df / 2) + np.log(df / 2)

    if covariate is None or n < 10:
        emean = np.full(n, e.mean())
        n_coef = 1
    else:
        emean, n_coef = _spline_fit(np.asarray(covariate, dtype=float), e)

    evar = np.sum((e - emean) ** 2) / max(n - n_coef, 1) - np.mean(polygamma(1, df / 2))
    if evar > 0:
        d0 = 2.0 * trigamma_inverse(evar)[0]
        s0_2 = np.exp(emean + digamma(d0 / 2) - np.log(d0 / 2))
    else:
        d0 = np.inf
        s0_2 = np.exp(emean)
    return {"d0": d0, "s0_2": s0_2}


def fit_variance_prior(s2, df, covariate=None, robust: bool = False, two_sided: bool = False,
                       outlier_fdr: float = 0.05):
    """
    Fits the variance prior. Returns dict with
        d0     : per-feature prior df (array; 0 for robust outliers, inf if no extra variability)
        s0_2   : per-feature prior variance (array)
        d0_global, n_outliers, trend (bool)
    """
    s2 = np.asarray(s2, dtype=float)
    df = np.asarray(df, dtype=float)
    cov = None if covariate is None else np.asarray(covariate, dtype=float)
    prior = fit_f_dist(s2, df, cov)
    d0 = np.full(len(s2), prior["d0"])
    s0_2 = np.broadcast_to(prior["s0_2"], s2.shape).copy()
    n_out = 0

    if robust and np.isfinite(prior["d0"]):
        f = s2 / s0_2
        p_up = stats.f.sf(f, df, prior["d0"])
        p = 2 * np.minimum(p_up, 1 - p_up) if two_sided else p_up
        out = benjamini_hochberg(p) < outlier_fdr
        n_out = int(out.sum())
        if 0 < n_out < len(s2) - 10:
            keep = ~out
            prior2 = fit_f_dist(s2[keep], df[keep], None if cov is None else cov[keep])
            d0[:] = prior2["d0"]
            if cov is None:
                s0_2[:] = prior2["s0_2"]
            else:  # evaluate the re-fitted trend at every feature by interpolation
                order = np.argsort(cov[keep])
                s0_2 = np.interp(cov, cov[keep][order], np.asarray(prior2["s0_2"])[order])
            d0[out] = 0.0
            prior = prior2

    return {"d0": d0, "s0_2": s0_2, "d0_global": float(prior["d0"]), "n_outliers": n_out,
            "trend": cov is not None}


def moderate_variances(s2, df, prior):
    """Posterior variances and total df. Infinite d0 gives s0^2; df_total is capped at sum(df)."""
    s2 = np.asarray(s2, dtype=float)
    df = np.asarray(df, dtype=float)
    d0, s0_2 = prior["d0"], prior["s0_2"]
    inf = np.isinf(d0)
    d0f = np.where(inf, 0.0, d0)
    post = np.where(inf, s0_2, (d0f * s0_2 + df * s2) / np.maximum(d0f + df, 1e-12))
    df_total = np.minimum(np.where(inf, np.inf, df + d0f), df.sum())
    return post, df_total
