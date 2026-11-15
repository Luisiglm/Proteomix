"""
Per-feature selection-model regression with frozen sigmoid missingness.

For each feature, y_i = x_i^T beta + eps, eps ~ N(0, sigma^2), fitted by Newton-Raphson on

    observed y_i:  N(y_i | mu_i, sigma^2) * (1 - P(miss | y_i))
    missing  y_i:  P(miss | mu_i, sigma) = integral N(y | mu_i, sigma^2) P(miss | y) dy

With a probit dropout curve this marginal is a normal CDF. Here the CDF is replaced by its
logistic approximation (MacKay's probit approximation, Phi(x) ~ sigmoid(x / lambda),
lambda^2 = pi/8), giving the closed form

    P(miss | mu_i, sigma) ~ sigmoid( w_b (mu_i - r_b) / sqrt(1 + (pi/8) w_b^2 sigma^2) )

No integral is evaluated: one sigmoid per missing entry, with exact analytical gradients and
Hessians. With include_sigma=False the plug-in sigmoid(w_b (mu_i - r_b)) is used instead.
The observed-entry factor (1 - P(miss | y_i)) does not depend on (beta, sigma) once w, r are
frozen, so it drops out.

Conventions: P(missing | y) = sigmoid(w_b * (y - r_b)), same as MissingParametersNR.
With one-hot cell-means coding in X, beta are group means and contrasts are group differences.
missingness="ignore" drops the missing-value term, so missing entries are simply left out
(observed-only regression; with no missing values this is ordinary least squares).

Inference: Wald test with the ML covariance rescaled by n_obs / (n_obs - p) (the OLS/REML
small-sample correction) and a t distribution with n_obs - p degrees of freedom; optionally
limma-style empirical-Bayes variance moderation (see moderation.py).
"""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from .moderation import fit_variance_prior, moderate_variances
from .testing import benjamini_hochberg

LOG_2PI = math.log(2.0 * math.pi)
TAU_MIN = math.log(1e-4)   # floor on log sigma: keeps zero-variance features (e.g. after
                           # constant imputation) from driving the likelihood to infinity


class DifferentialAbundanceNR(nn.Module):
    """
    Per-feature linear model with sigmoid missing-not-at-random likelihood.
    Parameters per feature: theta = (beta_1..beta_p, tau = log sigma). Fitted for all features
    at once with batched Newton-Raphson, eigenvalue-shifted Hessian and per-feature step-halving.
    """

    def __init__(self, w: torch.Tensor | None = None, r: torch.Tensor | None = None,
                 include_sigma: bool = True, beta_prior_var: float | None = 100.0,
                 missingness: str = "sigmoid"):
        """
        w, r: (num_batches,) frozen missingness parameters (not needed if missingness="ignore").
        missingness: "sigmoid" (selection model) or "ignore" (missing entries are dropped).
        include_sigma: use the sigma-corrected sigmoid (recommended); False = plug-in sigmoid.
        beta_prior_var: variance of a weak Gaussian prior on each beta, centred at the feature's
            observed mean. Keeps groups with all values missing finite; None disables it.
        """
        super().__init__()
        if missingness not in ("sigmoid", "ignore"):
            raise ValueError("missingness must be 'sigmoid' or 'ignore'")
        if missingness == "sigmoid" and (w is None or r is None):
            raise ValueError("w and r are required when missingness='sigmoid'")
        self.missingness = missingness
        w = torch.zeros(1) if w is None else w
        r = torch.zeros(1) if r is None else r
        self.register_buffer("w", w.detach().double().flatten().clone())
        self.register_buffer("r", r.detach().double().flatten().clone())
        self.include_sigma = include_sigma
        self.beta_prior_var = beta_prior_var
        self.ll_history: list[float] = []

    # ---- likelihood, gradient, Hessian ------------------------------------------------
    def _terms(self, beta, tau):
        """
        beta: (F, p), tau: (F,). Returns per-feature ll (F,), gradient (F, d), Hessian (F, d, d)
        with d = p + 1, ordered (beta, tau).
        """
        X, y0, obs = self._X, self._y0, self._obs
        mis_n, mis_f = self._mis_n, self._mis_f
        N, Fn = y0.shape

        mu = X @ beta.T + self._offset                    # (N, F)
        inv_s2 = torch.exp(-2.0 * tau)                    # (F,)
        sigma = torch.exp(tau)

        # Observed entries: log N(y | mu, sigma^2)
        res = (y0 - mu) * obs
        ll = obs * (-0.5 * LOG_2PI - tau) - 0.5 * res ** 2 * inv_s2
        l_m = res * inv_s2
        l_mm = -obs * inv_s2
        l_t = obs * (-1.0) + res ** 2 * inv_s2
        l_tt = -2.0 * res ** 2 * inv_s2
        l_mt = -2.0 * res * inv_s2

        # Missing entries: log sigmoid(A), A = w (mu - r) * kappa, kappa = (1 + k)^(-1/2),
        # k = (pi/8) w^2 sigma^2. With rho = k / (1 + k):
        #   A_mu = w kappa, A_tau = -A rho, A_mu,mu = 0, A_mu,tau = -A_mu rho,
        #   A_tau,tau = A rho (3 rho - 2)
        if mis_n.numel() > 0 and self.missingness == "sigmoid":
            wi = self.w[self._batch[mis_n]]               # (M,)
            ri = self.r[self._batch[mis_n]]
            mu_m = mu[mis_n, mis_f]
            if self.include_sigma:
                k = (math.pi / 8.0) * wi ** 2 * sigma[mis_f] ** 2
                rho = k / (1.0 + k)
                kappa = torch.rsqrt(1.0 + k)
            else:
                rho = torch.zeros_like(wi)
                kappa = torch.ones_like(wi)
            A = wi * (mu_m - ri) * kappa
            A_m = wi * kappa
            A_t = -A * rho
            A_mt = -A_m * rho
            A_tt = A * rho * (3.0 * rho - 2.0)
            s = torch.sigmoid(A)
            s1 = 1.0 - s                                  # d log sigmoid / dA
            v = s * s1                                    # -d2 log sigmoid / dA2

            for arr, val in ((ll, F.logsigmoid(A)),
                             (l_m, s1 * A_m),
                             (l_t, s1 * A_t),
                             (l_mm, -v * A_m ** 2),
                             (l_mt, -v * A_m * A_t + s1 * A_mt),
                             (l_tt, -v * A_t ** 2 + s1 * A_tt)):
                arr[mis_n, mis_f] = val

        ll_f = ll.sum(0)
        g = torch.cat([torch.einsum("np,nf->fp", X, l_m), l_t.sum(0)[:, None]], dim=1)
        H_bb = torch.einsum("np,nq,nf->fpq", X, X, l_mm)
        H_bt = torch.einsum("np,nf->fp", X, l_mt)
        H_tt = l_tt.sum(0)

        if self.beta_prior_var is not None:
            v = self.beta_prior_var
            dev = beta - self._prior_mean[:, None]
            ll_f = ll_f - 0.5 * (dev ** 2).sum(1) / v
            g[:, :-1] -= dev / v
            H_bb = H_bb - torch.eye(beta.shape[1], dtype=beta.dtype, device=beta.device) / v

        p = beta.shape[1]
        H = torch.zeros(Fn, p + 1, p + 1, dtype=beta.dtype, device=beta.device)
        H[:, :p, :p] = H_bb
        H[:, :p, p] = H_bt
        H[:, p, :p] = H_bt
        H[:, p, p] = H_tt
        return ll_f, g, H

    # ---- fitting -----------------------------------------------------------------------
    @torch.no_grad()
    def fit(self, y: torch.Tensor, x: torch.Tensor, batch_ids: torch.Tensor,
            max_iter: int = 100, tol: float = 1e-9, max_halvings: int = 30,
            min_obs: int | None = None, verbose: bool = False,
            offset: torch.Tensor | None = None, keep: torch.Tensor | None = None):
        """
        y: (N, F) log2 (normalised) intensities with NaN for missing.
        x: (N, p) design. batch_ids: (N,) long.
        min_obs: features with fewer observed values are not fitted (NaN results);
                 default p + 2, so sigma has at least one residual degree of freedom.
        offset: optional (N, F) known per-sample offsets (no NaN), added to the mean:
                mu = x beta + offset. Used for protein-adjusted PTM analysis, where the
                offset is the protein's log2 abundance in each sample; beta is then the
                change in the site / protein ratio, while missingness is still evaluated on
                the site's own intensity scale (mu).
        keep:   optional (F,) bool; features with keep=False are not fitted (NaN results).
        """
        y = y.double()
        X = x.double()
        N, Fn = y.shape
        p = X.shape[1]
        d = p + 1
        min_obs = p + 2 if min_obs is None else min_obs

        obs = (~torch.isnan(y)).double()
        n_obs = obs.sum(0)
        self.fitted = n_obs >= min_obs                     # (F,)
        if keep is not None:
            self.fitted &= torch.as_tensor(keep, dtype=torch.bool)
        fit_idx = torch.nonzero(self.fitted).squeeze(1)

        y0 = torch.nan_to_num(y, nan=0.0)[:, fit_idx]
        obs_f = obs[:, fit_idx]
        if offset is None:
            off_full = torch.zeros_like(y)
        else:
            off_full = torch.as_tensor(offset, dtype=y.dtype).clone()
            if torch.isnan(off_full[:, fit_idx]).any():
                raise ValueError("offset contains NaN for features that are fitted")
            off_full = torch.nan_to_num(off_full, nan=0.0)
        self._offset = off_full[:, fit_idx]
        self.offset_input = off_full                        # kept for residuals downstream
        self._X, self._y0, self._obs = X, y0, obs_f
        self._batch = batch_ids.long()
        mis = obs_f == 0
        self._mis_n, self._mis_f = torch.nonzero(mis, as_tuple=True)
        y_adj = (y0 - self._offset) * obs_f                 # observed values minus offset
        self._prior_mean = y_adj.sum(0) / obs_f.sum(0)

        # Initialise: masked least squares (ridge-pulled to the feature mean for empty groups)
        lam = 1e-3
        XtWX = torch.einsum("np,nq,nf->fpq", X, X, obs_f) + lam * torch.eye(p, dtype=y.dtype)
        XtWy = torch.einsum("np,nf->fp", X, y_adj) + lam * self._prior_mean[:, None]
        beta = torch.linalg.solve(XtWX, XtWy.unsqueeze(-1)).squeeze(-1)
        res = (y_adj - X @ beta.T) * obs_f
        df0 = (obs_f.sum(0) - p).clamp(min=1.0)
        tau = (0.5 * torch.log(((res ** 2).sum(0) / df0).clamp(min=1e-8))).clamp(min=TAU_MIN)

        ll, g, H = self._terms(beta, tau)
        self.ll_history = [ll.sum().item()]
        eye = torch.eye(d, dtype=y.dtype)

        for it in range(max_iter):
            # Ascent direction: solve (-H + shift I) delta = g, shift makes -H positive definite
            negH = -H
            eig_min = torch.linalg.eigvalsh(negH)[:, 0]
            scale = negH.diagonal(dim1=1, dim2=2).abs().amax(1).clamp(min=1.0)
            shift = (1e-8 * scale - eig_min).clamp(min=0.0)
            delta = torch.linalg.solve(negH + shift[:, None, None] * eye,
                                       g.unsqueeze(-1)).squeeze(-1)

            step = torch.ones(len(fit_idx), dtype=y.dtype)
            for _ in range(max_halvings):
                theta_b = beta + step[:, None] * delta[:, :p]
                theta_t = (tau + step * delta[:, p]).clamp(min=TAU_MIN)
                ll_new, g_new, H_new = self._terms(theta_b, theta_t)
                worse = ~(ll_new >= ll - 1e-12 * ll.abs())   # also catches NaN
                if not worse.any():
                    break
                step = torch.where(worse, step / 2, step)
            ok = ~worse
            beta = torch.where(ok[:, None], theta_b, beta)
            tau = torch.where(ok, theta_t, tau)
            ll = torch.where(ok, ll_new, ll)
            g = torch.where(ok[:, None], g_new, g)
            H = torch.where(ok[:, None, None], H_new, H)

            prev, cur = self.ll_history[-1], ll.sum().item()
            self.ll_history.append(cur)
            if verbose:
                print(f"iter {it + 1:3d}  ll = {cur:.6f}  max|grad| = {g.abs().max().item():.3e}")
            if abs(cur - prev) <= tol * max(1.0, abs(prev)):
                break

        # Covariance = inverse observed information at the optimum; NaN where -H is not PD
        negH = -H
        L, info = torch.linalg.cholesky_ex(negH)
        cov = torch.full_like(negH, torch.nan)
        pd_ok = info == 0
        if pd_ok.any():
            cov[pd_ok] = torch.cholesky_inverse(L[pd_ok])

        # Store full-size results (NaN for unfitted features)
        def full(t, shape):
            out = torch.full(shape, torch.nan, dtype=y.dtype)
            out[fit_idx] = t
            return out

        self.beta = full(beta, (Fn, p))
        self.sigma = full(torch.exp(tau), (Fn,))
        self.cov = full(cov, (Fn, d, d))
        # Small-sample correction: ML divides by n_obs, OLS/REML by n_obs - p
        self.cov = self.cov * (n_obs / (n_obs - p).clamp(min=1.0))[:, None, None]
        self.s2 = self.sigma ** 2 * n_obs / (n_obs - p).clamp(min=1.0)   # unbiased-scale variance
        # model-based mean abundance on the measured scale (trend covariate for moderation)
        self.amean = self.beta @ X.mean(0) + off_full.mean(0)
        self.grad_norm = full(g.norm(dim=1), (Fn,))
        self.n_obs = n_obs
        self.df_resid = (n_obs - p).clamp(min=1.0)
        del self._X, self._y0, self._obs, self._mis_n, self._mis_f, self._prior_mean, self._offset
        return self

    # ---- testing -----------------------------------------------------------------------
    def test(self, contrast, feature_names=None, use_t: bool = True, moderate: bool = False,
             trend: bool = True, robust: bool = False) -> pd.DataFrame:
        """
        Wald test of H0: c^T beta = 0 for each feature.
        contrast: (p,) vector, or a tuple (g1, g0) meaning beta[g1] - beta[g0].
        use_t: t distribution with df = n_obs - p; else normal.
        moderate: limma-style empirical-Bayes variance moderation; df become n_obs - p + d0.
        trend: let the prior variance depend on mean abundance (only with moderate=True).
        robust: protect the prior from outlier variances (only with moderate=True).
        After a moderated test the fitted prior is in self.var_prior.
        """
        p = self.beta.shape[1]
        if isinstance(contrast, tuple):
            c = torch.zeros(p, dtype=torch.float64)
            c[contrast[0]], c[contrast[1]] = 1.0, -1.0
        else:
            c = torch.as_tensor(contrast, dtype=torch.float64)
        est = self.beta @ c
        var = torch.einsum("p,fpq,q->f", c, self.cov[:, :p, :p], c).numpy()
        df = self.df_resid.numpy().copy()
        s2_post = self.s2.numpy().copy()

        if moderate:
            ok = self.fitted.numpy() & np.isfinite(var)
            s2 = self.s2.numpy()[ok]
            d = df[ok]
            prior = fit_variance_prior(s2, d, covariate=self.amean.numpy()[ok] if trend else None,
                                       robust=robust)
            post, df_total = moderate_variances(s2, d, prior)
            var[ok] = var[ok] * post / s2
            df[ok] = df_total
            s2_post[ok] = post
            self.var_prior = prior

        se = np.sqrt(np.clip(var, 0, None))
        stat = est.numpy() / se
        pval = 2 * (stats.t.sf(np.abs(stat), df) if use_t else stats.norm.sf(np.abs(stat)))

        out = pd.DataFrame({
            "estimate": est.numpy(),
            "se": se,
            "stat": stat,
            "df": df if use_t else np.inf,
            "pvalue": pval,
            "padj": benjamini_hochberg(pval),
            "sigma": np.sqrt(s2_post),
            "amean": self.amean.numpy(),
            "n_obs": self.n_obs.numpy().astype(int),
            "fitted": self.fitted.numpy(),
        }, index=feature_names)
        return out
