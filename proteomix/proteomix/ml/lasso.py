"""
Elastic-net / lasso by cyclical coordinate descent (Friedman, Hastie & Tibshirani 2010,
"Regularization paths for generalized linear models via coordinate descent"; Simon et al.
2011 for the Cox model). Objective, with standardised features:

    gaussian     (1/2n) sum (y - b0 - x b)^2              + lambda * P(b)
    binomial     -(1/n) loglik(logistic)                  + lambda * P(b)
    multinomial  -(1/n) loglik(softmax), one b per class  + lambda * sum_k P(b_k)
    cox          -(1/n) partial loglik (Breslow ties)     + lambda * P(b)
    P(b) = alpha * |b|_1 + (1 - alpha) / 2 * |b|_2^2      (alpha = 1: lasso)

Non-Gaussian families use the usual quadratic (IRLS) approximation in an outer loop, the
multinomial one class at a time ("ungrouped", as glmnet's default). Speed comes from warm
starts along a decreasing lambda path, an active set, and the sequential strong rule
(Tibshirani et al. 2012) with a KKT check on the discarded features, so only a handful of
features are updated at each lambda even with thousands of proteins.

ElasticNet.fit computes the path; cv_elastic_net chooses lambda by K-fold cross-validated
deviance (lambda_min, or lambda_1se: the largest lambda within one standard error of the
minimum, a sparser and usually more stable model). For the Cox model the cross-validated
deviance is the Verweij & van Houwelingen (1993) estimate, which uses each held-out fold
together with its training data instead of the few events in a small fold alone.
"""
import numpy as np

FAMILIES = ("gaussian", "binomial", "multinomial", "cox")


# --------------------------------------------------------------------------------------
# Weighted least-squares elastic net (the inner problem of every family)
# --------------------------------------------------------------------------------------
def _wls_path_step(X, z, w, lam, lam_prev, alpha, beta, b0, pf, fit_intercept, ever_active,
                   tol, max_passes):
    """
    Minimises (1/2) sum_i w_i (z_i - b0 - x_i b)^2 + lam * P(b), with sum(w) = 1, starting
    from (beta, b0). Uses the strong rule relative to lam_prev and a KKT check.
    Returns (beta, b0, n_passes).
    """
    xw2 = w @ (X ** 2)                                       # (p,)
    r = z - b0 - X @ beta
    grad = (w * r) @ X                                       # (p,) correlation with residual
    l1 = lam * alpha * pf
    strong = (np.abs(grad) >= alpha * pf * (2 * lam - lam_prev)) | ever_active
    passes = 0
    while True:
        S = np.flatnonzero(strong)
        while passes < max_passes:
            passes += 1
            max_change = 0.0
            for j in S:
                bj = beta[j]
                xj = X[:, j]
                g = (w * r) @ xj + xw2[j] * bj
                denom = xw2[j] + lam * (1 - alpha) * pf[j]
                new = np.sign(g) * max(abs(g) - l1[j], 0.0) / denom if denom > 0 else 0.0
                if new != bj:
                    d = new - bj
                    r -= d * xj
                    beta[j] = new
                    max_change = max(max_change, xw2[j] * d * d)
            if fit_intercept:
                d0 = (w @ r) / w.sum()
                if d0 != 0:
                    b0 += d0
                    r -= d0
                    max_change = max(max_change, d0 * d0)
            if max_change < tol:
                break
        # KKT check on features outside the strong set
        grad = (w * r) @ X
        viol = (~strong) & (np.abs(grad) > l1 * (1 + 1e-7)) & (pf > 0)
        if not viol.any() or passes >= max_passes:
            return beta, b0, passes
        strong |= viol


# --------------------------------------------------------------------------------------
# Families: working responses / weights and deviance
# --------------------------------------------------------------------------------------
def _cox_prepare(time, event):
    order = np.argsort(time, kind="stable")
    t = np.asarray(time, float)[order]
    d = np.asarray(event, float)[order]
    # index of the first observation of each tied time block (risk set start)
    first = np.searchsorted(t, t, side="left")
    last = np.searchsorted(t, t, side="right") - 1
    return order, t, d, first, last


def _cox_grad_hess(eta_sorted, d, first, last):
    """Gradient and diagonal Hessian of the Breslow log partial likelihood (sorted data)."""
    m = eta_sorted.max()
    e = np.exp(eta_sorted - m)
    risk = np.cumsum(e[::-1])[::-1]                          # sum over j with t_j >= t_i
    S = risk[first]                                          # risk-set sum at each time
    # events per tied block, attributed to the block's first index
    n = len(d)
    dk = np.zeros(n)
    np.add.at(dk, first, d)
    is_first = np.zeros(n, dtype=bool)
    is_first[np.unique(first)] = True
    a = np.where(is_first, dk / S, 0.0)                     # d_k / S_k at block starts
    b = np.where(is_first, dk / S ** 2, 0.0)
    ca, cb = np.cumsum(a), np.cumsum(b)                      # sum over k with t_k <= t_i
    A = ca[last]
    B = cb[last]
    grad = d - e * A
    hess = e * A - e ** 2 * B
    return grad, hess


def _cox_full_hessian(eta_sorted, d, first):
    """Full (n x n) Hessian of the negative Breslow log partial likelihood (sorted data)."""
    n = len(d)
    e = np.exp(eta_sorted - eta_sorted.max())
    starts = np.unique(first)
    dk = np.array([d[first == k].sum() for k in starts])
    keep = dk > 0
    starts, dk = starts[keep], dk[keep]
    M = (np.arange(n)[None, :] >= starts[:, None]).astype(float)   # risk sets (K x n)
    P = M * e
    P /= P.sum(1, keepdims=True)                                   # pi_k
    return np.diag(dk @ P) - (P.T * dk) @ P


def _cox_loglik(eta_sorted, d, first):
    m = eta_sorted.max()
    risk = np.cumsum(np.exp(eta_sorted - m)[::-1])[::-1]
    return float(np.sum(d * (eta_sorted - (np.log(risk[first]) + m))))


def deviance(family, y, eta, event=None):
    """Mean deviance (lower is better) of linear predictor(s) eta."""
    if family == "gaussian":
        return float(np.mean((y - eta) ** 2))
    if family == "binomial":
        return float(-2 * np.mean(y * eta - np.logaddexp(0, eta)))
    if family == "multinomial":
        ll = eta[np.arange(len(y)), y] - _logsumexp(eta)
        return float(-2 * np.mean(ll))
    if family == "cox":
        order, _, d, first, _ = _cox_prepare(y, event)
        return -2 * _cox_loglik(eta[order], d, first) / len(y)
    raise ValueError(family)


def _logsumexp(a):
    m = a.max(1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(1, keepdims=True))).ravel()


# --------------------------------------------------------------------------------------
# Estimator
# --------------------------------------------------------------------------------------
class ElasticNet:
    """
    Elastic-net GLM fitted along a lambda path by coordinate descent.

    family: gaussian, binomial, multinomial or cox (y = time, event = 0/1 indicator).
    alpha: 1 = lasso, 0 < alpha < 1 elastic net (alpha = 0, ridge, is allowed but then use a
        fixed lambda grid; the path is anchored at alpha = 0.001).
    n_lambda, lambda_min_ratio: path of n_lambda values from lambda_max (all coefficients 0)
        down to lambda_min_ratio * lambda_max (default 0.01 if n < p else 1e-4).
    standardize: features are scaled to unit variance internally; coefficients are returned
        on the original scale.
    penalty_factor: per-feature multiplier of lambda (0 = always in the model).
    dfmax: the path stops once this many proteins are in the model; default n when p > n
        (a lasso solution has at most n non-zero coefficients), as glmnet's dfmax / pmax.
    cox_hessian: quadratic approximation for the Cox model inside coordinate descent:
        "full" (default) uses the exact Hessian in the linear predictor, factorised as
        W = A^T A with A of rank <= n, so CD runs on the n projected rows A X; "diagonal"
        is glmnet's diagonal approximation. Both are coordinate descent over proteins; in
        tests "full" reached lower objectives at small lambda and was faster when p >> n.
    """

    def __init__(self, family: str = "gaussian", alpha: float = 1.0, n_lambda: int = 100,
                 lambda_min_ratio: float | None = None, lambdas=None, standardize: bool = True,
                 fit_intercept: bool = True, penalty_factor=None, tol: float = 1e-10,
                 max_passes: int = 10000, max_irls: int = 50, irls_tol: float = 1e-5,
                 cox_inner_tol: float = 1e-8, cox_hessian: str = "full",
                 dfmax: int | None = None):
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        self.family, self.alpha = family, float(alpha)
        self.n_lambda, self.lambda_min_ratio = n_lambda, lambda_min_ratio
        self.user_lambdas = None if lambdas is None else np.sort(np.asarray(lambdas, float))[::-1]
        self.standardize, self.fit_intercept = standardize, fit_intercept and family != "cox"
        self.penalty_factor, self.tol = penalty_factor, tol
        self.max_passes, self.max_irls, self.dfmax = max_passes, max_irls, dfmax
        self.irls_tol, self.cox_inner_tol = irls_tol, cox_inner_tol
        if cox_hessian not in ("diagonal", "full"):
            raise ValueError("cox_hessian must be 'diagonal' or 'full'")
        self.cox_hessian = cox_hessian

    # -- path --------------------------------------------------------------------------
    def fit(self, X, y, event=None):
        X = np.asarray(X, float)
        n, p = X.shape
        self.n_features_ = p
        self.mean_ = X.mean(0) if self.standardize or self.fit_intercept else np.zeros(p)
        sd = X.std(0) if self.standardize else np.ones(p)
        self.sd_ = np.where(sd > 0, sd, 1.0)
        Xs = (X - self.mean_) / self.sd_
        Xs[:, sd == 0] = 0.0
        pf = np.ones(p) if self.penalty_factor is None else np.asarray(self.penalty_factor, float)
        pf = pf * p / pf.sum()
        a = max(self.alpha, 1e-3)

        fam = self.family
        if fam in ("binomial", "multinomial"):
            self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
            if fam == "binomial" and len(self.classes_) != 2:
                raise ValueError("binomial needs exactly two classes; use multinomial")
            K = len(self.classes_) if fam == "multinomial" else 1
        else:
            yi, K = np.asarray(y, float), 1
        if fam == "cox":
            if event is None:
                raise ValueError("cox needs event")
            self._cox = _cox_prepare(yi, event)

        lam_max = self._lambda_max(Xs, yi, event, a, pf)
        if self.user_lambdas is not None:
            lams = self.user_lambdas
        else:
            ratio = self.lambda_min_ratio or (0.01 if n < p else 1e-4)
            lams = lam_max * np.logspace(0, np.log10(ratio), self.n_lambda)
        coefs = np.zeros((len(lams), K, p))
        b0s = np.zeros((len(lams), K))
        beta = np.zeros((K, p))
        b0 = self._null_intercept(yi, K)
        ever = np.zeros((K, p), dtype=bool)
        lam_prev = lam_max
        null_eta = self._eta_std(Xs, 0 * beta, self._null_intercept(yi, K))
        dev_null = deviance(fam, yi, null_eta if K > 1 else null_eta[:, 0], event)
        devs = []
        for li, lam in enumerate(lams):
            if not (li == 0 and self.user_lambdas is None):     # lambda_max: null model
                beta, b0 = self._solve(Xs, yi, event, lam, lam_prev, a, beta, b0, pf, ever, K)
            ever |= beta != 0
            coefs[li], b0s[li] = beta, b0
            eta = self._eta_std(Xs, beta, b0)
            dev = deviance(fam, yi, eta if K > 1 else eta[:, 0], event)
            devs.append(dev)
            lam_prev = lam
            nz = int((beta != 0).any(0).sum())
            # glmnet-style early stopping of the path: saturated model (more non-zero
            # coefficients than samples), almost all deviance explained, or no progress
            dfmax = self.dfmax if self.dfmax is not None else (n if n < p else p + 1)
            ratio = 1 - dev / dev_null if dev_null else 0.0
            gain = (devs[-2] - dev) / dev_null if (len(devs) > 1 and dev_null) else 1.0
            if self.user_lambdas is None and (nz >= dfmax or ratio > 0.999
                                              or (li > 5 and gain < 1e-5)):
                coefs, b0s, lams = coefs[: li + 1], b0s[: li + 1], lams[: li + 1]
                break
        # back to the original feature scale
        self.lambdas_ = np.asarray(lams)
        self.coef_path_ = coefs / self.sd_
        self.intercept_path_ = b0s - (self.coef_path_ * self.mean_).sum(-1)
        self.deviance_ratio_ = 1 - np.asarray(devs) / dev_null if dev_null else np.zeros(len(devs))
        self.lambda_ = self.lambdas_[-1]
        self._set_lambda(len(self.lambdas_) - 1)
        return self

    def _set_lambda(self, idx):
        self.lambda_index_ = idx
        self.lambda_ = self.lambdas_[idx]
        self.coef_ = self.coef_path_[idx]
        self.intercept_ = self.intercept_path_[idx]

    def set_lambda(self, lam):
        """Uses the path solution closest to lam (on the log scale)."""
        idx = int(np.argmin(np.abs(np.log(self.lambdas_) - np.log(lam))))
        self._set_lambda(idx)
        return self

    # -- prediction --------------------------------------------------------------------
    def decision_function(self, X, idx=None):
        """Linear predictor(s): (n,) or (n, K) for multinomial."""
        X = np.asarray(X, float)
        c = self.coef_ if idx is None else self.coef_path_[idx]
        b = self.intercept_ if idx is None else self.intercept_path_[idx]
        eta = X @ c.T + b
        return eta if self.family == "multinomial" else eta[:, 0]

    def predict(self, X):
        """gaussian: value; binomial / multinomial: class label; cox: risk score (eta)."""
        eta = self.decision_function(X)
        if self.family == "binomial":
            return self.classes_[(eta > 0).astype(int)]
        if self.family == "multinomial":
            return self.classes_[eta.argmax(1)]
        return eta

    def predict_proba(self, X):
        eta = self.decision_function(X)
        if self.family == "binomial":
            p1 = 1 / (1 + np.exp(-eta))
            return np.column_stack([1 - p1, p1])
        if self.family == "multinomial":
            return np.exp(eta - _logsumexp(eta)[:, None])
        raise ValueError("predict_proba is for classification")

    def selected_features(self, names=None):
        nz = np.flatnonzero((self.coef_ != 0).any(0))
        return nz if names is None else [names[i] for i in nz]

    # -- internals ---------------------------------------------------------------------
    def _eta_std(self, Xs, beta, b0):
        return Xs @ beta.T + b0

    def _null_intercept(self, y, K):
        if not self.fit_intercept:
            return np.zeros(K)
        if self.family == "gaussian":
            return np.array([y.mean()])
        if self.family == "binomial":
            m = np.clip(y.mean(), 1e-5, 1 - 1e-5)
            return np.array([np.log(m / (1 - m))])
        if self.family == "multinomial":
            f = np.clip(np.bincount(y, minlength=K) / len(y), 1e-5, 1)
            return np.log(f) - np.log(f).mean()
        return np.zeros(K)

    def _lambda_max(self, Xs, y, event, a, pf):
        n = len(y)
        if self.family == "gaussian":
            g = Xs.T @ (y - y.mean()) / n
        elif self.family == "binomial":
            g = Xs.T @ (y - y.mean()) / n
        elif self.family == "multinomial":
            K = len(self.classes_)
            Y = np.eye(K)[y]
            g = np.abs(Xs.T @ (Y - Y.mean(0)) / n).max(1)
        else:
            order, _, d, first, last = self._cox
            grad, _ = _cox_grad_hess(np.zeros(n), d, first, last)
            g = Xs[order].T @ grad / n
        ok = pf > 0
        return float(np.max(np.abs(g[ok]) / (a * pf[ok]))) if ok.any() else 1.0

    def _solve(self, Xs, y, event, lam, lam_prev, a, beta, b0, pf, ever, K):
        n = len(y)
        fam = self.family
        if fam == "gaussian":
            w = np.full(n, 1.0 / n)
            beta[0], b0[0], _ = _wls_path_step(Xs, y, w, lam, lam_prev, a, beta[0], b0[0], pf,
                                               self.fit_intercept, ever[0], self.tol,
                                               self.max_passes)
            return beta, b0
        if fam == "cox":
            # Newton steps with the full Hessian in eta (n is small in proteomics): the
            # quadratic approximation is rewritten as an ordinary least-squares problem
            # with rows W^(1/2) X, so coordinate descent applies unchanged. The diagonal
            # approximation converges only linearly when n < p.
            order, _, d, first, last = self._cox
            Xo = Xs[order]
            prev_obj = np.inf
            for _ in range(self.max_irls):
                eta = Xo @ beta[0]
                grad, hdiag = _cox_grad_hess(eta, d, first, last)
                if self.cox_hessian == "diagonal":          # glmnet's quadratic approximation
                    wd = np.maximum(hdiag, 1e-5)
                    Xt, z, wts = Xo, eta + grad / wd, wd / n
                else:
                    W = _cox_full_hessian(eta, d, first) / n
                    vals, U = np.linalg.eigh(W)
                    keep = vals > 1e-10 * max(vals.max(), 1e-300)
                    sq = np.sqrt(vals[keep])
                    A = (U[:, keep] * sq).T                     # (r, n): A^T A = W
                    z = A @ eta + (U[:, keep].T @ grad / n) / sq
                    Xt, wts = A @ Xo, np.ones(len(z))
                old = beta[0].copy()
                beta[0], _, _ = _wls_path_step(Xt, z, wts, lam, lam_prev, a,
                                               beta[0], 0.0, pf, False, ever[0],
                                               max(self.tol, self.cox_inner_tol),
                                               self.max_passes)
                obj = (-_cox_loglik(Xo @ beta[0], d, first) / n
                       + lam * (a * np.abs(beta[0]) @ pf + (1 - a) / 2 * (beta[0] ** 2) @ pf))
                change = np.max(np.abs(beta[0] - old)) / max(1.0, np.max(np.abs(beta[0])))
                if change < self.irls_tol or (_ > 0 and abs(prev_obj - obj) < 1e-8 * abs(obj)):
                    break
                prev_obj = obj
            return beta, b0
        # binomial / multinomial
        Yk = (y == 1).astype(float)[:, None] if fam == "binomial" else np.eye(K)[y]
        prev_obj = np.inf
        for it in range(self.max_irls):
            old = beta.copy()
            for k in range(K):
                eta = Xs @ beta.T + b0
                if fam == "binomial":
                    pk = 1 / (1 + np.exp(-eta[:, 0]))
                else:
                    pk = np.exp(eta[:, k] - _logsumexp(eta))
                wk = np.maximum(pk * (1 - pk), 1e-5)
                z = eta[:, k] + (Yk[:, k] - pk) / wk
                beta[k], b0[k], _ = _wls_path_step(Xs, z, wk / n, lam, lam_prev, a, beta[k],
                                                   b0[k], pf, self.fit_intercept, ever[k],
                                                   max(self.tol, self.cox_inner_tol),
                                                   self.max_passes)
            if fam == "multinomial" and self.fit_intercept:
                b0 -= b0.mean()
            eta = Xs @ beta.T + b0
            obj = deviance(fam, y, eta if K > 1 else eta[:, 0]) / 2 + lam * (
                a * np.abs(beta) @ pf + (1 - a) / 2 * (beta ** 2) @ pf).sum()
            change = np.max(np.abs(beta - old)) / max(1.0, np.max(np.abs(beta)))
            if change < self.irls_tol or (it > 0 and abs(prev_obj - obj) < 1e-8 * abs(obj)):
                break
            prev_obj = obj
        return beta, b0


# --------------------------------------------------------------------------------------
# Cross-validation over lambda
# --------------------------------------------------------------------------------------
def cv_elastic_net(X, y, family: str = "gaussian", event=None, alpha: float = 1.0,
                   n_folds: int = 5, folds=None, rule: str = "1se", seed: int = 0, **kw):
    """
    Fits the path on all data, then K-fold CV deviance for every lambda of that path.
    rule: "min" (lambda_min) or "1se" (lambda_1se). Returns the fitted ElasticNet with
    lambda set, plus cv_mean_, cv_se_, lambda_min_, lambda_1se_ attributes.
    """
    from .cv import make_folds
    X = np.asarray(X, float)
    full = ElasticNet(family, alpha=alpha, **kw).fit(X, y, event)
    lams = full.lambdas_
    folds = folds if folds is not None else make_folds(y, n_folds, family_task(family), seed,
                                                       event)
    n = len(y)
    D = np.full((len(folds), len(lams)), np.nan)
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(n), test)
        if family in ("binomial", "multinomial") and \
                len(np.unique(np.asarray(y)[train])) < len(full.classes_):
            continue                                  # a class is absent from this fold
        m = ElasticNet(family, alpha=alpha, lambdas=lams, **{k: v for k, v in kw.items()
                                                             if k != "n_lambda"})
        m.fit(X[train], np.asarray(y)[train], None if event is None else np.asarray(event)[train])
        L = min(len(m.lambdas_), len(lams))
        for li in range(L):
            if family == "cox":
                ev = np.asarray(event)
                eta_all = m.decision_function(X, li)
                d_all = deviance("cox", np.asarray(y), eta_all, ev) * n
                d_tr = deviance("cox", np.asarray(y)[train], eta_all[train], ev[train]) * len(train)
                D[f, li] = (d_all - d_tr) / len(test)
            else:
                yt = np.asarray(y)[test]
                eta = m.decision_function(X[test], li)
                if family in ("binomial", "multinomial"):
                    yt = np.searchsorted(full.classes_, yt)
                D[f, li] = deviance(family, yt, eta)
    cv_mean = np.nanmean(D, 0)
    cv_se = np.nanstd(D, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(D), 0))
    i_min = int(np.nanargmin(cv_mean))
    within = np.flatnonzero(cv_mean <= cv_mean[i_min] + cv_se[i_min])
    i_1se = int(within.min()) if len(within) else i_min
    full.cv_mean_, full.cv_se_ = cv_mean, cv_se
    full.lambda_min_, full.lambda_1se_ = lams[i_min], lams[i_1se]
    full._set_lambda(i_1se if rule == "1se" else i_min)
    return full


def family_task(family):
    return {"gaussian": "regression", "binomial": "classification",
            "multinomial": "classification", "cox": "survival"}[family]
