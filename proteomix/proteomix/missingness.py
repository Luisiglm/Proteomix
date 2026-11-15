"""
Per-batch logistic missingness model, P(missing) = sigmoid(w_b * (u - r_b)), where u is the
group mean of the observed values. Fitted by Newton/IRLS in the concave parameterisation
z = a * (u - u_bar) + c with step-halving (see MissingParametersNR.fit).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MissingParametersNR(nn.Module):
    """Per-batch logistic missingness model: P(missing) = sigmoid(w_b * (u - r_b))."""

    def __init__(self, num_batches: int, w_init: float = 1.0, r_init: float = 1.0):
        super().__init__()
        self.num_batches = num_batches
        self.w = nn.Parameter(torch.full((num_batches, 1), w_init))
        self.r = nn.Parameter(torch.full((num_batches, 1), r_init))
        # Log-likelihood per iteration, filled by fit(); last entry = final parameters
        self.ll_history: list[float] = []

    @staticmethod
    def compute_group_means(y: torch.Tensor, x: torch.Tensor):
        """
        y: (N, F) with NaNs for missing values. x: (N, G) one-hot group design.
        Returns u: (N, F) group mean assigned to each sample, and
        defined: (N, F) bool, False where the sample's group has no observed
        value for that feature (the mean is undefined, not 0).
        """
        x = x.to(y.dtype)
        valid = ~torch.isnan(y)
        sums = x.T @ torch.nan_to_num(y, nan=0.0)          # (G, F)
        counts = x.T @ valid.to(y.dtype)                   # (G, F)
        means = sums / counts.clamp(min=1.0)
        u = x @ means                                      # (N, F)
        defined = (x @ (counts > 0).to(y.dtype)) > 0       # (N, F)
        return u, defined

    def forward(self, y: torch.Tensor, x: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
        u, _ = self.compute_group_means(y, x)
        w = self.w.to(y.dtype)[batch_ids]                  # (N, 1)
        r = self.r.to(y.dtype)[batch_ids]
        return torch.sigmoid(w * (u - r))

    def _per_batch_sum(self, t: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
        """Sum an (N, F) tensor over features and over samples within each batch -> (B,)."""
        out = torch.zeros(self.num_batches, dtype=t.dtype, device=t.device)
        return out.index_add_(0, batch_ids, t.sum(dim=1))

    def compute_grad_and_hessian(self, y: torch.Tensor, x: torch.Tensor, batch_ids: torch.Tensor,
                                 fisher: bool = False):
        """
        Gradient (B, 2) and Hessian (B, 2, 2) of the Bernoulli log-likelihood
        w.r.t. (w_b, r_b), for all batches at once.
        fisher=True drops the -sum(e) cross term (expected Hessian / Fisher scoring),
        which keeps H negative semi-definite so every step is an ascent direction.
        """
        u, defined = self.compute_group_means(y, x)
        obs = defined.to(y.dtype)                          # entries that enter the likelihood
        m = torch.isnan(y).to(y.dtype)

        w_b = self.w.to(y.dtype).squeeze(1)                # (B,)
        r_b = self.r.to(y.dtype).squeeze(1)
        d = u - r_b[batch_ids, None]                       # (N, F)
        pi = torch.sigmoid(w_b[batch_ids, None] * d)
        e = (m - pi) * obs
        v = pi * (1.0 - pi) * obs

        S = lambda t: self._per_batch_sum(t, batch_ids)
        sum_e = S(e)

        g = torch.stack([S(e * d), -w_b * sum_e], dim=-1)  # (B, 2)

        H_ww = -S(v * d * d)
        H_rr = -(w_b ** 2) * S(v)
        H_wr = w_b * S(v * d)
        if not fisher:
            H_wr = H_wr - sum_e
        H = torch.stack([torch.stack([H_ww, H_wr], dim=-1),
                         torch.stack([H_wr, H_rr], dim=-1)], dim=-2)  # (B, 2, 2)
        return g, H

    @torch.no_grad()
    def step_newton_raphson(self, y: torch.Tensor, x: torch.Tensor, batch_ids: torch.Tensor,
                            alpha: float = 1.0, damping: float = 1e-5, fisher: bool = False):
        """
        One Newton-Raphson (or Fisher scoring) step on all batches simultaneously.
        Returns the per-batch gradient norm, useful as a convergence criterion.
        Batches with no samples get g = 0, H = 0 -> delta = 0 (requires damping > 0).
        """
        g, H = self.compute_grad_and_hessian(y, x, batch_ids, fisher=fisher)
        eye = torch.eye(2, dtype=H.dtype, device=H.device)
        # Subtracting damping*I makes H more negative definite (correct sign for maximisation)
        delta = torch.linalg.solve(H - damping * eye, g.unsqueeze(-1)).squeeze(-1)  # (B, 2)
        self.w[:, 0] -= alpha * delta[:, 0].to(self.w.dtype)
        self.r[:, 0] -= alpha * delta[:, 1].to(self.r.dtype)
        return g.norm(dim=-1)

    def log_likelihood(self, y: torch.Tensor, x: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
        """Total Bernoulli log-likelihood over entries with a defined group mean."""
        u, defined = self.compute_group_means(y, x)
        m = torch.isnan(y).to(y.dtype)
        z = self.w.to(y.dtype)[batch_ids] * (u - self.r.to(y.dtype)[batch_ids])
        ll = m * F.logsigmoid(z) + (1.0 - m) * F.logsigmoid(-z)
        return (ll * defined.to(y.dtype)).sum()

    def _logistic_terms(self, a, c, uc, m, obs, batch_ids):
        """Per-batch log-likelihood (B,), gradient (B, 2) and Hessian (B, 2, 2)
        for z = a*uc + c, where uc is the centred group mean."""
        z = a[batch_ids, None] * uc + c[batch_ids, None]
        pi = torch.sigmoid(z)
        e = (m - pi) * obs
        v = pi * (1.0 - pi) * obs
        S = lambda t: self._per_batch_sum(t, batch_ids)
        ll_b = S(obs * (m * F.logsigmoid(z) + (1.0 - m) * F.logsigmoid(-z)))
        g = torch.stack([S(e * uc), S(e)], dim=-1)
        H_aa, H_ac, H_cc = -S(v * uc * uc), -S(v * uc), -S(v)
        H = torch.stack([torch.stack([H_aa, H_ac], dim=-1),
                         torch.stack([H_ac, H_cc], dim=-1)], dim=-2)
        return ll_b, g, H

    @torch.no_grad()
    def fit(self, y: torch.Tensor, x: torch.Tensor, batch_ids: torch.Tensor,
            max_iter: int = 100, tol: float = 1e-10, damping: float = 1e-8,
            max_halvings: int = 30, warm_start: bool = False,
            verbose: bool = False) -> list[float]:
        """
        Fits (w, r) per batch by Newton/IRLS in the concave parameterisation
        z = a*(u - u_bar) + c, with three safeguards against divergence:
          * u is centred at its global mean (u_bar), so the 2x2 Hessian is well-conditioned;
          * the default start is a = 0, c = logit(batch missingness rate), the standard
            IRLS start, from which nothing is saturated;
          * per-batch step-halving: a batch's step is halved until its log-likelihood
            does not decrease, so the log-likelihood is monotone non-decreasing.
        Set warm_start=True to start from the current (w, r) instead.
        Returns self.ll_history; its last entry is the log-likelihood at the final parameters.
        Afterwards: w = a, r = u_bar - c / a.
        """
        u, defined = self.compute_group_means(y, x)
        obs = defined.to(y.dtype)
        m = torch.isnan(y).to(y.dtype)
        u_bar = (u * obs).sum() / obs.sum()
        uc = (u - u_bar) * obs
        B = self.num_batches
        S = lambda t: self._per_batch_sum(t, batch_ids)

        if warm_start:
            a = self.w.detach().to(y.dtype).squeeze(1).clone()
            c = -a * (self.r.detach().to(y.dtype).squeeze(1) - u_bar)
        else:
            rate = (S(m * obs) + 0.5) / (S(obs) + 1.0)   # smoothed, never exactly 0 or 1
            a = torch.zeros(B, dtype=y.dtype, device=y.device)
            c = torch.logit(rate)

        terms = lambda a_, c_: self._logistic_terms(a_, c_, uc, m, obs, batch_ids)
        ll_b, g, H = terms(a, c)
        self.ll_history = [ll_b.sum().item()]
        eye = torch.eye(2, dtype=y.dtype, device=y.device)

        for it in range(max_iter):
            delta = torch.linalg.solve(H - damping * eye, g.unsqueeze(-1)).squeeze(-1)
            step = torch.ones(B, dtype=y.dtype, device=y.device)
            for _ in range(max_halvings):
                a_new = a - step * delta[:, 0]
                c_new = c - step * delta[:, 1]
                ll_new, g_new, H_new = terms(a_new, c_new)
                worse = ~(ll_new >= ll_b - 1e-12 * ll_b.abs())   # also catches NaN
                if not worse.any():
                    break
                step = torch.where(worse, step / 2, step)
            ok = ~worse                                          # batches that failed keep old values
            a, c = torch.where(ok, a_new, a), torch.where(ok, c_new, c)
            ll_b = torch.where(ok, ll_new, ll_b)
            g = torch.where(ok[:, None], g_new, g)
            H = torch.where(ok[:, None, None], H_new, H)

            prev, cur = self.ll_history[-1], ll_b.sum().item()
            self.ll_history.append(cur)
            if verbose:
                print(f"iter {it + 1:3d}  ll = {cur:.6f}  max|grad| = {g.abs().max().item():.3e}")
            if abs(cur - prev) <= tol * max(1.0, abs(prev)):
                break

        # Large |a| * sd(u) with ll near 0 indicates (quasi-)separation: no finite MLE
        u_sd = torch.sqrt((uc ** 2).sum() / obs.sum())
        if torch.any(a.abs() * u_sd > 50):
            print("Warning: |w| very large for some batches; missingness may be (almost) "
                  "perfectly separated by abundance there, so w and r are not identifiable.")

        r = torch.where(a != 0, u_bar - c / a, torch.full_like(a, float("nan")))
        self.w[:, 0] = a.to(self.w.dtype)
        self.r[:, 0] = r.to(self.r.dtype)
        return self.ll_history
