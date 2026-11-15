"""
Multilayer perceptron (PyTorch) for regression, classification and survival.

Architecture: input -> [Linear -> ReLU -> Dropout] x len(hidden) -> output.
Losses: mean squared error on the standardised outcome (regression), cross-entropy
(classification), negative Cox partial log-likelihood with Breslow ties (survival; the
"DeepSurv" loss, output = log risk).
Training: Adam with weight decay (L2), full-batch by default (proteomics cohorts are small),
early stopping on a stratified validation split of the training data (val_fraction) with
the best weights restored. Deterministic for a given seed on CPU.
With few samples and many proteins MLPs overfit easily: keep hidden layers small, use
dropout and weight decay, and compare with the lasso in cross-validation.
"""
import numpy as np
import torch
import torch.nn as nn


def cox_loss(log_risk, time, event):
    """Negative mean Breslow partial log-likelihood."""
    order = torch.argsort(time, descending=True)
    lr, ev, t = log_risk[order], event[order], time[order]
    lse = torch.logcumsumexp(lr, 0)
    # tied times: all members of a tie share the risk set of the earliest (largest cumsum)
    _, inv, counts = torch.unique_consecutive(t, return_inverse=True, return_counts=True)
    last = torch.cumsum(counts, 0) - 1
    lse = lse[last][inv]
    n_ev = ev.sum().clamp(min=1)
    return -((lr - lse) * ev).sum() / n_ev


class MLP:
    def __init__(self, task: str = "classification", hidden=(64, 32), dropout: float = 0.3,
                 weight_decay: float = 1e-3, lr: float = 1e-3, epochs: int = 500,
                 batch_size: int | None = None, val_fraction: float = 0.2, patience: int = 30,
                 seed: int = 0):
        self.task, self.hidden, self.dropout = task, tuple(hidden), dropout
        self.weight_decay, self.lr, self.epochs = weight_decay, lr, epochs
        self.batch_size, self.val_fraction, self.patience = batch_size, val_fraction, patience
        self.seed = seed

    def _net(self, p, out):
        layers, d = [], p
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        layers.append(nn.Linear(d, out))
        return nn.Sequential(*layers).double()

    def _loss(self, out, y, ev):
        if self.task == "regression":
            return ((out[:, 0] - y) ** 2).mean()
        if self.task == "classification":
            return nn.functional.cross_entropy(out, y)
        return cox_loss(out[:, 0], y, ev)

    def fit(self, X, y, event=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, float)
        n, p = X.shape
        y = np.asarray(y)
        if self.task == "classification":
            self.classes_, yi = np.unique(y, return_inverse=True)
            out, yt = len(self.classes_), torch.tensor(yi, dtype=torch.long)
        elif self.task == "regression":
            self.y_mean_, self.y_sd_ = y.astype(float).mean(), y.astype(float).std() or 1.0
            out, yt = 1, torch.tensor((y.astype(float) - self.y_mean_) / self.y_sd_)
        else:
            out, yt = 1, torch.tensor(y.astype(float))
        ev = torch.tensor(np.asarray(event, float)) if event is not None else None
        Xt = torch.tensor(X)

        # stratified validation split for early stopping
        idx = np.arange(n)
        strat = yi if self.task == "classification" else (np.asarray(event) if ev is not None
                                                         else np.zeros(n))
        val = []
        if self.val_fraction > 0 and n >= 10:
            for s in np.unique(strat):
                members = rng.permutation(idx[strat == s])
                val += list(members[: max(1, int(round(self.val_fraction * len(members))))])
        val = np.array(sorted(val), dtype=int)
        tr = np.setdiff1d(idx, val)
        self.net_ = self._net(p, out)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bs = self.batch_size or len(tr)
        best, best_state, wait = np.inf, None, 0
        self.history_ = []
        for epoch in range(self.epochs):
            self.net_.train()
            perm = rng.permutation(tr)
            for start in range(0, len(perm), bs):
                b = torch.tensor(perm[start:start + bs])
                if self.task == "survival" and ev[b].sum() == 0:
                    continue
                opt.zero_grad()
                loss = self._loss(self.net_(Xt[b]), yt[b], None if ev is None else ev[b])
                loss.backward()
                opt.step()
            if len(val):
                self.net_.eval()
                with torch.no_grad():
                    vb = torch.tensor(val)
                    vloss = float(self._loss(self.net_(Xt[vb]), yt[vb],
                                             None if ev is None else ev[vb]))
                self.history_.append(vloss)
                if vloss < best - 1e-6:
                    best, wait = vloss, 0
                    best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.epochs_run_ = epoch + 1
        self.net_.eval()
        return self

    def _forward(self, X):
        with torch.no_grad():
            return self.net_(torch.tensor(np.asarray(X, float))).numpy()

    def predict_proba(self, X):
        o = self._forward(X)
        o = o - o.max(1, keepdims=True)
        e = np.exp(o)
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        o = self._forward(X)
        if self.task == "classification":
            return self.classes_[o.argmax(1)]
        if self.task == "regression":
            return o[:, 0] * self.y_sd_ + self.y_mean_
        return o[:, 0]                                       # log risk
