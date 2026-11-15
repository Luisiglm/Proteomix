"""Cross-validation folds (filled in below)."""
import numpy as np


def make_folds(y, n_folds: int = 5, task: str = "regression", seed: int = 0, event=None):
    """
    Test-index arrays for K folds. Classification: stratified by class; survival: stratified
    by event status; regression: stratified by outcome quantile bins.
    """
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)
    if task == "classification":
        strata = np.unique(y, return_inverse=True)[1]
    elif task == "survival":
        strata = np.asarray(event).astype(int)
    else:
        strata = np.zeros(n, dtype=int)
        order = np.argsort(y, kind="stable")
        strata[order] = np.arange(n) // max(n_folds, 1)
    folds = [[] for _ in range(n_folds)]
    offset = 0
    for s in np.unique(strata):
        members = rng.permutation(np.flatnonzero(strata == s))
        for i, m in enumerate(members):
            folds[(i + offset) % n_folds].append(m)
        offset += len(members)
    return [np.sort(np.array(f, dtype=int)) for f in folds if len(f)]


# --------------------------------------------------------------------------------------
# Nested cross-validation and final models
# --------------------------------------------------------------------------------------
import logging
import pickle
from dataclasses import dataclass, field

import pandas as pd

_log = logging.getLogger("proteomix")


@dataclass
class CVResult:
    task: str
    fold_metrics: pd.DataFrame          # one row per repeat x fold x model
    pooled_metrics: pd.DataFrame        # one row per repeat x model (metrics on pooled OOF)
    oof: dict                           # model -> list (per repeat) of out-of-fold outputs
    selection: dict                     # model -> Series: fraction of outer fits selecting
    params: pd.DataFrame                # tuned hyperparameters per repeat x fold x model
    y: np.ndarray
    event: np.ndarray | None
    classes: np.ndarray | None
    samples: list
    feature_names: list = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """Mean and SD over folds, and the pooled out-of-fold metric (mean over repeats)."""
        m = [c for c in self.fold_metrics.columns if c not in ("repeat", "fold", "model",
                                                               "n_test", "n_features")]
        a = self.fold_metrics.groupby("model", sort=False)[m].agg(["mean", "std"])
        a.columns = [f"{c}_{s}" for c, s in a.columns]
        pooled = self.pooled_metrics.groupby("model", sort=False)[m].mean()
        pooled.columns = [f"{c}_pooled" for c in pooled.columns]
        return a.join(pooled)


def _align(out, learner, classes):
    if classes is None:
        return np.asarray(out, float)
    from .models import _align_proba
    return _align_proba(out, learner.classes_, classes)


def cross_validate(Y: pd.DataFrame, y, task: str, learners: dict, event=None,
                   n_folds: int = 5, repeats: int = 1, preprocess: dict | None = None,
                   seed: int = 0, progress=None) -> CVResult:
    """
    Nested CV: for every repeat, K outer folds; preprocessing and every learner (with its
    own inner-CV tuning) are fitted on the outer training part only and evaluated on the
    held-out part. All learners see the same folds, so they can be compared pairwise.

    Y: samples x proteins (log2, normalised, NaN = missing); y: outcome (values, labels or
    survival times); event: 0/1 for survival. learners: {name: Learner}.
    """
    from .metrics import compute_metrics
    from .models import clone
    from .preprocess import Preprocessor
    y = np.asarray(y)
    ev = None if event is None else np.asarray(event).astype(int)
    classes = np.unique(y) if task == "classification" else None
    n = len(y)
    rows, pooled_rows, prow = [], [], []
    oof = {name: [] for name in learners}
    sel_counts = {name: pd.Series(0.0, index=Y.columns) for name in learners}
    n_fits = 0
    for r in range(repeats):
        folds = make_folds(y, n_folds, task, seed + r, ev)
        store = {name: (np.zeros((n, len(classes))) if classes is not None else np.zeros(n))
                 for name in learners}
        for f, test in enumerate(folds):
            train = np.setdiff1d(np.arange(n), test)
            pre = Preprocessor(**(preprocess or {}), seed=seed + r).fit(Y.iloc[train])
            Xtr, Xte = pre.transform(Y.iloc[train]), pre.transform(Y.iloc[test])
            n_fits += 1
            for name, proto in learners.items():
                lr = clone(proto)
                lr.seed = seed + 1000 * r + f
                lr.fit(Xtr, y[train], None if ev is None else ev[train])
                out = _align(lr.predict_output(Xte), lr, classes)
                store[name][test] = out
                met = compute_metrics(task, y[test], out, None if ev is None else ev[test],
                                      classes)
                rows.append({"repeat": r, "fold": f, "model": name, "n_test": len(test),
                             "n_features": Xtr.shape[1], **met})
                prow.append({"repeat": r, "fold": f, "model": name, **lr.params()})
                s = lr.selected()
                if s is not None:
                    sel_counts[name][np.asarray(pre.feature_names_)[s]] += 1
                if progress is not None:
                    progress.step(f"repeat {r + 1} fold {f + 1}: {name}")
                _log.debug(f"[ml] repeat {r + 1} fold {f + 1} {name}: "
                           + ", ".join(f"{k}={v:.3f}" for k, v in met.items()
                                       if isinstance(v, float)) + f" {lr.params()}")
        for name in learners:
            oof[name].append(store[name])
            pooled_rows.append({"repeat": r, "model": name,
                                **compute_metrics(task, y, store[name], ev, classes)})
    selection = {k: (v / n_fits).sort_values(ascending=False) for k, v in sel_counts.items()
                 if v.sum() > 0}
    return CVResult(task=task, fold_metrics=pd.DataFrame(rows),
                    pooled_metrics=pd.DataFrame(pooled_rows), oof=oof, selection=selection,
                    params=pd.DataFrame(prow), y=y, event=ev, classes=classes,
                    samples=list(Y.index), feature_names=list(Y.columns))


class TrainedModel:
    """Preprocessing + learner fitted on all samples; predicts for new samples."""

    def __init__(self, preprocessor, learner, task, classes=None, info=None):
        self.preprocessor, self.learner, self.task = preprocessor, learner, task
        self.classes, self.info = classes, info or {}

    def predict(self, Y: pd.DataFrame) -> pd.DataFrame:
        """Y: samples x proteins (log2, normalised as for training; NaN allowed)."""
        X = self.preprocessor.transform(Y)
        out = self.learner.predict_output(X)
        if self.task == "classification":
            P = _align(out, self.learner, self.classes)
            t = pd.DataFrame(P, index=Y.index, columns=[f"p_{c}" for c in self.classes])
            t.insert(0, "predicted", self.classes[P.argmax(1)])
            return t
        col = "risk_score" if self.task == "survival" else "predicted"
        return pd.DataFrame({col: np.asarray(out, float)}, index=Y.index)

    def save(self, path: str):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "TrainedModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def train_final(Y: pd.DataFrame, y, task: str, learner, event=None,
                preprocess: dict | None = None, seed: int = 0) -> TrainedModel:
    """Fits preprocessing and the learner (with its inner-CV tuning) on all samples."""
    from .preprocess import Preprocessor
    y = np.asarray(y)
    pre = Preprocessor(**(preprocess or {}), seed=seed).fit(Y)
    learner.fit(pre.transform(Y), y, None if event is None else np.asarray(event).astype(int))
    classes = np.unique(y) if task == "classification" else None
    return TrainedModel(pre, learner, task, classes,
                        {"n_samples": len(y), "features": pre.feature_names_,
                         "params": learner.params()})
