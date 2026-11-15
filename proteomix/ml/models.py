"""
Learners with a common interface and built-in hyperparameter tuning by inner
cross-validation (so the outer cross-validation in cv.py is a nested CV).

    lasso       L1-penalised GLM (gaussian / binomial / multinomial / Cox); lambda chosen by
                inner K-fold CV deviance (rule "1se" by default, "min" optional)
    elasticnet  the same with alpha = 0.5 (keeps groups of correlated proteins together)
    knn         k-nearest neighbours; k chosen by inner CV from k_grid (regression and
                classification only)
    mlp         multilayer perceptron; architecture / regularisation chosen by inner CV from
                grid (a single setting = no tuning), early stopping within each fit

Every learner: fit(X, y, event=None) and predict_output(X):
    regression -> predicted values; classification -> class probabilities (n, K) in the
    order of classes_; survival -> risk score (higher = shorter survival).
"""
import copy

import numpy as np

from .cv import make_folds
from .knn import KNN
from .lasso import cv_elastic_net
from .metrics import HIGHER_IS_BETTER, PRIMARY, compute_metrics
from .mlp import MLP


class Learner:
    name = "learner"
    tasks = ("regression", "classification", "survival")

    def __init__(self, task, inner_folds: int = 5, seed: int = 0):
        if task not in self.tasks:
            raise ValueError(f"{self.name} does not support {task}")
        self.task, self.inner_folds, self.seed = task, inner_folds, seed

    def params(self) -> dict:
        return {}

    def selected(self):
        return None


class LassoLearner(Learner):
    name = "lasso"

    def __init__(self, task, alpha: float = 1.0, rule: str = "1se", inner_folds: int = 5,
                 seed: int = 0, n_lambda: int = 50):
        super().__init__(task, inner_folds, seed)
        self.alpha, self.rule, self.n_lambda = alpha, rule, n_lambda

    def fit(self, X, y, event=None):
        if self.task == "regression":
            fam = "gaussian"
        elif self.task == "survival":
            fam = "cox"
        else:
            self.classes_ = np.unique(y)
            fam = "binomial" if len(self.classes_) == 2 else "multinomial"
        self.model_ = cv_elastic_net(X, y, fam, event=event, alpha=self.alpha,
                                     n_folds=self.inner_folds, rule=self.rule, seed=self.seed,
                                     n_lambda=self.n_lambda)
        return self

    def predict_output(self, X):
        if self.task == "classification":
            return self.model_.predict_proba(X)
        return self.model_.decision_function(X) if self.task == "survival" else \
            self.model_.predict(X)

    def selected(self):
        return self.model_.selected_features()

    def coefficients(self):
        return self.model_.coef_

    def params(self):
        return {"lambda": float(self.model_.lambda_),
                "n_selected": int(len(self.selected()))}


class ElasticNetLearner(LassoLearner):
    name = "elasticnet"

    def __init__(self, task, alpha: float = 0.5, **kw):
        super().__init__(task, alpha=alpha, **kw)


def _inner_select(task, X, y, event, candidates, make, inner_folds, seed):
    """Index of the candidate with the best inner-CV primary metric."""
    folds = make_folds(y, inner_folds, task, seed, event)
    key = PRIMARY[task]
    scores = []
    for cand in candidates:
        vals = []
        for test in folds:
            train = np.setdiff1d(np.arange(len(y)), test)
            if task == "classification" and len(np.unique(y[train])) < len(np.unique(y)):
                continue
            m = make(cand).fit(X[train], y[train], None if event is None else event[train])
            out = _output(m, task, X[test])
            if task == "classification":
                out = _align_proba(out, m.classes_, np.unique(y))
            vals.append(compute_metrics(task, y[test], out,
                                        None if event is None else event[test],
                                        np.unique(y))[key])
        scores.append(np.nanmean(vals) if vals else np.nan)
    scores = np.array(scores, float)
    best = np.nanargmax(scores) if HIGHER_IS_BETTER[key] else np.nanargmin(scores)
    return int(best), scores


def _output(m, task, X):
    if task == "classification":
        return m.predict_proba(X)
    return m.predict(X)


def _align_proba(P, model_classes, classes):
    out = np.zeros((P.shape[0], len(classes)))
    for j, c in enumerate(model_classes):
        out[:, np.searchsorted(classes, c)] = P[:, j]
    return out


class KNNLearner(Learner):
    name = "knn"
    tasks = ("regression", "classification")

    def __init__(self, task, k_grid=(1, 3, 5, 7, 9, 15), metric: str = "euclidean",
                 weights: str = "distance", inner_folds: int = 5, seed: int = 0):
        super().__init__(task, inner_folds, seed)
        self.k_grid, self.metric, self.weights = tuple(k_grid), metric, weights

    def fit(self, X, y, event=None):
        y = np.asarray(y)
        n_inner = len(y) * (1 - 1 / self.inner_folds)
        grid = [k for k in self.k_grid if k < n_inner] or [1]
        make = lambda k: KNN(self.task, k, self.metric, self.weights)  # noqa: E731
        best, self.cv_scores_ = _inner_select(self.task, X, y, None, grid, make,
                                              self.inner_folds, self.seed)
        self.k_ = grid[best]
        self.model_ = make(self.k_).fit(X, y)
        if self.task == "classification":
            self.classes_ = self.model_.classes_
        return self

    def predict_output(self, X):
        return _output(self.model_, self.task, X)

    def params(self):
        return {"k": int(self.k_)}


class MLPLearner(Learner):
    name = "mlp"
    DEFAULT_GRID = ({"hidden": (32,), "dropout": 0.3, "weight_decay": 1e-2},
                    {"hidden": (64, 32), "dropout": 0.3, "weight_decay": 1e-3})

    def __init__(self, task, grid=None, epochs: int = 500, lr: float = 1e-3,
                 patience: int = 30, inner_folds: int = 3, seed: int = 0):
        super().__init__(task, inner_folds, seed)
        self.grid = list(grid or self.DEFAULT_GRID)
        self.epochs, self.lr, self.patience = epochs, lr, patience

    def _make(self, cfg):
        return MLP(self.task, epochs=self.epochs, lr=self.lr, patience=self.patience,
                   seed=self.seed, **cfg)

    def fit(self, X, y, event=None):
        y = np.asarray(y)
        ev = None if event is None else np.asarray(event)
        if len(self.grid) > 1:
            best, self.cv_scores_ = _inner_select(self.task, X, y, ev, self.grid, self._make,
                                                  self.inner_folds, self.seed)
        else:
            best = 0
        self.config_ = self.grid[best]
        self.model_ = self._make(self.config_).fit(X, y, ev)
        if self.task == "classification":
            self.classes_ = self.model_.classes_
        return self

    def predict_output(self, X):
        return _output(self.model_, self.task, X)

    def params(self):
        return {**{k: str(v) for k, v in self.config_.items()},
                "epochs_run": int(self.model_.epochs_run_)}


LEARNERS = {"lasso": LassoLearner, "elasticnet": ElasticNetLearner, "knn": KNNLearner,
            "mlp": MLPLearner}


def make_learner(name: str, task: str, seed: int = 0, **kw):
    if name not in LEARNERS:
        raise ValueError(f"unknown model {name!r}; choose from {list(LEARNERS)}")
    return LEARNERS[name](task, seed=seed, **kw)


def clone(learner):
    return copy.deepcopy(learner)
