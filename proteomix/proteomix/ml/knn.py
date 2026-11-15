"""
k-nearest-neighbour regression and classification (sample neighbours in protein space).

metric: "euclidean" on the preprocessed (standardised) features, or "correlation"
    (1 - Pearson correlation between sample profiles; insensitive to overall shifts).
weights: "uniform" or "distance" (1 / distance, exact matches dominate).
Survival is not supported (use the lasso Cox model or the MLP).
"""
import numpy as np


class KNN:
    def __init__(self, task: str = "classification", k: int = 5, metric: str = "euclidean",
                 weights: str = "uniform"):
        if task not in ("regression", "classification"):
            raise ValueError("kNN supports regression and classification")
        self.task, self.k, self.metric, self.weights = task, int(k), metric, weights

    def fit(self, X, y, event=None):
        self.X_ = np.asarray(X, float)
        y = np.asarray(y)
        if self.task == "classification":
            self.classes_, self.y_ = np.unique(y, return_inverse=True)
        else:
            self.y_ = y.astype(float)
        return self

    def _distances(self, X):
        X = np.asarray(X, float)
        if self.metric == "correlation":
            A = X - X.mean(1, keepdims=True)
            B = self.X_ - self.X_.mean(1, keepdims=True)
            A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
            B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
            return 1 - A @ B.T
        d2 = (X ** 2).sum(1)[:, None] + (self.X_ ** 2).sum(1)[None, :] - 2 * X @ self.X_.T
        return np.sqrt(np.maximum(d2, 0))

    def _neighbours(self, X):
        D = self._distances(X)
        k = min(self.k, self.X_.shape[0])
        idx = np.argpartition(D, k - 1, axis=1)[:, :k]
        d = np.take_along_axis(D, idx, 1)
        w = np.ones_like(d) if self.weights == "uniform" else 1 / np.maximum(d, 1e-12)
        return idx, w / w.sum(1, keepdims=True)

    def predict_proba(self, X):
        idx, w = self._neighbours(X)
        K = len(self.classes_)
        P = np.zeros((len(idx), K))
        for c in range(K):
            P[:, c] = (w * (self.y_[idx] == c)).sum(1)
        return P

    def predict(self, X):
        if self.task == "classification":
            return self.classes_[self.predict_proba(X).argmax(1)]
        idx, w = self._neighbours(X)
        return (w * self.y_[idx]).sum(1)
