"""
Performance metrics for regression, classification and survival predictions.

regression      rmse, mae, r2 (1 - SSE/SST), pearson
classification  accuracy, balanced_accuracy (mean recall per class), auc (binary ROC AUC;
                macro one-vs-rest for more classes), log_loss
survival        c_index (Harrell's concordance of risk scores with survival times; ties in
                risk count 1/2), and a log-rank test between high- and low-risk halves
"""
import numpy as np
from scipy import stats

PRIMARY = {"regression": "rmse", "classification": "auc", "survival": "c_index"}
HIGHER_IS_BETTER = {"rmse": False, "mae": False, "r2": True, "pearson": True,
                    "accuracy": True, "balanced_accuracy": True, "auc": True,
                    "log_loss": False, "c_index": True}


def regression_metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    res = y - pred
    sst = np.sum((y - y.mean()) ** 2)
    return {"rmse": float(np.sqrt(np.mean(res ** 2))), "mae": float(np.mean(np.abs(res))),
            "r2": float(1 - np.sum(res ** 2) / sst) if sst > 0 else np.nan,
            "pearson": float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 0 else np.nan}


def roc_auc(y01, score):
    """Binary ROC AUC via the Mann-Whitney statistic (ties count 1/2)."""
    y01 = np.asarray(y01).astype(bool)
    n1, n0 = y01.sum(), (~y01).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(score)
    return float((r[y01].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc_curve(y01, score):
    """False- and true-positive rates over all thresholds."""
    y01 = np.asarray(y01).astype(bool)
    order = np.argsort(-np.asarray(score, float), kind="stable")
    tp = np.cumsum(y01[order])
    fp = np.cumsum(~y01[order])
    return np.r_[0, fp / max(fp[-1], 1)], np.r_[0, tp / max(tp[-1], 1)]


def classification_metrics(y, proba, classes):
    """y: labels; proba: (n, K) probabilities in the order of classes."""
    y = np.asarray(y)
    classes = np.asarray(classes)
    P = np.clip(np.asarray(proba, float), 1e-12, 1)
    yi = np.searchsorted(classes, y)
    pred = P.argmax(1)
    recalls = [np.mean(pred[yi == k] == k) for k in range(len(classes)) if (yi == k).any()]
    if len(classes) == 2:
        auc = roc_auc(yi == 1, P[:, 1])
    else:
        auc = float(np.nanmean([roc_auc(yi == k, P[:, k]) for k in range(len(classes))]))
    return {"accuracy": float(np.mean(pred == yi)), "balanced_accuracy": float(np.mean(recalls)),
            "auc": auc, "log_loss": float(-np.mean(np.log(P[np.arange(len(y)), yi])))}


def concordance_index(time, event, risk):
    """Harrell's C: fraction of comparable pairs ordered correctly by risk (higher = earlier)."""
    t, e, r = (np.asarray(v, float) for v in (time, event, risk))
    num = den = 0.0
    for i in np.flatnonzero(e == 1):
        comp = t > t[i]
        den += comp.sum()
        num += (r[i] > r[comp]).sum() + 0.5 * (r[i] == r[comp]).sum()
    return float(num / den) if den > 0 else np.nan


def logrank_test(time, event, group):
    """Two-group log-rank test; returns (chi2, p)."""
    t, e, g = np.asarray(time, float), np.asarray(event, float), np.asarray(group).astype(bool)
    o_minus_e = var = 0.0
    for u in np.unique(t[e == 1]):
        at_risk = t >= u
        n, n1 = at_risk.sum(), (at_risk & g).sum()
        d = ((t == u) & (e == 1)).sum()
        d1 = ((t == u) & (e == 1) & g).sum()
        if n > 1:
            o_minus_e += d1 - d * n1 / n
            var += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    chi2 = o_minus_e ** 2 / var if var > 0 else 0.0
    return float(chi2), float(stats.chi2.sf(chi2, 1))


def kaplan_meier(time, event):
    """Kaplan-Meier survival curve: (times, survival) as step points starting at (0, 1)."""
    t, e = np.asarray(time, float), np.asarray(event, float)
    ts = np.unique(t[e == 1])
    surv, s = [1.0], 1.0
    for u in ts:
        n = (t >= u).sum()
        d = ((t == u) & (e == 1)).sum()
        s *= 1 - d / n
        surv.append(s)
    return np.r_[0, ts], np.array(surv)


def survival_metrics(time, event, risk):
    chi2, p = logrank_test(time, event, np.asarray(risk) > np.median(risk))
    return {"c_index": concordance_index(time, event, risk), "logrank_chi2": chi2,
            "logrank_p": p}


def compute_metrics(task, y, pred, event=None, classes=None):
    if task == "regression":
        return regression_metrics(y, pred)
    if task == "classification":
        return classification_metrics(y, pred, classes)
    return survival_metrics(y, event, pred)
