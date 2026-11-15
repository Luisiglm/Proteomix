"""
Clustering of temporal abundance profiles.

Profiles are built from the model's group estimates (beta of DifferentialAbundanceNR), not
from raw means: the selection model accounts for values missing because they were low, so
early / late time points with many missing values are not biased upwards.

Design: each group has a series (e.g. treatment "Ctrl" / "PDL1") and a time. A baseline
group (e.g. "unstimulated", time 0) can be shared by every series. A protein's profile is
the concatenation of its series (e.g. Ctrl 0, 5, 20, 240 min | PDL1 0, 5, 20, 240 min), so
clusters capture both the shape over time and the difference between series. Profiles are
z-scored per protein (shape, not abundance); Euclidean distance on z-scores is equivalent to
correlation distance.

Clustering:
    kmeans  k-means with many random starts (k-means++ initialisation)
    fuzzy   fuzzy c-means (as in Mfuzz): every protein gets a membership in each cluster;
            proteins with a low top membership have no clear temporal pattern
k: fixed, or chosen by the mean silhouette width over k_range ("auto").
Because time points here are independent cultures, profiles describe group-mean dynamics,
not individual trajectories.
"""
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------------------
def time_design(groups, series=None, times=None, baseline: str | None = None) -> pd.DataFrame:
    """
    Group -> (series, time). If series / times are not given, they are parsed from the group
    names: the time is a duration such as "5 min", "20min" or "4 h" (in minutes), the series
    is the rest of the name; groups without a duration are time 0 of their own series, or of
    every series if their name equals baseline.
    """
    rows = []
    for k, g in enumerate(groups):
        if series is not None and times is not None:
            s, t = series[k], float(times[k])
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(min|h|hr|s)\b", g, re.I)
            if m:
                unit = m.group(2).lower()
                t = float(m.group(1)) * {"min": 1, "h": 60, "hr": 60, "s": 1 / 60}[unit]
                s = (g[: m.start()] + g[m.end():]).strip()
            else:
                s, t = g, 0.0
        rows.append({"group": g, "series": s, "time": t})
    d = pd.DataFrame(rows)
    d["baseline"] = d["group"] == baseline if baseline else False
    return d


def build_profiles(model, feature_names, design: pd.DataFrame, group_names, proteins=None,
                   series_order=None) -> pd.DataFrame:
    """
    Proteins x (series, time) matrix of model-based group means. Baseline groups are copied
    to time 0 of every series. proteins: subset of feature names (default all fitted).
    """
    beta = pd.DataFrame(model.beta.numpy(), index=list(feature_names), columns=list(group_names))
    beta = beta[model.fitted.numpy()]
    if proteins is not None:
        beta = beta.loc[[p for p in proteins if p in beta.index]]
    d = design.set_index("group")
    series = series_order or [s for s in dict.fromkeys(d.loc[~d["baseline"], "series"])]
    cols, data = [], []
    for s in series:
        rows = d[(d["series"] == s) | d["baseline"]].sort_values("time")
        rows = rows[~rows.index.duplicated()]
        for g, r in rows.iterrows():
            if g not in beta.columns:
                continue
            cols.append((s, r["time"]))
            data.append(beta[g].to_numpy())
    P = pd.DataFrame(np.column_stack(data), index=beta.index,
                     columns=pd.MultiIndex.from_tuples(cols, names=["series", "time"]))
    return P.dropna()


def zscore_rows(P: pd.DataFrame) -> pd.DataFrame:
    Z = P.sub(P.mean(1), axis=0)
    sd = Z.std(1, ddof=1).replace(0, np.nan)
    return Z.div(sd, axis=0).dropna()


# --------------------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------------------
def _kmeans_pp(Z, k, rng):
    centers = [Z[rng.integers(len(Z))]]
    for _ in range(1, k):
        d2 = np.min(((Z[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), 1)
        centers.append(Z[rng.choice(len(Z), p=d2 / d2.sum())])
    return np.array(centers)


def kmeans(Z, k: int, n_init: int = 25, max_iter: int = 300, seed: int = 0):
    """Returns (labels, centers, inertia) of the best of n_init k-means++ starts."""
    Z = np.asarray(Z, float)
    rng = np.random.default_rng(seed)
    best = (None, None, np.inf)
    for _ in range(n_init):
        C = _kmeans_pp(Z, k, rng)
        for _ in range(max_iter):
            lab = ((Z[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            newC = np.array([Z[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = ((Z - C[lab]) ** 2).sum()
        if inertia < best[2]:
            best = (lab, C, inertia)
    return best


def fuzzy_cmeans(Z, k: int, m: float = 1.5, n_init: int = 10, max_iter: int = 500,
                 tol: float = 1e-6, seed: int = 0):
    """Returns (memberships n x k, centers, objective) of the best of n_init starts."""
    Z = np.asarray(Z, float)
    rng = np.random.default_rng(seed)
    best = (None, None, np.inf)
    for _ in range(n_init):
        U = rng.dirichlet(np.ones(k), len(Z))
        for _ in range(max_iter):
            W = U ** m
            C = (W.T @ Z) / W.sum(0)[:, None]
            D = np.sqrt(((Z[:, None, :] - C[None]) ** 2).sum(-1)) + 1e-12
            Unew = 1.0 / ((D[:, :, None] / D[:, None, :]) ** (2 / (m - 1))).sum(-1)
            if np.max(np.abs(Unew - U)) < tol:
                U = Unew
                break
            U = Unew
        obj = ((U ** m) * D ** 2).sum()
        if obj < best[2]:
            best = (U, C, obj)
    return best


def silhouette(Z, labels) -> float:
    """Mean silhouette width (Euclidean)."""
    Z = np.asarray(Z, float)
    labels = np.asarray(labels)
    ks = np.unique(labels)
    if len(ks) < 2:
        return np.nan
    sq = (Z ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0))
    s = np.zeros(len(Z))
    for i in range(len(Z)):
        own = labels == labels[i]
        if own.sum() <= 1:
            continue
        a = D[i, own].sum() / (own.sum() - 1)
        b = min(D[i, labels == k].mean() for k in ks if k != labels[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


@dataclass
class TimecourseResult:
    profiles: pd.DataFrame          # proteins x (series, time): model-based log2 means
    z: pd.DataFrame                 # z-scored profiles used for clustering
    assignments: pd.DataFrame       # protein -> cluster, membership (fuzzy), n_proteins
    centroids: pd.DataFrame         # cluster x (series, time), z-scale
    k: int
    method: str
    silhouette: dict                # k -> mean silhouette width (when k was chosen)


def cluster_profiles(profiles: pd.DataFrame, k="auto", k_range=range(2, 11),
                     method: str = "kmeans", m: float = 1.5, seed: int = 0,
                     max_silhouette_proteins: int = 3000) -> TimecourseResult:
    """
    Clusters z-scored profiles. With k="auto", k maximises the mean silhouette over k_range
    (estimated on a random subset of max_silhouette_proteins for speed). Clusters are
    numbered by size (1 = largest).
    """
    Z = zscore_rows(profiles)
    X = Z.to_numpy()
    sil = {}
    if k == "auto":
        rng = np.random.default_rng(seed)
        sub = X if len(X) <= max_silhouette_proteins else \
            X[rng.choice(len(X), max_silhouette_proteins, replace=False)]
        for kk in k_range:
            if kk >= len(sub):
                break
            lab, _, _ = kmeans(sub, kk, n_init=10, seed=seed)
            sil[kk] = silhouette(sub, lab)
        k = max(sil, key=sil.get)
    k = int(k)
    if method == "fuzzy":
        U, C, _ = fuzzy_cmeans(X, k, m=m, seed=seed)
        lab = U.argmax(1)
        memb = U.max(1)
    else:
        lab, C, _ = kmeans(X, k, seed=seed)
        memb = np.ones(len(X))
    order = pd.Series(lab).value_counts().index.tolist()        # relabel by size
    remap = {old: new + 1 for new, old in enumerate(order)}
    lab = np.array([remap[v] for v in lab])
    C = C[[order[i] for i in range(k)]]
    assign = pd.DataFrame({"cluster": lab, "membership": memb}, index=Z.index)
    assign["cluster_size"] = assign["cluster"].map(assign["cluster"].value_counts())
    cent = pd.DataFrame(C, index=pd.Index(range(1, k + 1), name="cluster"), columns=Z.columns)
    return TimecourseResult(profiles=profiles.loc[Z.index], z=Z, assignments=assign,
                            centroids=cent, k=k, method=method, silhouette=sil)


def responsive_proteins(results: dict, contrast_vectors: dict, groups, alpha: float = 0.05,
                        groups_subset=None) -> list:
    """
    Proteins significant in at least one contrast that only involves groups_subset (all
    contrasts if None), e.g. the IP time / treatment contrasts but not IP vs IgG.
    """
    groups = list(groups)
    keep = set()
    for label, res in results.items():
        v = np.asarray(contrast_vectors[label])
        used = {groups[i] for i in np.flatnonzero(v != 0)}
        if groups_subset is not None and not used <= set(groups_subset):
            continue
        keep |= set(res.index[res["padj"] < alpha])
    return sorted(keep)


def time_label(minutes: float) -> str:
    if minutes == 0:
        return "0"
    if minutes >= 60 and minutes % 60 == 0:
        return f"{int(minutes // 60)} h"
    return f"{minutes:g} min"
