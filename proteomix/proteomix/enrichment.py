"""
Pathway / gene-set enrichment for differential abundance results.

Gene sets come from local GMT files (reproducible, no web access needed), e.g. Reactome
("ReactomePathways.gmt") or MSigDB collections (hallmark, GO, KEGG ... as gene symbols).

Two tests per contrast:

camera (default, competitive, ranks every tested protein)
    For each set S of m genes, compares the mean z-score of S with that of all other genes,
    inflating the variance by the variance inflation factor VIF = 1 + (m - 1) * rho, where
    rho is the average correlation between genes of S estimated from the model residuals
    (Wu & Smyth 2012, CAMERA). Ignoring that correlation, as most pre-ranked tests do,
    makes p-values for pathways of co-regulated proteins far too small. rho is not allowed
    to be negative. z-scores are the signed normal quantiles of each protein's p-value.
ora (over-representation of hits)
    Hypergeometric test of the overlap between significant proteins (padj < alpha and
    |log2 FC| >= lfc) and each set, separately for up, down and all hits. Simple and
    threshold-dependent; useful when few proteins are significant in a clear direction.

Universe: proteins tested in this analysis (fitted, with a gene symbol) that belong to at
least one set of the collection. Using the genome instead would make detectable-proteome
categories (e.g. ribosome, metabolism) look enriched in any proteomics hit list.
Sets are kept if they have between min_size and max_size genes in the universe.
BH correction is applied within each collection and test.
Proteins sharing a gene symbol are reduced to the one with the largest |z|; for site-level
(PTM) results this also means genes with many sites are more likely to have an extreme
value, which favours heavily modified proteins.
"""
import logging
import os

import numpy as np
import pandas as pd
from scipy import stats

from .testing import benjamini_hochberg

_log = logging.getLogger("proteomix")


# --------------------------------------------------------------------------------------
# Gene sets
# --------------------------------------------------------------------------------------
def read_gmt(path: str, upper: bool = True) -> dict:
    """GMT file -> {set name: set of gene symbols}. Symbols are upper-cased by default."""
    sets = {}
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            genes = {g.strip().upper() if upper else g.strip() for g in parts[2:] if g.strip()}
            if genes:
                sets[parts[0]] = sets.get(parts[0], set()) | genes
    if not sets:
        raise ValueError(f"no gene sets read from {path}")
    return sets


def collection_name(path: str) -> str:
    base = os.path.basename(path)
    for ext in (".gz", ".gmt", ".txt"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
    return base


# --------------------------------------------------------------------------------------
# Gene-level statistics
# --------------------------------------------------------------------------------------
def gene_level_stats(results: pd.DataFrame, genes, upper: bool = True) -> pd.DataFrame:
    """
    One row per gene symbol: z (signed normal score of the p-value), estimate, pvalue, padj
    and the feature it came from. Features without a gene or without a p-value are dropped;
    duplicated genes keep the feature with the largest |z|.
    """
    g = pd.Series(list(genes), index=results.index).astype(str).str.strip()
    if upper:
        g = g.str.upper()
    p = results["pvalue"].astype(float).clip(lower=1e-300)
    est = results["estimate"].astype(float)
    z = np.sign(est) * stats.norm.isf(p / 2)
    t = pd.DataFrame({"gene": g, "z": z, "estimate": est, "pvalue": p,
                      "padj": results["padj"].astype(float), "feature": results.index})
    t = t[t["gene"].ne("") & t["gene"].ne("NAN") & t["gene"].ne("nan") & np.isfinite(t["z"])]
    t = t.loc[t["z"].abs().groupby(t["gene"]).idxmax()]
    return t.set_index("gene")


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
def camera(gene_stats: pd.DataFrame, gene_sets: dict, residuals: np.ndarray | None = None,
           df_resid: float | None = None, min_size: int = 5, max_size: int = 500) -> pd.DataFrame:
    """
    Competitive test (CAMERA). residuals: standardised model residuals, samples x genes,
    columns in gene_stats order; without them, rho = 0 (an uncorrected pre-ranked test).
    """
    genes = gene_stats.index.to_numpy()
    pos = {g: i for i, g in enumerate(genes)}
    z = gene_stats["z"].to_numpy(float)
    G = len(z)
    rows = []
    if residuals is not None:
        R = np.asarray(residuals, dtype=float)
        n_eff = df_resid if df_resid else max(R.shape[0] - 1, 1)
        var_g = (R ** 2).sum(0) / n_eff
    for name, members in gene_sets.items():
        idx = np.array([pos[g] for g in members if g in pos], dtype=int)
        m = len(idx)
        if m < min_size or m > max_size or m >= G - 1:
            continue
        rho = 0.0
        if residuals is not None and m > 1:
            U = R[:, idx].sum(1)
            vif_raw = ((U ** 2).sum() / n_eff) / max(var_g[idx].sum(), 1e-12)
            rho = float(np.clip((vif_raw - 1) / (m - 1), 0.0, 1.0))
        vif = 1 + (m - 1) * rho
        inset = np.zeros(G, dtype=bool)
        inset[idx] = True
        mean_in, mean_out = z[inset].mean(), z[~inset].mean()
        pooled = (((z[inset] - mean_in) ** 2).sum() + ((z[~inset] - mean_out) ** 2).sum()) / (G - 2)
        delta = mean_in - mean_out
        tstat = delta / np.sqrt(pooled * (vif / m + 1 / (G - m)))
        p = 2 * stats.t.sf(abs(tstat), G - 2)
        sub = gene_stats.iloc[idx]
        lead = sub.sort_values("z", ascending=delta < 0).index[:15]
        rows.append({"term": name, "n_genes": m, "direction": "Up" if delta > 0 else "Down",
                     "mean_z": mean_in, "delta_z": delta, "correlation": rho, "t": tstat,
                     "pvalue": p, "n_significant": int((sub["padj"] < 0.05).sum()),
                     "leading_genes": ";".join(lead)})
    out = pd.DataFrame(rows)
    if len(out):
        out["padj"] = benjamini_hochberg(out["pvalue"].to_numpy())
        out = out.sort_values("pvalue").reset_index(drop=True)
    return out


def ora(hits: set, universe: set, gene_sets: dict, min_size: int = 5,
        max_size: int = 500) -> pd.DataFrame:
    """Hypergeometric over-representation of hits among the universe, per gene set."""
    hits = set(hits) & universe
    N, n = len(universe), len(hits)
    rows = []
    for name, members in gene_sets.items():
        s = set(members) & universe
        K = len(s)
        if K < min_size or K > max_size:
            continue
        ov = hits & s
        k = len(ov)
        p = stats.hypergeom.sf(k - 1, N, K, n) if n else 1.0
        exp = K * n / N if N else 0
        rows.append({"term": name, "n_genes": K, "overlap": k, "expected": exp,
                     "fold_enrichment": k / exp if exp > 0 else np.nan, "pvalue": p,
                     "genes": ";".join(sorted(ov))})
    out = pd.DataFrame(rows)
    if len(out):
        out["padj"] = benjamini_hochberg(out["pvalue"].to_numpy())
        out = out.sort_values("pvalue").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def run_enrichment(results: pd.DataFrame, genes, collections: dict, model=None,
                   features=None, methods=("camera", "ora"), alpha: float = 0.05,
                   lfc: float = 0.0, min_size: int = 5, max_size: int = 500) -> dict:
    """
    results: one contrast's table (estimate, pvalue, padj, indexed by feature name).
    genes: gene symbol per row of results. collections: {name: gene sets} (from read_gmt).
    model: fitted DifferentialAbundanceNR whose residuals give CAMERA's inter-gene correlation
        (features: its feature names, in model order). Without a model, rho = 0.
    Returns {(collection, test): DataFrame}; test in camera, ora_up, ora_down, ora_all.
    """
    gs = gene_level_stats(results, genes)
    R = df_res = None
    if model is not None and "camera" in methods:
        feat = list(features) if features is not None else list(results.index)
        pos = {f: i for i, f in enumerate(feat)}
        y = model.y_input.numpy()
        mu = model.x_input.numpy() @ model.beta.numpy().T
        off = getattr(model, "offset_input", None)
        if off is not None:
            mu = mu + off.numpy()
        Z = (y - mu) / np.sqrt(model.s2.numpy())
        Z = np.where(np.isfinite(Z), Z, 0.0)
        R = Z[:, [pos[f] for f in gs["feature"]]]
        df_res = max(y.shape[0] - model.beta.shape[1], 1)
    out = {}
    for cname, sets in collections.items():
        universe = {g for g in gs.index if any(g in s for s in sets.values())}
        if not universe:
            _log.warning(f"enrichment: no tested gene is in collection {cname} "
                         "(check that the GMT uses gene symbols)")
            continue
        g_u = gs.loc[sorted(universe)]
        if "camera" in methods:
            Ru = None if R is None else R[:, [gs.index.get_loc(g) for g in g_u.index]]
            out[(cname, "camera")] = camera(g_u, sets, Ru, df_res, min_size, max_size)
        if "ora" in methods:
            sig = (g_u["padj"] < alpha) & (g_u["estimate"].abs() >= lfc)
            for lab, mask in (("ora_up", sig & (g_u["estimate"] > 0)),
                              ("ora_down", sig & (g_u["estimate"] < 0)), ("ora_all", sig)):
                out[(cname, lab)] = ora(set(g_u.index[mask]), universe, sets, min_size,
                                        max_size)
        _log.debug(f"enrichment {cname}: universe {len(universe)} genes, "
                   + ", ".join(f"{k[1]} {int((v['padj'] < alpha).sum()) if len(v) else 0}"
                               for k, v in out.items() if k[0] == cname)
                   + f" sets at FDR {alpha:g}")
    return out
