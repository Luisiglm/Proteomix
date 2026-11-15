"""
Kinase activity inference and sequence motifs for phosphosite results.

Inputs (user-provided, like gene sets and structures)
  kinase predictions  per-site kinase predictions, e.g. exported from the Kinase Library
                      (Johnson et al. 2023), in long format (one row per site and kinase:
                      site or sequence, kinase, percentile and / or score, optional rank) or
                      wide format (one row per site, one column per kinase holding percentiles)
  FASTA               protein sequences (canonical UniProt, same numbering as the results):
                      gives each site's sequence window (+/- 7 residues), used to match
                      predictions made from sequences and for motif enrichment

Kinase sets: for each kinase, the quantified sites predicted for it with percentile >= 90
and among the site's top 15 kinases (both adjustable); sets with fewer than 3 sites are
skipped. Predictions are probabilistic: a set is "sites this kinase could phosphorylate".

Activity tests (per contrast, on the protein-adjusted site changes for phospho data)
  ksea    KSEA z-score (Casado et al. 2013): (mean log2 FC of the kinase's substrates - mean
          of all sites) * sqrt(m) / SD of all sites; normal p-value, BH-adjusted. Simple and
          widely used, but treats substrates as independent.
  camera  the competitive CAMERA-style test of enrichment.py on site-level z-scores, with the
          inter-site correlation estimated from the model residuals (substrates on the same
          protein or pathway are correlated, which makes KSEA p-values too small).
A positive score = substrates go up = higher inferred activity.

Motifs (from the windows): proline-directed (S/T-P), basophilic (R/K at -3), acidophilic
(D/E at +1..+3), tyrosine; tested per cluster against all quantified sites (Fisher's test).
"""
import logging
import re

import numpy as np
import pandas as pd
from scipy import stats

from .testing import benjamini_hochberg

_log = logging.getLogger("proteomix")
SITE_RE = re.compile(r"(?:^|[_\-\s])([STY])(\d+)(?:$|[_\-\s])")
FLANK = 7


# --------------------------------------------------------------------------------------
# Sequences and windows
# --------------------------------------------------------------------------------------
def read_fasta(path: str) -> dict:
    """{accession: sequence} plus {gene: sequence} keys from UniProt-style headers."""
    seqs, header, chunks = {}, None, []

    def flush():
        if header is not None:
            seq = "".join(chunks).replace(" ", "").upper()
            parts = header.split("|")
            acc = parts[1] if len(parts) >= 3 else header.split()[0]
            seqs[acc] = seq
            m = re.search(r"\bGN=(\S+)", header)
            if m:
                seqs.setdefault(m.group(1).upper(), seq)
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                flush()
                header, chunks = line[1:], []
            elif line:
                chunks.append(line)
    flush()
    if not seqs:
        raise ValueError(f"no sequences in {path}")
    return seqs


def window(seq: str, pos: int, flank: int = FLANK) -> str:
    """Sequence centred on 1-based pos, padded with '_' beyond the ends."""
    padded = "_" * flank + seq + "_" * flank
    return padded[pos - 1: pos + 2 * flank]


def site_positions(names, genes=None, keys=None) -> pd.DataFrame:
    """Residue and position parsed from site names like GENE_S123 (NaN when absent)."""
    rows = []
    for k, n in enumerate(names):
        m = SITE_RE.search(str(n)) or SITE_RE.search(str(n) + "_")
        rows.append({"site": n, "residue": m.group(1) if m else None,
                     "position": int(m.group(2)) if m else np.nan,
                     "gene": None if genes is None else str(genes[k]),
                     "protein_key": None if keys is None else str(keys[k])})
    return pd.DataFrame(rows)


def site_windows(sites: pd.DataFrame, fasta: dict, flank: int = FLANK) -> pd.DataFrame:
    """Adds the sequence window and a residue check (residue at position in the FASTA)."""
    out = sites.copy()
    wins, ok = [], []
    for _, r in out.iterrows():
        seq = None
        for key in (r.get("protein_key"), r.get("gene")):
            if key and str(key).upper() in fasta:
                seq = fasta[str(key).upper()]
                break
            if key and str(key) in fasta:
                seq = fasta[str(key)]
                break
        if seq is None or not np.isfinite(r["position"]) or r["position"] > len(seq):
            wins.append(None)
            ok.append(False)
            continue
        pos = int(r["position"])
        wins.append(window(seq, pos, flank))
        ok.append(seq[pos - 1] == r["residue"])
    out["window"], out["residue_matches"] = wins, ok
    return out


def _norm_window(w: str, flank: int = FLANK) -> str | None:
    """Uppercase, drop phospho markers (* or lowercase is kept as the letter), centre crop."""
    if not isinstance(w, str):
        return None
    w = w.replace("*", "").replace("-", "_").strip().upper()
    w = re.sub(r"[^A-Z_]", "", w)
    if len(w) < 2 * flank + 1:
        return w
    mid = len(w) // 2
    return w[mid - flank: mid + flank + 1]


# --------------------------------------------------------------------------------------
# Kinase predictions -> kinase sets
# --------------------------------------------------------------------------------------
def read_kinase_predictions(path: str, sites: pd.DataFrame) -> pd.DataFrame:
    """
    Long table site, kinase, percentile, score, rank for the quantified sites. Rows are
    matched to `sites` (from site_positions / site_windows) by site name, by gene + position,
    or by sequence window. Accepts long or wide Kinase Library-style exports.
    """
    sep = "\t" if open(path, encoding="utf-8").readline().count("\t") else ","
    df = pd.read_csv(path, sep=sep)
    low = {c.lower().strip(): c for c in df.columns}

    def find(*names):
        for n in names:
            if n in low:
                return low[n]
        return None
    c_site = find("site", "site_name", "site name", "phosphosite")
    c_seq = find("sequence", "window", "seq", "site sequence", "peptide")
    c_gene = find("gene", "protein", "gene name", "uniprot", "accession")
    c_pos = find("position", "pos", "site position")
    c_kin = find("kinase", "kinase name", "kinases")
    # match each row to a quantified site
    by_name = {str(s).upper(): s for s in sites["site"]}
    by_gp = {(str(g).upper(), int(p)): s for s, g, p in
             zip(sites["site"], sites["gene"].fillna(""), sites["position"].fillna(-1))}
    by_gp.update({(str(k).upper(), int(p)): s for s, k, p in
                  zip(sites["site"], sites["protein_key"].fillna(""), sites["position"].fillna(-1))})
    by_win = {}
    if "window" in sites:
        for s, w in zip(sites["site"], sites["window"]):
            if isinstance(w, str):
                by_win.setdefault(w, s)

    def match(row):
        if c_site and str(row[c_site]).upper() in by_name:
            return by_name[str(row[c_site]).upper()]
        if c_gene and c_pos and pd.notna(row[c_pos]):
            s = by_gp.get((str(row[c_gene]).upper(), int(row[c_pos])))
            if s:
                return s
        if c_site:
            m = SITE_RE.search("_" + str(row[c_site]) + "_")
            g = str(row[c_gene]).upper() if c_gene else str(row[c_site]).split("_")[0].upper()
            if m:
                s = by_gp.get((g, int(m.group(2))))
                if s:
                    return s
        if c_seq:
            return by_win.get(_norm_window(row[c_seq]))
        return None

    if c_kin:                                                    # long format
        df["_site"] = df.apply(match, axis=1)
        out = pd.DataFrame({"site": df["_site"], "kinase": df[c_kin].astype(str)})
        for name, cands in (("percentile", ("percentile", "site_percentile", "score_percentile")),
                            ("score", ("score", "log2_score", "log2 score")),
                            ("rank", ("rank", "percentile_rank", "score_rank"))):
            c = find(*cands)
            out[name] = pd.to_numeric(df[c], errors="coerce") if c else np.nan
    else:                                                        # wide format
        id_cols = [c for c in (c_site, c_seq, c_gene, c_pos) if c]
        kin_cols = [c for c in df.columns if c not in id_cols
                    and pd.api.types.is_numeric_dtype(df[c])]
        if not kin_cols:
            raise ValueError("kinase prediction file: no 'kinase' column and no numeric "
                             "kinase columns found")
        df["_site"] = df.apply(match, axis=1)
        out = df.melt(id_vars=["_site"], value_vars=kin_cols, var_name="kinase",
                      value_name="percentile").rename(columns={"_site": "site"})
        out["score"] = np.nan
        out["rank"] = out.groupby("site")["percentile"].rank(ascending=False, method="first")
    n_rows = len(df)
    out = out.dropna(subset=["site"])
    if out.empty:
        raise ValueError("no kinase predictions could be matched to quantified sites "
                         "(check site names, gene + position, or sequence windows / FASTA)")
    if out["rank"].isna().all():
        key = "percentile" if out["percentile"].notna().any() else "score"
        out["rank"] = out.groupby("site")[key].rank(ascending=False, method="first")
    _log.debug(f"kinase predictions: {n_rows} rows, matched {out['site'].nunique()} sites, "
               f"{out['kinase'].nunique()} kinases")
    return out.reset_index(drop=True)


def kinase_sets(pred: pd.DataFrame, percentile_min: float = 90.0, top_n: int = 15,
                min_size: int = 3) -> dict:
    """{kinase: set of sites} from predictions (percentile and per-site rank thresholds)."""
    keep = pred["rank"].fillna(np.inf) <= top_n
    if pred["percentile"].notna().any():
        keep &= pred["percentile"].fillna(-np.inf) >= percentile_min
    sets = pred[keep].groupby("kinase")["site"].apply(set).to_dict()
    return {k: v for k, v in sets.items() if len(v) >= min_size}


# --------------------------------------------------------------------------------------
# Activity inference
# --------------------------------------------------------------------------------------
def ksea(estimates: pd.Series, sets: dict, min_size: int = 3) -> pd.DataFrame:
    """KSEA z-scores of the mean substrate log2 FC against all sites."""
    est = estimates.dropna()
    mu, sd = est.mean(), est.std(ddof=1)
    rows = []
    for k, members in sets.items():
        sub = est.reindex([s for s in members if s in est.index]).dropna()
        m = len(sub)
        if m < min_size or sd == 0:
            continue
        z = (sub.mean() - mu) * np.sqrt(m) / sd
        rows.append({"kinase": k, "n_sites": m, "mean_log2fc": sub.mean(), "z": z,
                     "pvalue": 2 * stats.norm.sf(abs(z)),
                     "direction": "Up" if z > 0 else "Down",
                     "sites": ";".join(sub.sort_values(ascending=z < 0).index[:10])})
    out = pd.DataFrame(rows)
    if len(out):
        out["padj"] = benjamini_hochberg(out["pvalue"].to_numpy())
        out = out.sort_values("pvalue").reset_index(drop=True)
    return out


def kinase_activity(results: pd.DataFrame, sets: dict, methods=("ksea", "camera"),
                    model=None, features=None, estimate_col="estimate", pvalue_col="pvalue",
                    padj_col="padj", min_size: int = 3) -> dict:
    """
    {method: table} for one contrast. results: site table indexed by site name with
    estimate / pvalue / padj columns (for phospho: the protein-adjusted columns renamed).
    model / features: the fitted site model, for CAMERA's inter-site correlation.
    """
    from .enrichment import run_enrichment
    r = results.rename(columns={estimate_col: "estimate", pvalue_col: "pvalue",
                                padj_col: "padj"})
    out = {}
    if "ksea" in methods:
        out["ksea"] = ksea(r["estimate"], sets, min_size)
    if "camera" in methods:
        upper = {k: {str(s).upper() for s in v} for k, v in sets.items()}
        res = run_enrichment(r[["estimate", "pvalue", "padj"]], list(r.index),
                             {"kinases": upper}, model=model, features=features,
                             methods=("camera",), min_size=min_size, max_size=10 ** 6)
        cam = res.get(("kinases", "camera"), pd.DataFrame())
        if len(cam):
            cam = cam.rename(columns={"term": "kinase", "n_genes": "n_sites",
                                      "leading_genes": "sites"})
        out["camera"] = cam
    return out


# --------------------------------------------------------------------------------------
# Motifs
# --------------------------------------------------------------------------------------
MOTIFS = {
    "proline-directed (S/T-P)": lambda w: w[FLANK] in "ST" and w[FLANK + 1] == "P",
    "basophilic (R/K at -3)": lambda w: w[FLANK] in "ST" and w[FLANK - 3] in "RK",
    "acidophilic (D/E at +1..+3)": lambda w: w[FLANK] in "ST" and
    sum(c in "DE" for c in w[FLANK + 1: FLANK + 4]) >= 2,
    "tyrosine": lambda w: w[FLANK] == "Y",
}


def motif_table(windows: pd.Series) -> pd.DataFrame:
    """Boolean motif membership per site (index = site)."""
    rows = {}
    for s, w in windows.items():
        if isinstance(w, str) and len(w) == 2 * FLANK + 1:
            rows[s] = {m: bool(f(w)) for m, f in MOTIFS.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def cluster_enrichment(assignments: pd.Series, membership: pd.DataFrame,
                       universe) -> pd.DataFrame:
    """
    Fisher's exact test per cluster and feature (motif or kinase set, boolean columns of
    membership indexed by site), against all sites in `universe`. BH within the table.
    """
    uni = [s for s in universe if s in membership.index]
    M = membership.loc[uni]
    rows = []
    for c in sorted(assignments.unique()):
        inc = set(assignments.index[assignments == c]) & set(uni)
        if not inc:
            continue
        in_c = M.index.isin(list(inc))
        for feat in M.columns:
            f = M[feat].to_numpy(bool)
            a, b = int((f & in_c).sum()), int((~f & in_c).sum())
            cc, d = int((f & ~in_c).sum()), int((~f & ~in_c).sum())
            if a + cc == 0:
                continue
            odds, p = stats.fisher_exact([[a, b], [cc, d]], alternative="greater")
            rows.append({"cluster": c, "feature": feat, "in_cluster": a,
                         "cluster_size": a + b, "fraction_in_cluster": a / max(a + b, 1),
                         "fraction_background": (a + cc) / len(uni), "odds_ratio": odds,
                         "pvalue": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["padj"] = benjamini_hochberg(out["pvalue"].to_numpy())
        out = out.sort_values(["cluster", "pvalue"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def plot_kinase_heatmap(tables: dict, method: str, alpha: float = 0.05, top: int = 30,
                        font_size: float = 16, figsize=None):
    """
    Kinase x contrast heatmap of activity (KSEA z, or signed -log10 p for CAMERA); cells
    with FDR < alpha marked with a dot. tables: {contrast: activity table}.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from .figures import publication_style
    frames = []
    for contrast, t in tables.items():
        if t is None or not len(t):
            continue
        score = t["z"] if method == "ksea" else \
            -np.log10(t["pvalue"].clip(lower=1e-300)) * np.where(t["direction"] == "Up", 1, -1)
        frames.append(pd.DataFrame({"kinase": t["kinase"], "contrast": contrast,
                                    "score": score.to_numpy(), "padj": t["padj"].to_numpy()}))
    if not frames:
        raise ValueError("no kinase activity results to plot")
    d = pd.concat(frames)
    S = d.pivot_table(index="kinase", columns="contrast", values="score")
    Q = d.pivot_table(index="kinase", columns="contrast", values="padj")
    S = S[[c for c in tables if c in S.columns]]
    Q = Q.reindex_like(S)
    keep = S.abs().max(1).sort_values(ascending=False).index[:top]
    S, Q = S.loc[keep], Q.loc[keep]
    S = S.loc[S.mean(1).sort_values(ascending=False).index]
    Q = Q.loc[S.index]
    # colour scale capped so one extreme kinase does not wash out the rest
    vals = np.abs(S.to_numpy()[np.isfinite(S.to_numpy())])
    lim = max(min(np.nanmax(vals), 10.0 if method == "camera" else 8.0), 2.0) if len(vals) else 2.0
    figsize = figsize or (max(5, 1.1 * S.shape[1] + 4), max(4, 0.36 * S.shape[0] + 2))
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(S.to_numpy(), cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim),
                       aspect="auto")
        yy, xx = np.where(Q.to_numpy() < alpha)
        ax.scatter(xx, yy, s=18, color="black")
        ax.set_xticks(range(S.shape[1]))
        ax.set_xticklabels(S.columns, rotation=45, ha="right", fontsize=font_size - 4)
        ax.set_yticks(range(S.shape[0]))
        ax.set_yticklabels(S.index, fontsize=font_size - 4)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(("KSEA z" if method == "ksea" else "signed −log$_{10}$ p")
                     + f"\n(capped at ±{lim:g})")
        cb.outline.set_visible(False)
        ax.set_title(f"Inferred kinase activity ({method.upper()}; dot = FDR < {alpha:g})",
                     loc="left", fontsize=font_size)
        fig.tight_layout()
    return fig
