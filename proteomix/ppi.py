"""
Protein-protein interaction (PPI) analysis of differential abundance results.

Networks come from local files (reproducible, no web access needed):
    STRING   <taxon>.protein.links.<version>.txt.gz (protein1 protein2 combined_score) plus
             <taxon>.protein.info.<version>.txt.gz to map STRING IDs to gene symbols
             (e.g. 9606.* for human, 10090.* for mouse, from the STRING download page)
    any edge list with two gene-symbol columns and an optional score column (.tsv / .csv)

ppi_analysis, per contrast:
  * network: interactions (score >= min_score, STRING 700 = high confidence) among the tested
    proteins only, so the background is what could have been detected;
  * PPI enrichment: are the hits more connected among themselves than expected? The null
    draws random protein sets matched to the hits' degrees (degree bins), because well-studied
    proteins have both more recorded interactions and a higher chance of detection; a plain
    random null makes almost any hit list look "enriched";
  * modules: connected groups of hits (with networkx, split further by greedy modularity);
  * node and edge tables (for Cytoscape: import edges.tsv as a network, nodes.tsv as node
    attributes) and a GraphML file when networkx is installed.
"""
import gzip
import logging
import os

import numpy as np
import pandas as pd
from scipy import sparse

_log = logging.getLogger("proteomix")


# --------------------------------------------------------------------------------------
# Reading networks
# --------------------------------------------------------------------------------------
def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else \
        open(path, encoding="utf-8")


def read_string(links_path: str, info_path: str | None = None, min_score: int = 700) -> pd.DataFrame:
    """STRING links (+ info for symbols) -> undirected edges gene_a, gene_b, score."""
    with _open(links_path) as fh:
        links = pd.read_csv(fh, sep=r"\s+", engine="c")
    cols = list(links.columns)
    score_col = "combined_score" if "combined_score" in cols else cols[-1]
    links = links[links[score_col] >= min_score]
    a, b = links[cols[0]].astype(str), links[cols[1]].astype(str)
    if info_path:
        with _open(info_path) as fh:
            info = pd.read_csv(fh, sep="\t")
        id_col = info.columns[0]
        name_col = "preferred_name" if "preferred_name" in info else info.columns[1]
        m = dict(zip(info[id_col].astype(str), info[name_col].astype(str)))
        a, b = a.map(m), b.map(m)
    else:
        a, b = a.str.split(".", n=1).str[-1], b.str.split(".", n=1).str[-1]
    edges = pd.DataFrame({"gene_a": a.str.upper(), "gene_b": b.str.upper(),
                          "score": links[score_col].to_numpy()}).dropna()
    return _undirected(edges)


def read_edge_list(path: str, min_score: float | None = None) -> pd.DataFrame:
    """Two gene-symbol columns and an optional numeric score column."""
    sep = "\t" if path.lower().endswith((".tsv", ".txt", ".tsv.gz", ".txt.gz")) else ","
    with _open(path) as fh:
        t = pd.read_csv(fh, sep=sep)
    edges = pd.DataFrame({"gene_a": t.iloc[:, 0].astype(str).str.upper(),
                          "gene_b": t.iloc[:, 1].astype(str).str.upper(),
                          "score": pd.to_numeric(t.iloc[:, 2], errors="coerce")
                          if t.shape[1] > 2 else 1.0})
    if min_score is not None and t.shape[1] > 2:
        edges = edges[edges["score"] >= min_score]
    return _undirected(edges)


def read_network(path: str, info_path: str | None = None, min_score: float = 700) -> pd.DataFrame:
    """STRING file (name contains 'protein.links') or a generic edge list."""
    if "protein.links" in os.path.basename(path):
        return read_string(path, info_path, int(min_score))
    return read_edge_list(path, None if min_score is None else min_score)


def _undirected(edges):
    e = edges[edges["gene_a"] != edges["gene_b"]].copy()
    lo = np.where(e["gene_a"] < e["gene_b"], e["gene_a"], e["gene_b"])
    hi = np.where(e["gene_a"] < e["gene_b"], e["gene_b"], e["gene_a"])
    e["gene_a"], e["gene_b"] = lo, hi
    return e.groupby(["gene_a", "gene_b"], as_index=False)["score"].max()


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------
def _degree_matched_null(A, hits_idx, n_perm, rng, n_bins=10):
    deg = np.asarray(A.sum(1)).ravel()
    ranks = pd.Series(deg).rank(method="first").to_numpy()
    bins = np.minimum((ranks - 1) * n_bins // len(deg), n_bins - 1).astype(int)
    members = [np.where(bins == b)[0] for b in range(n_bins)]
    need = np.bincount(bins[hits_idx], minlength=n_bins)
    counts = np.empty(n_perm)
    for i in range(n_perm):
        pick = np.concatenate([rng.choice(members[b], need[b], replace=False)
                               for b in range(n_bins) if need[b]])
        x = np.zeros(A.shape[0])
        x[pick] = 1
        counts[i] = x @ (A @ x) / 2
    return counts


def ppi_analysis(results: pd.DataFrame, genes, edges: pd.DataFrame, alpha: float = 0.05,
                 lfc: float = 0.0, n_perm: int = 2000, seed: int = 0,
                 min_module_size: int = 3) -> dict:
    """
    results: one contrast's table; genes: gene symbol per row. Hits: padj < alpha and
    |estimate| >= lfc. Returns dict with summary (one-row DataFrame), nodes, edges (network
    among hits), modules and the tested network.
    """
    from .enrichment import gene_level_stats
    gs = gene_level_stats(results, genes)
    tested = set(gs.index)
    e = edges[edges["gene_a"].isin(tested) & edges["gene_b"].isin(tested)]
    nodes = sorted(set(e["gene_a"]) | set(e["gene_b"]))
    pos = {g: i for i, g in enumerate(nodes)}
    n = len(nodes)
    hits = gs.index[(gs["padj"] < alpha) & (gs["estimate"].abs() >= lfc)]
    hits_in = np.array([pos[g] for g in hits if g in pos], dtype=int)

    summary = {"tested_genes": len(tested), "genes_in_network": n, "edges_in_network": len(e),
               "hits": len(hits), "hits_in_network": len(hits_in)}
    if n == 0 or len(hits_in) < 2:
        summary.update(observed_edges=0, expected_edges=np.nan, enrichment=np.nan,
                       pvalue=np.nan)
        _log.debug(f"ppi: {summary}")
        return {"summary": pd.DataFrame([summary]), "nodes": pd.DataFrame(),
                "edges": pd.DataFrame(columns=["gene_a", "gene_b", "score"]),
                "modules": pd.DataFrame(), "network": e}

    i = e["gene_a"].map(pos).to_numpy()
    j = e["gene_b"].map(pos).to_numpy()
    A = sparse.coo_matrix((np.ones(len(e)), (i, j)), shape=(n, n)).tocsr()
    A = A + A.T
    x = np.zeros(n)
    x[hits_in] = 1
    observed = x @ (A @ x) / 2
    null = _degree_matched_null(A, hits_in, n_perm, np.random.default_rng(seed))
    p = (1 + (null >= observed).sum()) / (1 + n_perm)
    summary.update(observed_edges=int(observed), expected_edges=float(null.mean()),
                   enrichment=float(observed / null.mean()) if null.mean() > 0 else np.nan,
                   pvalue=float(p))

    hit_set = set(hits)
    he = e[e["gene_a"].isin(hit_set) & e["gene_b"].isin(hit_set)].reset_index(drop=True)
    deg = np.asarray(A.sum(1)).ravel()
    node_tab = gs.loc[[g for g in hits]].copy()
    node_tab["degree_tested_network"] = [int(deg[pos[g]]) if g in pos else 0
                                         for g in node_tab.index]
    hit_deg = pd.concat([he["gene_a"], he["gene_b"]]).value_counts()
    node_tab["degree_among_hits"] = hit_deg.reindex(node_tab.index).fillna(0).astype(int)
    node_tab["module"] = _modules(node_tab.index, he, min_module_size)
    mods = (node_tab[node_tab["module"] > 0].groupby("module")
            .agg(size=("estimate", "size"), mean_log2fc=("estimate", "mean"),
                 genes=("estimate", lambda s: ";".join(s.index)))
            .sort_values("size", ascending=False).reset_index())
    _log.debug(f"ppi: {summary}; {len(mods)} modules")
    return {"summary": pd.DataFrame([summary]), "nodes": node_tab.reset_index(),
            "edges": he, "modules": mods, "network": e}


def _modules(genes, edges, min_size):
    """Module id per gene (0 = not in a module of at least min_size)."""
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(genes)
        G.add_edges_from(zip(edges["gene_a"], edges["gene_b"]))
        comms = []
        for comp in nx.connected_components(G):
            if len(comp) < min_size:
                continue
            sub = G.subgraph(comp)
            if len(comp) > 15:
                comms += [set(c) for c in nx.community.greedy_modularity_communities(sub)]
            else:
                comms.append(set(comp))
    except ImportError:                                   # connected components only
        parent = {g: g for g in genes}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for a, b in zip(edges["gene_a"], edges["gene_b"]):
            parent[find(a)] = find(b)
        groups = {}
        for g in genes:
            groups.setdefault(find(g), set()).add(g)
        comms = list(groups.values())
    comms = sorted([c for c in comms if len(c) >= min_size], key=len, reverse=True)
    label = {g: k + 1 for k, c in enumerate(comms) for g in c}
    return [label.get(g, 0) for g in genes]


def write_network(res: dict, prefix: str) -> list:
    """nodes.tsv, edges.tsv, modules.tsv, summary.tsv (+ .graphml with networkx)."""
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    paths = []
    for key in ("summary", "nodes", "edges", "modules"):
        path = f"{prefix}_{key}.tsv"
        res[key].to_csv(path, sep="\t", index=False)
        paths.append(path)
    try:
        import networkx as nx
        G = nx.Graph()
        for _, r in res["nodes"].iterrows():
            G.add_node(r["gene"], log2fc=float(r["estimate"]), padj=float(r["padj"]),
                       module=int(r["module"]))
        for _, r in res["edges"].iterrows():
            G.add_edge(r["gene_a"], r["gene_b"], score=float(r["score"]))
        path = f"{prefix}.graphml"
        nx.write_graphml(G, path)
        paths.append(path)
    except ImportError:
        pass
    return paths
