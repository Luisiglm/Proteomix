"""FragPipe reading, QC, export, enrichment, PPI, PTM and time-course modules."""
import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import integrate, stats
from scipy.special import expit

from proteomix.export import conditional_missing_mean, draw_missing, export_matrix
from proteomix.fragpipe import build_inputs, parse_run_names, read_fragpipe
from proteomix.metadata import condition_order, read_run_metadata


# ---- FragPipe reader ---------------------------------------------------------------------
def test_read_fragpipe(fragpipe_files):
    data = read_fragpipe(str(fragpipe_files["protein"]))
    assert data.intensities.shape == (148, 24)               # contaminant + decoy removed
    assert not (data.intensities == 0).any().any()            # zeros became NaN
    assert data.runs["sample"].nunique() == 12
    assert data.runs["acquisition"].is_monotonic_increasing
    assert data.proteins["feature_name"].is_unique


def test_parse_run_names():
    t = parse_run_names(["230801kw_MKreileder_117_PD1_A10_S3_B5_1_16061"])
    assert t.loc[0, ["sample", "tray", "well", "acquisition"]].tolist() == ["A10", "S3", "B5", 16061]


def test_build_inputs_design(fragpipe_files):
    data = read_fragpipe(str(fragpipe_files["protein"]))
    meta = read_run_metadata(str(fragpipe_files["metadata"]))
    inp = build_inputs(data, meta, key="run", group_col="condition",
                       group_order=condition_order(meta))
    assert inp.Y.shape == (24, 148)
    assert inp.group_names[0] == "IP unstimulated"
    assert inp.X.sum(0).tolist() == [6, 6, 6, 6]
    assert len(inp.sample_names) == 12
    assert np.isclose(np.nanmean(inp.Y.numpy()), 22.4, atol=1.0)  # log2 scale


# ---- export / model-based imputation -------------------------------------------------------
def test_conditional_missing_mean_matches_integral():
    for mu, s, w, r in [(18, 0.5, -1.5, 19), (21, 0.4, -1.5, 19), (17, 1.0, -2, 19)]:
        lo, hi = mu - 12 * s, mu + 12 * s
        f = lambda v: stats.norm.pdf(v, mu, s) * expit(w * (v - r))  # noqa: E731
        exact = integrate.quad(lambda v: v * f(v), lo, hi, points=[mu])[0] / \
            integrate.quad(f, lo, hi, points=[mu])[0]
        assert abs(conditional_missing_mean(mu, s, w, r) - exact) < 1e-3


def test_draws_follow_the_conditional_distribution():
    rng = np.random.default_rng(0)
    n = 20000
    d = draw_missing(np.full(n, 20.0), np.full(n, 0.6), np.full(n, -1.5), np.full(n, 19.5), rng)
    assert abs(d.mean() - conditional_missing_mean(20.0, 0.6, -1.5, 19.5)) < 0.02


def test_export_model_is_less_biased_than_mean_imputation(two_group_data):
    from proteomix.fragpipe import ProteomixInputs
    d = two_group_data
    S, F = d["Y"].shape
    names = [f"P{i}" for i in range(F)]
    runs = pd.DataFrame({"run": [f"R{i}" for i in range(S)], "sample": [f"S{i}" for i in range(S)]})
    inp = ProteomixInputs(Y=d["Y"], X=d["X"], batch_ids=d["batch"],
                          replicate_ids=torch.arange(S), run_names=list(runs["run"]),
                          group_names=["A", "B"], batch_names=["all"], feature_names=names,
                          sample_names=list(runs["sample"]), runs=runs,
                          proteins=pd.DataFrame(index=names))
    errs = {}
    for method in ("model", "mean"):
        ex = export_matrix(inp, method=method, normalisation="none")
        cols = [int(c[1:]) for c in ex.values.columns]
        mask = ex.imputed.to_numpy()
        errs[method] = np.mean((ex.values.to_numpy() - d["true"][:, cols])[mask])
        assert not ex.values.isna().any().any()
    assert abs(errs["model"]) < 0.15
    assert abs(errs["model"]) < abs(errs["mean"])


# ---- QC --------------------------------------------------------------------------------------
def test_qc_flags_injected_outlier_only(rng):
    from proteomix.qc import sample_qc
    g = np.repeat(["a", "b", "c"], 5)
    y = rng.normal(20, 2, 400) + rng.normal(0, 0.2, (15, 400))
    clean = sample_qc(y, g, [f"s{i}" for i in range(15)])
    assert not clean["outlier"].any()
    y[7] += rng.normal(0, 1.2, 400)
    flagged = sample_qc(y, g, [f"s{i}" for i in range(15)])
    assert flagged["outlier"].tolist() == [i == 7 for i in range(15)]


# ---- enrichment ------------------------------------------------------------------------------
def test_ora_matches_hypergeometric():
    from proteomix.enrichment import ora
    universe = {f"G{i}" for i in range(100)}
    sets = {"S": {f"G{i}" for i in range(20)}}
    hits = {f"G{i}" for i in range(10)} | {"G50", "G51"}
    t = ora(hits, universe, sets)
    assert t.loc[0, "overlap"] == 10
    assert np.isclose(t.loc[0, "pvalue"], stats.hypergeom.sf(9, 100, 20, 12))


def test_camera_finds_planted_set_and_estimates_correlation(rng):
    from proteomix.enrichment import camera
    G, n = 600, 12
    genes = [f"G{i}" for i in range(G)]
    z = rng.normal(0, 1, G)
    z[:30] += 2.5
    res = pd.DataFrame({"z": z, "padj": 1.0}, index=genes)
    shared = rng.normal(0, 1, (n, 1))
    R = rng.normal(0, 1, (n, G))
    R[:, 30:60] = 0.7 * shared + 0.3 * R[:, 30:60]           # correlated null set
    sets = {"UP": set(genes[:30]), "COR": set(genes[30:60]), "RND": set(genes[100:140])}
    t = camera(res, sets, R, df_resid=n - 1).set_index("term")
    assert t.loc["UP", "padj"] < 0.01 and t.loc["UP", "direction"] == "Up"
    assert t.loc["COR", "correlation"] > 0.5 and t.loc["RND", "correlation"] < 0.2


# ---- PPI ---------------------------------------------------------------------------------------
def test_ppi_enrichment_detects_dense_hits(rng):
    from proteomix.ppi import _undirected, ppi_analysis
    genes = [f"G{i}" for i in range(300)]
    e = [(genes[i], genes[j]) for i in range(20) for j in range(i + 1, 20) if rng.random() < 0.4]
    e += [(genes[a], genes[b]) for a, b in rng.integers(0, 300, (400, 2)) if a != b]
    edges = _undirected(pd.DataFrame(e, columns=["gene_a", "gene_b"]).assign(score=900))
    res = pd.DataFrame({"estimate": 1.0, "pvalue": 0.5, "padj": 0.5}, index=genes)
    res.iloc[:20, 1:] = 1e-6
    out = ppi_analysis(res, genes, edges, n_perm=500)
    assert out["summary"].loc[0, "pvalue"] < 0.01
    assert len(out["modules"]) >= 1
    res.iloc[:20, 1:] = 0.5
    res.iloc[200:220, 1:] = 1e-6                               # random hits
    assert ppi_analysis(res, genes, edges, n_perm=500)["summary"].loc[0, "pvalue"] > 0.05


# ---- PTM ---------------------------------------------------------------------------------------
@pytest.mark.slow
def test_paired_ptm_controls_protein_driven_sites():
    from proteomix.ptm import evaluate_ptm_methods, run_paired_ptm
    from proteomix.simulation import paired_ptm_inputs, simulate_paired_ptm
    sim = simulate_paired_ptm(seed=11, n_proteins=600, shared_sd=0.4)
    si, pi = paired_ptm_inputs(sim)
    res = run_paired_ptm(si, pi, {"c": ("Treatment", "Control")})
    rows = {r["method"]: r for r in evaluate_ptm_methods(res, sim, si, pi, "c")}
    paired = rows["paired (protein offset)"]
    assert paired["fpr_protein_driven"] < 0.12
    assert rows["unadjusted"]["fpr_protein_driven"] > 0.5
    assert paired["power"] > rows["independent (Welch-Satterthwaite)"]["power"]


def test_independent_combination_formula():
    from proteomix.ptm import combine_independent
    site = pd.DataFrame({"estimate": [1.0], "se": [0.3], "df": [6.0]}, index=["s"])
    prot = pd.DataFrame({"estimate": [0.4], "se": [0.4], "df": [10.0]}, index=["p"])
    t = combine_independent(site, prot, ["p"], ["p"])
    v = 0.3 ** 2 + 0.4 ** 2
    assert np.isclose(t.loc["s", "estimate"], 0.6)
    assert np.isclose(t.loc["s", "se"], math.sqrt(v))
    assert np.isclose(t.loc["s", "df"], v ** 2 / (0.3 ** 4 / 6 + 0.4 ** 4 / 10))


# ---- time course -------------------------------------------------------------------------------
def test_time_design_parsing():
    from proteomix.timecourse import time_design
    d = time_design(["Ctrl 5 min", "Ctrl 4h", "PDL1 20min", "Baseline"], baseline="Baseline")
    assert d["time"].tolist() == [5, 240, 20, 0]
    assert d["series"].tolist() == ["Ctrl", "Ctrl", "PDL1", "Baseline"]
    assert d["baseline"].tolist() == [False, False, False, True]


@pytest.mark.parametrize("method", ["kmeans", "fuzzy"])
def test_clustering_recovers_known_shapes(method, rng):
    from proteomix.timecourse import cluster_profiles
    cols = pd.MultiIndex.from_tuples([(s, t) for s in ("Ctrl", "PDL1") for t in (0, 5, 20, 240)],
                                     names=["series", "time"])
    shapes = [[0, 0, 0, 0, 0, 2, .5, 0], [0, 0, .3, 1.5, 0, 0, .3, 1.5],
              [0, 0, 0, 0, 0, 1, 1.5, 1.5], [0, -1, -1.5, -1.5, 0, -1, -1.5, -1.5]]
    rows, truth = [], []
    for k, sh in enumerate(shapes):
        for _ in range(40):
            rows.append(20 + np.array(sh) * rng.uniform(.7, 1.5) + rng.normal(0, .25, 8))
            truth.append(k)
    P = pd.DataFrame(rows, columns=cols, index=[f"P{i}" for i in range(len(rows))])
    tc = cluster_profiles(P, k="auto", method=method)
    lab = tc.assignments["cluster"].reindex(P.index).to_numpy()
    truth = np.array(truth)
    assert tc.k == 4
    assert all(len(set(truth[lab == c])) == 1 for c in np.unique(lab))
