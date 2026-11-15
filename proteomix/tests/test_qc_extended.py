"""Extended QC checks (qc_extended.py)."""
import numpy as np
import pandas as pd
import pytest

from proteomix import qc_extended as qe


def test_contaminants_kept_for_qc_and_share(fragpipe_files):
    from proteomix.fragpipe import build_inputs, read_fragpipe
    from proteomix.metadata import read_run_metadata
    data = read_fragpipe(str(fragpipe_files["protein"]))
    assert data.contaminants is not None and len(data.contaminants) == 1
    inp = build_inputs(data, read_run_metadata(str(fragpipe_files["metadata"])), key="run",
                       group_col="condition")
    t, top = qe.contaminant_share(inp.Y, inp.extra["contaminants"], inp.run_names,
                                  ["g"] * len(inp.run_names))
    assert (t["contaminant_pct"] > 0).all() and (t["contaminant_pct"] < 20).all()
    assert top.loc[0, "contaminant"] == "GENE0"


def test_low_identifications_flagged(rng):
    Y = rng.normal(20, 1, (12, 300))
    Y[rng.random(Y.shape) < 0.1] = np.nan
    Y[5, :150] = np.nan
    t = qe.identifications_per_run(Y, [f"r{i}" for i in range(12)], ["a"] * 6 + ["b"] * 6)
    assert t["low_identifications"].tolist() == [i == 5 for i in range(12)]


def test_cv_and_replicate_pairs(rng):
    base = rng.normal(20, 1.5, 200)
    Ys = base + rng.normal(0, 0.2, (8, 200))
    long, summ = qe.coefficient_of_variation(Ys, ["a"] * 4 + ["b"] * 4)
    assert 5 < summ.loc["a", "median_cv_pct"] < 25          # ~0.2 log2 SD = ~14% CV
    Yr = np.repeat(Ys, 2, axis=0) + rng.normal(0, 0.05, (16, 200))
    pairs = qe.replicate_pairs(Yr, np.repeat([f"s{i}" for i in range(8)], 2),
                               [f"r{i}" for i in range(16)])
    assert len(pairs) == 8 and (pairs["pearson_r"] > 0.95).all()


def test_mislabel_detected_only_when_swapped(rng):
    base = rng.normal(20, 2, 400)
    effect = np.where(rng.random(400) < 0.3, rng.choice([-2, 2], 400), 0)
    g = np.repeat(["A", "B"], 6)
    Y = base + (g == "B")[:, None] * effect + rng.normal(0, 0.3, (12, 400))
    names = [f"s{i}" for i in range(12)]
    _, _, nn = qe.sample_correlation(Y, g, names)
    assert not nn["possible_mislabel"].any()
    g_bad = g.copy()
    g_bad[0], g_bad[11] = "B", "A"
    _, _, nn = qe.sample_correlation(Y, g_bad, names)
    assert set(nn.loc[nn["possible_mislabel"], "sample"]) == {"s0", "s11"}


def test_run_order_drift_detected(rng):
    n = 30
    Y = rng.normal(20, 1, (n, 200)) + np.linspace(0, 1.5, n)[:, None]   # steady drift
    runs = pd.DataFrame({"run": [f"r{i}" for i in range(n)], "acquisition": np.arange(n)})
    _, tests = qe.run_order_effects(Y, runs, np.tile(["a", "b", "c"], n // 3))
    assert tests.set_index("metric").loc["median_log2", "drift"]
    _, tests = qe.run_order_effects(rng.normal(20, 1, (n, 200)), runs,
                                    np.tile(["a", "b", "c"], n // 3))
    assert not tests.set_index("metric").loc["median_log2", "drift"]


def test_confounding_detected():
    runs = pd.DataFrame({"run": [f"r{i}" for i in range(12)],
                         "condition": np.repeat(["a", "b", "c", "d"], 3),
                         "tray": np.repeat(["S1", "S3", "S1", "S3"], 3),
                         "batch": np.tile(["x", "y", "z"], 4),
                         "acquisition": np.arange(12)})
    t, tabs = qe.confounding_check(runs, factors=["tray", "batch"])
    t = t.set_index("factor")
    assert t.loc["tray", "confounded"] and t.loc["tray", "each_condition_single_level"]
    assert not t.loc["batch", "confounded"]


def test_bait_recovery_flags_failed_pulldown(rng):
    genes = ["BAIT"] + [f"G{i}" for i in range(99)]
    Y = rng.normal(20, 1, (10, 100))
    pulldown = ["IP"] * 6 + ["IgG"] * 4
    Y[:6, 0] = 25 + rng.normal(0, 0.2, 6)
    Y[6:, 0] = np.nan
    Y[2, 0] = np.nan                                          # failed IP
    t, s = qe.bait_recovery(Y, genes, "bait", [f"s{i}" for i in range(10)],
                            ["g"] * 10, pulldown)
    assert t["failed_pulldown"].tolist() == [i == 2 for i in range(10)]
    assert s["ip_detected"] == 5 and s["control_detected"] == 0
    with pytest.raises(ValueError):
        qe.bait_recovery(Y, genes, "NOPE", [f"s{i}" for i in range(10)], ["g"] * 10, pulldown)


def test_igg_background(rng):
    Y = rng.normal(20, 1, (8, 50))
    Y[4:, 10:] = np.nan                                       # IgG sees only 10 proteins
    per, bg = qe.igg_background(Y, [f"G{i}" for i in range(50)], [f"s{i}" for i in range(8)],
                                ["IP"] * 4 + ["IgG"] * 4)
    assert per.loc[per["pulldown"] == "IgG", "n_quantified"].eq(10).all()
    assert set(bg["gene"]) == {f"G{i}" for i in range(10)}


def test_phospho_residues_and_enrichment(tmp_path):
    names = ["GENE1_S12", "GENE1_T40", "GENE2_Y7", "GENE3_S101"]
    Y = np.array([[20, 21, np.nan, 19], [20, np.nan, 18, 19.5]])
    r = qe.site_residues(Y, names, ["run1", "run2"]).set_index("run")
    assert np.isclose(r.loc["run1", "S_pct"], 200 / 3) and np.isclose(r.loc["run2", "Y_pct"], 100 / 3)
    pep = pd.DataFrame({"Modified Sequence": ["AS[167]PK", "APEPK", "T[181]K"],
                        "run1 Intensity": [100.0, 100.0, 0.0], "run2 Intensity": [10, 90, 0]})
    path = tmp_path / "pep.tsv"
    pep.to_csv(path, sep="\t", index=False)
    e = qe.enrichment_efficiency(str(path), ["run1", "run2"]).set_index("run")
    assert np.isclose(e.loc["run1", "phospho_intensity_pct"], 50)
    assert np.isclose(e.loc["run2", "phospho_peptides_pct"], 50)
