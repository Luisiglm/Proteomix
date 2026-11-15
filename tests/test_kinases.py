"""Kinase activity inference, sequence windows, motifs and site clustering (kinases.py)."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from proteomix import kinases as kn

AA = "ACDEFGHIKLMNPQRSTVWY"


def _phospho_files(tmp_path):
    """
    Synthetic phospho time course: Control and Stim at 5 / 20 / 60 min, 4 samples each.
    Planted: 'AKT1' substrates (basophilic R-x-x-S) up at 20 min, 'CDK1' substrates
    (proline-directed S-P) down at 60 min. Kinase predictions in long format.
    """
    rng = np.random.default_rng(5)
    groups = ["Control", "Stim 5 min", "Stim 20 min", "Stim 60 min"]
    n_prot, L = 80, 300
    seqs, sites = {}, []
    for p in range(n_prot):
        seq = list(rng.choice(list(AA), L))
        for k in range(2):
            pos = int(rng.integers(20, L - 20))
            kind = rng.choice(["akt", "cdk", "none"], p=[0.2, 0.2, 0.6])
            seq[pos - 1] = "S"
            if kind == "akt":
                seq[pos - 4] = "R"
            if kind == "cdk":
                seq[pos] = "P"
            sites.append((f"G{p}", f"P{p:05d}", pos, kind))
        seqs[p] = "".join(seq)
    sites = pd.DataFrame(sites, columns=["gene", "acc", "pos", "kind"]).drop_duplicates(
        ["gene", "pos"])
    with open(tmp_path / "prot.fasta", "w") as fh:
        for p, s in seqs.items():
            fh.write(f">sp|P{p:05d}|G{p}_HUMAN test OS=Homo sapiens GN=G{p}\n{s}\n")
    run = lambda layer, gi, i: f"x_{layer}_{'ABCD'[i]}{gi + 1}_S1_A1_1_{100 * gi + i}"  # noqa
    base = rng.normal(20, 1.5, len(sites))
    prot_base = rng.normal(22, 1.5, n_prot)
    site_cols, prot_cols, meta = {}, {}, []
    for gi, g in enumerate(groups):
        for i in range(4):
            eff = np.where((sites["kind"] == "akt") & (g == "Stim 20 min"), 2.0, 0.0) + \
                np.where((sites["kind"] == "cdk") & (g == "Stim 60 min"), -2.0, 0.0)
            shared = rng.normal(0, 0.2, n_prot)
            prot_of_site = sites["gene"].str[1:].astype(int).to_numpy()
            y = base + eff + shared[prot_of_site] + rng.normal(0, 0.25, len(sites))
            miss = rng.random(len(y)) < expit(-1.5 * (y - 17))
            site_cols[f"{run('PH', gi, i)} Intensity"] = np.where(miss, 0, 2 ** y)
            prot_cols[f"{run('WP', gi, i)} MaxLFQ Intensity"] = 2 ** (prot_base + shared)
            meta += [f"{run('PH', gi, i)} Intensity\t{g}", f"{run('WP', gi, i)} MaxLFQ Intensity\t{g}"]
    st = pd.DataFrame({"Index": [f"{a}_S{p}" for a, p in zip(sites["acc"], sites["pos"])],
                       "Protein ID": sites["acc"], "Gene": sites["gene"],
                       "Best Localization Probability": 0.99, **site_cols})
    st.to_csv(tmp_path / "sites.tsv", sep="\t", index=False)
    pt = pd.DataFrame({"Protein": [f"sp|P{p:05d}|G{p}_HUMAN" for p in range(n_prot)],
                       "Protein ID": [f"P{p:05d}" for p in range(n_prot)],
                       "Gene": [f"G{p}" for p in range(n_prot)], **prot_cols})
    pt.to_csv(tmp_path / "proteins.tsv", sep="\t", index=False)
    (tmp_path / "meta.tsv").write_text("\n".join(meta) + "\n")
    # Kinase Library-style long predictions keyed by sequence window
    rows = []
    for _, r in sites.iterrows():
        w = kn.window(seqs[int(r["gene"][1:])], int(r["pos"]))
        w = w[:7] + w[7].lower() + w[8:]                          # phospho residue in lowercase
        kin = {"akt": "AKT1", "cdk": "CDK1"}.get(r["kind"])
        ranked = ([kin] if kin else []) + list(rng.choice([f"RAND{k}" for k in range(8)], 4,
                                                          replace=False))
        for rank, k in enumerate(ranked, start=1):
            rows.append({"Sequence": w, "kinase": k, "percentile": 99 - rank, "rank": rank})
    pd.DataFrame(rows).to_csv(tmp_path / "kinase_library.tsv", sep="\t", index=False)
    return sites


def test_fasta_and_windows(tmp_path):
    (tmp_path / "f.fasta").write_text(">sp|Q1|X_HUMAN desc GN=GENEX\nMKRSPLLSY\n")
    f = kn.read_fasta(str(tmp_path / "f.fasta"))
    assert f["Q1"] == f["GENEX"] == "MKRSPLLSY"
    assert kn.window("MKRSPLLSY", 4, flank=3) == "MKRSPLL"
    assert kn.window("MKRSPLLSY", 1, flank=3) == "___MKRS"
    s = kn.site_windows(kn.site_positions(["GENEX_S4", "GENEX_T5"], ["GENEX"] * 2), f)
    assert s["residue_matches"].tolist() == [True, False]


def test_ksea_detects_shifted_set(rng):
    est = pd.Series(rng.normal(0, 1, 500), index=[f"s{i}" for i in range(500)])
    est.iloc[:20] += 2
    t = kn.ksea(est, {"UP": set(est.index[:20]), "RND": set(est.index[100:130])}).set_index("kinase")
    assert t.loc["UP", "padj"] < 1e-4 and t.loc["UP", "direction"] == "Up"
    assert t.loc["RND", "padj"] > 0.05


def test_motifs():
    w = "AAAARAASPAAAAAA"                                       # R at -3 and P at +1 of S
    m = kn.motif_table(pd.Series({"x": w})).loc["x"]
    assert m["basophilic (R/K at -3)"] and m["proline-directed (S/T-P)"]
    assert not m["tyrosine"]


@pytest.mark.slow
def test_phospho_kinase_pipeline(tmp_path):
    from proteomix.cli import main
    _phospho_files(tmp_path)
    out = tmp_path / "out"
    main([str(tmp_path / "sites.tsv"), "-m", str(tmp_path / "meta.tsv"), "--protein-data",
          str(tmp_path / "proteins.tsv"), "-o", str(out), "-t", "phospho", "--no-figures",
          "--no-progress", "--no-qc", "--fasta", str(tmp_path / "prot.fasta"),
          "--kinase-predictions", str(tmp_path / "kinase_library.tsv"), "--kinase-top", "5",
          "--timecourse", "--baseline", "Control"])
    k = out / "kinases"
    for method in ("ksea", "camera"):
        t20 = pd.read_csv(k / f"Stim_20_min_vs_Control_{method}.csv").set_index("kinase")
        assert t20.loc["AKT1", "padj"] < 0.05 and t20.loc["AKT1", "direction"] == "Up"
        t60 = pd.read_csv(k / f"Stim_60_min_vs_Control_{method}.csv").set_index("kinase")
        assert t60.loc["CDK1", "padj"] < 0.05 and t60.loc["CDK1", "direction"] == "Down"
        assert (t20.drop(index="AKT1", errors="ignore")["padj"] > 0.05).mean() > 0.8
    ce = pd.read_csv(out / "timecourse_sites" / "cluster_enrichment.csv")
    top = ce[ce["padj"] < 0.05]
    assert {"AKT1", "basophilic (R/K at -3)"} <= set(top["feature"])
    assert {"CDK1", "proline-directed (S/T-P)"} <= set(top["feature"])
