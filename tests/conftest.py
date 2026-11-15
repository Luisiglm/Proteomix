"""Shared fixtures: small simulated datasets and FragPipe-style files."""
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import expit


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end or simulation tests (deselect with -m 'not slow')")


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def two_group_data():
    """16 samples (2 groups) x 300 proteins, 15% DE, abundance-dependent missingness."""
    rng = np.random.default_rng(7)
    S, F = 16, 300
    g = np.repeat([0, 1], S // 2)
    mu = rng.normal(21, 2, F)
    lfc = np.zeros(F)
    de = rng.random(F) < 0.15
    lfc[de] = rng.choice([-1.5, 1.5], de.sum())
    true = mu + g[:, None] * lfc + rng.normal(0, 0.3, (S, F))
    miss = rng.random((S, F)) < expit(-1.5 * (true - 19.5))
    Y = np.where(miss, np.nan, true)
    X = np.eye(2)[g]
    return {"Y": torch.tensor(Y), "X": torch.tensor(X),
            "batch": torch.zeros(S, dtype=torch.long), "true": true, "lfc": lfc, "de": de,
            "group": g}


def _run_name(row, col, tray, well, acq, layer="PD1"):
    return f"230801kw_Lab_117_{layer}_{row}{col}_{tray}_{well}_1_{acq}"


@pytest.fixture
def fragpipe_files(tmp_path):
    """
    FragPipe-like combined_protein.tsv + run metadata for an IP / IgG design:
    4 conditions x 3 biological samples x 2 injections; 150 proteins, the first 20 enriched
    in IP, contaminants and decoys included.
    """
    rng = np.random.default_rng(3)
    conds = ["IP unstimulated", "IP TCS PDL1 5 min", "IgG unstimulated ", "IgG TCS PDL1 5min"]
    runs, labels = [], []
    acq = 1000
    for c, cond in enumerate(conds, start=1):
        for r in "ABC":
            for w in (1, 2):
                runs.append(_run_name(r, c, "S1", f"A{w}", acq))
                labels.append(cond)
                acq += 1
    P = 150
    base = rng.normal(22, 1.5, P)
    sample_eff = {}
    cols = {"Protein": [("contam_" if i == 0 else "rev_" if i == 1 else "sp|") + f"P{i:05d}|X"
                        for i in range(P)],
            "Protein ID": [f"P{i:05d}" for i in range(P)],
            "Gene": [f"GENE{i}" for i in range(P)],
            "Protein Probability": np.ones(P)}
    data = {}
    for rn, cd in zip(runs, labels):
        s = rn.split("_")[4]
        sample_eff.setdefault(s, rng.normal(0, 0.2, P))
        y = base + sample_eff[s] + rng.normal(0, 0.1, P)
        if cd.startswith("IP"):
            y[:20] += 3
        miss = rng.random(P) < expit(-1.5 * (y - 19.5))
        data[f"{rn} MaxLFQ Intensity"] = np.where(miss, 0, 2 ** y)
        data[f"{rn} Intensity"] = np.where(miss, 0, 2 ** y)
        data[f"{rn} Spectral Count"] = np.where(miss, 0, 5)
    df = pd.concat([pd.DataFrame(cols), pd.DataFrame(data)], axis=1)
    prot = tmp_path / "combined_protein.tsv"
    df.to_csv(prot, sep="\t", index=False)
    meta = tmp_path / "metadata.tsv"
    with open(meta, "w") as fh:
        for rn, cd in zip(runs, labels):
            fh.write(f"{rn} MaxLFQ Intensity\t{cd}\n")
    return {"protein": prot, "metadata": meta, "runs": runs, "labels": labels}
