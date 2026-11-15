"""Wide gene-level area tables (e.g. precursor-area summaries with Area / Unshared Area)."""
import numpy as np
import pandas as pd
import pytest

from proteomix.fragpipe import read_fragpipe


@pytest.fixture
def area_table(tmp_path):
    rng = np.random.default_rng(0)
    samples = [f"AG-A0{i:02d}-01A-22" for i in range(10)]
    genes = ["Total"] + [f"GENE{i}" for i in range(60)]          # a summary row first
    cols = {"": genes}
    for s in samples:
        area = 2 ** rng.normal(26, 2, len(genes))
        area[rng.random(len(genes)) < 0.2] = 0
        cols[f"{s} Area"] = area
        cols[f"{s} Unshared Area"] = np.where(rng.random(len(genes)) < 0.1, 0, area * 0.8)
    df = pd.DataFrame(cols)
    df["Description"], df["Organism"], df["Chromosome"], df["Locus"] = "d", "Homo sapiens", "1", "x"
    path = tmp_path / "area.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path, samples


@pytest.mark.parametrize("quantity", ["Area", "Unshared Area"])
def test_reads_area_tables(area_table, quantity):
    path, samples = area_table
    data = read_fragpipe(str(path), quantity=quantity)
    assert list(data.runs["run"]) == samples                     # "X Unshared" is not a run
    assert data.intensities.shape == (60, 10)                    # "Total" row dropped
    assert "GENE0" in data.intensities.index
    assert not (data.intensities == 0).any().any()
    assert (data.runs["sample"] == data.runs["run"]).all()       # each run its own sample


def test_area_table_through_pipeline(area_table, tmp_path):
    from proteomix.cli import main
    path, samples = area_table
    meta = tmp_path / "meta.tsv"
    meta.write_text("".join(f"{s}\t{'A' if i < 5 else 'B'}\n" for i, s in enumerate(samples)))
    out = tmp_path / "o"
    main([str(path), "-m", str(meta), "-o", str(out), "-t", "proteome", "--quantity",
          "Unshared Area", "--no-figures", "--no-progress"])
    res = pd.read_csv(out / "all_contrasts.csv", index_col=0)
    assert len(res) == 60
