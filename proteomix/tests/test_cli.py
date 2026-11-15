"""End-to-end command-line runs on small FragPipe-style files."""
import json

import pandas as pd
import pytest

from proteomix.cli import main as proteomix_main
from proteomix.ml.cli import main as ml_main


@pytest.mark.slow
def test_proteomix_ip_run(fragpipe_files, tmp_path):
    out = tmp_path / "out"
    proteomix_main([str(fragpipe_files["protein"]), "-m", str(fragpipe_files["metadata"]),
                    "-o", str(out), "-t", "ip", "--formats", "png", "--dpi", "40",
                    "--no-progress", "--export", "model"])
    for f in ("summary.txt", "run.log", "settings.json", "all_contrasts.csv",
              "hits_per_contrast.csv", "runs.csv"):
        assert (out / f).exists(), f
    hits = pd.read_csv(out / "hits_per_contrast.csv").set_index("contrast")
    assert hits.loc["IP vs IgG: unstimulated", "hits"] >= 15     # 20 planted interactors
    assert (out / "figures" / "pca.png").exists()
    assert (out / "qc" / "protein_sample_qc.csv").exists()
    assert (out / "export" / "protein_matrix.csv").exists()
    settings = json.loads((out / "settings.json").read_text())
    assert settings["normalisation_used"] == "none"
    log = (out / "run.log").read_text()
    assert "sha256" in log and "Parameters" in log


@pytest.mark.slow
def test_proteomix_ml_train_and_predict(fragpipe_files, tmp_path):
    out = tmp_path / "ml"
    ml_main(["train", str(fragpipe_files["protein"]), "-m", str(fragpipe_files["metadata"]),
             "-o", str(out), "--task", "classification", "--target", "pulldown",
             "--models", "lasso,knn", "--folds", "3", "--inner-folds", "3",
             "--formats", "png", "--dpi", "40", "--no-progress"])
    summ = pd.read_csv(out / "cv_summary.csv", index_col=0)
    assert summ.loc["lasso", "auc_pooled"] > 0.9
    pred_path = tmp_path / "pred.csv"
    ml_main(["predict", str(out / "models" / "lasso.pkl"), str(fragpipe_files["protein"]),
             "-m", str(fragpipe_files["metadata"]), "-o", str(pred_path)])
    pred = pd.read_csv(pred_path, index_col=0)
    assert len(pred) == 12 and "predicted" in pred


def test_cli_reports_bad_metadata(fragpipe_files, tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("no_such_run\tX\n")
    with pytest.raises(SystemExit):
        proteomix_main([str(fragpipe_files["protein"]), "-m", str(bad), "-o",
                        str(tmp_path / "o"), "-t", "proteome", "--no-figures", "--no-progress"])
