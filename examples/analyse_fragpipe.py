"""
Example: FragPipe combined_protein.tsv + sample metadata -> differential abundance + figures.

Edit the settings below, then run:  python analyse_fragpipe.py
The metadata file needs one row per biological sample, with a column matching the sample ID
parsed from the run names (e.g. "A10" from ..._PD1_A10_S3_B5_1_16061), a condition column and,
optionally, a batch column.
"""
import pandas as pd

from proteomix import run_differential_abundance
from proteomix.fragpipe import build_inputs, read_fragpipe, short_run_names

# ---- settings -------------------------------------------------------------------------
PROTEIN_FILE = "combined_protein.tsv"
METADATA_FILE = "metadata.csv"
METADATA_SAMPLE_COLUMN = "sample_id"      # column in the metadata holding "A10", "B3", ...
GROUP_COLUMN = "condition"
GROUP_ORDER = ["Control", "PD1"]          # first level = reference
BATCH_COLUMN = None                       # e.g. "batch" (metadata) or "tray" (from run names)
CONTRASTS = {"PD1 vs Control": (1, 0)}    # indices into GROUP_ORDER
OUT_DIR = "results"
# ----------------------------------------------------------------------------------------

data = read_fragpipe(PROTEIN_FILE, quantity="MaxLFQ Intensity", min_observed=2)
print(data.runs[["run", "sample", "tray", "well", "acquisition"]].head())

metadata = pd.read_csv(METADATA_FILE)
inputs = build_inputs(data, metadata, key="sample", metadata_key=METADATA_SAMPLE_COLUMN,
                      group_col=GROUP_COLUMN, group_order=GROUP_ORDER, batch_col=BATCH_COLUMN)

kwargs = inputs.pipeline_kwargs()
kwargs["run_names"] = short_run_names(inputs)             # "A10_B5" instead of the full name
results, model, miss, log_sf = run_differential_abundance(
    **kwargs, contrast=CONTRASTS, missing="selection", moderate=True,
    figures_dir=f"{OUT_DIR}/figures")

for label, res in results.items():
    table = inputs.proteins[["Protein ID", "Gene", "Description"]].join(res)
    path = f"{OUT_DIR}/{label.replace(' ', '_')}.csv"
    table.sort_values("pvalue").to_csv(path)
    print(f"{label}: {(res['padj'] < 0.05).sum()} proteins at FDR 5% -> {path}")
