"""
PD1 IP / IgG pull-down experiment: FragPipe output + run -> condition file -> all standard
contrasts, result tables and figures.

Design (from the metadata): 2 pulldowns (IP, IgG) x 7 conditions (unstimulated; TCS Ctrl and
TCS PDL1 at 5 min, 20 min, 4 h) = 14 groups, 4 biological samples each (rows A-D of the plate
layout), each injected twice (technical replicates, collapsed before testing).

Run:  python analyse_pd1_apms.py
"""
import os

from proteomix import run_differential_abundance
from proteomix.fragpipe import build_inputs, read_fragpipe
from proteomix.metadata import ap_ms_contrasts, condition_order, read_run_metadata

PROTEIN_FILE = "combined_protein.tsv"
METADATA_FILE = "metadata.tsv"            # two columns: column name, condition
OUT_DIR = "results_pd1"

data = read_fragpipe(PROTEIN_FILE, quantity="MaxLFQ Intensity", min_observed=2)
meta = read_run_metadata(METADATA_FILE)
print(meta.drop_duplicates("condition")[["condition", "pulldown", "treatment", "time_min"]]
      .to_string(index=False))

inputs = build_inputs(data, meta, key="run", group_col="condition",
                      group_order=condition_order(meta))
contrasts = ap_ms_contrasts(meta)        # edit / extend this dict as needed
for label, spec in contrasts.items():
    print(f"  {label}: {spec}")

results, model, miss, log_sf = run_differential_abundance(
    **inputs.pipeline_kwargs(), contrast=contrasts,
    normalisation="none",                # MaxLFQ is already normalised; IP vs IgG is very
                                         # asymmetric, which biases median-based normalisation
    missing="selection", moderate=True,
    figures_dir=os.path.join(OUT_DIR, "figures"))

os.makedirs(OUT_DIR, exist_ok=True)
annotation = inputs.proteins[[c for c in ["Protein ID", "Gene", "Description"]
                              if c in inputs.proteins]]
summary = []
for label, res in results.items():
    slug = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")
    annotation.join(res).sort_values("pvalue").to_csv(os.path.join(OUT_DIR, f"{slug}.csv"))
    sig = res["padj"] < 0.05
    summary.append(f"{label:45s} {sig.sum():5d} at FDR 5%  "
                   f"(up {(sig & (res['estimate'] > 0)).sum()}, "
                   f"down {(sig & (res['estimate'] < 0)).sum()})")
print("\n".join(summary))
