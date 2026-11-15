"""
proteomix showcase on the PD1 / TCR IP data.

Runs each part of the package into its own folder under OUT, printing what each demo
shows, and ends with a side-by-side comparison of the settings demos.

    python showcase_pd1.py                    # everything (fast figures)
    python showcase_pd1.py --only 1 2 7       # selected demos
    python showcase_pd1.py --final            # publication figures (PDF + 600 dpi PNG)

Demos
  1  Default analysis: QC, outliers, figures, IP-vs-IgG and PDL1 contrasts, bait check
  2  Changing settings: normalisation, imputation, filters, moderation (same data)
  3  Pathway enrichment and protein interactions (needs the GMT / STRING files below)
  4  Time course: clusters of PDL1 response profiles over 5 min / 20 min / 4 h
  5  Planning: samples per group needed for a 2-fold change, calibrated on this data
  6  Interpretation: how often hits are real with our settings, in data like ours
  7  Machine learning: can the proteome tell IP from IgG (positive control) and
     PDL1 from Ctrl at 20 min (hard question, 4 vs 4 samples)?
  8  Export of a complete matrix for downstream analyses
"""
import argparse
import os
import subprocess
import sys
import time

import pandas as pd

# ---- paths (edit if needed) -----------------------------------------------------------
DATA = r"C:/Users/Luis Iglesias/Documents/Luis/Papers/Martina/PD1_TCR_IP_fragpipe.csv"
META = r"C:/Users/Luis Iglesias/Documents/Luis/Papers/Martina/Metadata_PD1_TCR_IP.csv"
OUT = r"C:/Users/Luis Iglesias/Documents/Luis/Papers/Martina/proteomix_showcase"
BAIT = "PDCD1"                       # gene symbol of the bait (PD1); change if needed
# Optional downloads for demo 3 (leave as they are to skip that demo):
GENE_SETS = r"C:/path/to/ReactomePathways.gmt"            # Reactome / MSigDB, gene symbols
STRING_LINKS = r"C:/path/to/9606.protein.links.v12.0.txt.gz"
STRING_INFO = r"C:/path/to/9606.protein.info.v12.0.txt.gz"


def run(title, explanation, cmd, folder=None):
    print("\n" + "=" * 78)
    print(title)
    print("-" * 78)
    print(explanation)
    print("\n> " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
    t0 = time.time()
    res = subprocess.run(cmd)
    status = "OK" if res.returncode == 0 else f"FAILED (exit {res.returncode})"
    print(f"[{status} in {time.time() - t0:.0f} s]" + (f"  -> {folder}" if folder else ""))
    return res.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", type=int, default=list(range(1, 9)))
    ap.add_argument("--final", action="store_true", help="publication-quality figures")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--metadata", default=META)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--bait", default=BAIT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig = [] if a.final else ["--formats", "png", "--dpi", "150"]
    base = [a.data, "-m", a.metadata]
    d = lambda name: os.path.join(a.out, name)  # noqa: E731
    # run the tools with this Python (works whether or not the commands are on PATH)
    proteomix = [sys.executable, "-m", "proteomix.cli"]
    plan = [sys.executable, "-m", "proteomix.plan_cli"]
    ml = [sys.executable, "-m", "proteomix.ml.cli"]

    if 1 in a.only:
        run("1. Default analysis",
            "One line with defaults: QC (identifications, contaminants, CVs, replicate\n"
            "agreement, sample correlation, run order, confounding, outliers), bait recovery,\n"
            "16 IP/IgG contrasts with the selection model and variance moderation, PCAs\n"
            "coloured by pulldown / treatment / time, volcano plots and heatmaps.\n"
            "Look at: summary.txt (design check, QC notes, hits), qc/, figures/.",
            proteomix + base + ["-o", d("1_default"), "-t", "ip", "--bait", a.bait,
                                "--pca-shape", "pulldown"] + fig, d("1_default"))

    if 2 in a.only:
        variants = {
            "2a_median_ratio": (["--normalisation", "median_ratio"],
                                "median-of-ratios normalisation instead of none"),
            "2b_min_prob": (["--missing", "min_prob"],
                            "MinProb imputation + ordinary test instead of the selection model"),
            "2c_strict_filter": (["--min-valid-per-group", "3"],
                                 "keep only proteins with >= 3 values in at least one group"),
            "2d_no_moderation": (["--no-moderation"], "no empirical-Bayes variance moderation"),
        }
        for name, (flags, what) in variants.items():
            run(f"2. Changing settings: {name}",
                f"Same data, one change: {what}. Compared with demo 1 at the end.",
                proteomix + base + ["-o", d(name), "-t", "ip", "--no-figures", "--no-qc"] + flags,
                d(name))

    if 3 in a.only:
        have_sets = os.path.exists(GENE_SETS)
        have_ppi = os.path.exists(STRING_LINKS)
        if not (have_sets or have_ppi):
            print("\n3. Enrichment / PPI skipped: set GENE_SETS / STRING_LINKS at the top of "
                  "this script to downloaded files.")
        else:
            extra = (["--gene-sets", GENE_SETS] if have_sets else []) + \
                (["--ppi", STRING_LINKS, "--ppi-info", STRING_INFO] if have_ppi else [])
            run("3. Pathways and interactions",
                "CAMERA-style competitive gene-set test (accounts for correlation within\n"
                "pathways) and ORA of hits; interaction network among hits with a degree-\n"
                "matched enrichment test and modules. Look at: enrichment/, ppi/, figures/.",
                proteomix + base + ["-o", d("3_enrichment"), "-t", "ip"] + extra + fig,
                d("3_enrichment"))

    if 4 in a.only:
        run("4. Time course",
            "Proteins responsive in the IP time / treatment contrasts are clustered by\n"
            "their profile over unstimulated -> 5 min -> 20 min -> 4 h, for Ctrl and PDL1\n"
            "side by side (model-based means, so missing values do not bias early points).\n"
            "Look at: timecourse/clusters.csv, figures/timecourse_clusters.",
            proteomix + base + ["-o", d("4_timecourse"), "-t", "ip", "--timecourse",
                                "--no-qc"] + fig, d("4_timecourse"))

    if 5 in a.only:
        run("5. Planning: how many samples?",
            "Simulations calibrated on THIS dataset (variance, missingness, loading):\n"
            "samples per group needed to detect a 2-fold change with 80% power at 5% FDR.\n"
            "The effect size is our choice, never taken from the observed results.",
            plan + ["power", "-o", d("5_power"), "--data", a.data, "-m", a.metadata,
                    "--fold-change", "2", "--target-power", "0.8", "--normalisation", "none",
                    "--n-range", "2,3,4,6,8,10"] + fig, d("5_power"))

    if 6 in a.only:
        run("6. Interpretation: how much can we trust our hits?",
            "Our settings applied to simulated data of our design (4 per group, 15% of\n"
            "proteins changed by up to 4-fold): share of hits that are real, by adjusted\n"
            "p-value and abundance, and how often a non-hit hides a real change.",
            plan + ["interpret", "-o", d("6_interpret"), "--data", a.data, "-m", a.metadata,
                    "--normalisation", "none", "--prop-de", "0.15", "--effect-dist", "uniform",
                    "--effect", "2"] + fig, d("6_interpret"))

    if 7 in a.only:
        run("7a. Machine learning: IP vs IgG (positive control)",
            "Lasso, elastic net, kNN and MLP with nested cross-validation (preprocessing\n"
            "and tuning inside each fold). This contrast is easy, so all models should do well.",
            ml + ["train"] + base + ["-o", d("7a_ml_pulldown"), "--task", "classification",
                                     "--target", "pulldown", "--models",
                                     "lasso,elasticnet,knn,mlp"] + fig, d("7a_ml_pulldown"))
        run("7b. Machine learning: PDL1 vs Ctrl at 20 min in IP (hard question)",
            "Only 4 vs 4 samples: an honest estimate will be uncertain. The selection-\n"
            "frequency plot shows which proteins the lasso keeps choosing.",
            ml + ["train"] + base + ["-o", d("7b_ml_pdl1_20min"), "--task", "classification",
                                     "--target", "condition", "--classes",
                                     "IP TCS PDL1 20 min,IP TCS Ctrl 20 min", "--models",
                                     "lasso,knn", "--folds", "4", "--inner-folds", "3",
                                     "--repeats", "5"] + fig, d("7b_ml_pdl1_20min"))

    if 8 in a.only:
        run("8. Export for downstream analyses",
            "Normalised matrix with missing values filled by the selection model (expected\n"
            "intensity given non-detection), without using group labels, plus a mask of\n"
            "imputed values. Look at: export/.",
            proteomix + base + ["-o", d("8_export"), "-t", "ip", "--no-figures", "--no-qc",
                                "--export", "model"], d("8_export"))

    # ---- side-by-side comparison of the settings demos ------------------------------
    cols = {}
    for name in ("1_default", "2a_median_ratio", "2b_min_prob", "2c_strict_filter",
                 "2d_no_moderation"):
        f = os.path.join(a.out, name, "hits_per_contrast.csv")
        if os.path.exists(f):
            cols[name] = pd.read_csv(f).set_index("contrast")["hits"]
    if len(cols) > 1:
        comp = pd.DataFrame(cols)
        comp.to_csv(os.path.join(a.out, "settings_comparison.csv"))
        print("\n" + "=" * 78 + "\nHits per contrast under each setting (FDR 5%)\n" + "-" * 78)
        with pd.option_context("display.width", 200, "display.max_colwidth", 45):
            print(comp.to_string())
        print("\nSame data, different choices: differences show how much results depend on\n"
              "settings. Demo 6 shows which setting keeps the share of false hits near 5%.")


if __name__ == "__main__":
    main()
