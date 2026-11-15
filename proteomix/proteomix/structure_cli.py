"""
Phosphosites of one protein in their structural context, from your own files.

  proteomix-structure results/tables/PD1_vs_Ctrl.csv --protein PDCD1 \\
        --structure AF-Q15116-F1-model_v4.pdb --domains PDCD1_domains.tsv -o pd1_structure

Inputs: a phospho results table from proteomix (or any table with site names like
GENE_S123 and estimate / padj columns), a structure (.pdb or .cif, e.g. an AlphaFold model)
and, optionally, a domain file (name / start / end table, UniProt TSV export or InterPro TSV).
Outputs: <protein>_sites.csv, lollipop plot, site-domain and site-site distance heatmaps,
and <protein>_view.pml: open it in PyMOL (File > Run Script) to rotate the structure and
save images (File > Export Image). Keep the .pml and the copied structure file together.
"""
import argparse
import sys

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(prog="proteomix-structure", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="results table (.csv / .tsv)")
    ap.add_argument("--protein", required=True, help="gene symbol or protein ID, e.g. PDCD1")
    ap.add_argument("--structure", required=True, help="structure file (.pdb or .cif)")
    ap.add_argument("--domains", default=None, help="domain file (optional)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--chain", default=None, help="chain (default: the first in the file)")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--estimate-col", default=None,
                    help="default: estimate_adjusted (phospho tables), else estimate")
    ap.add_argument("--padj-col", default=None)
    ap.add_argument("--formats", default="pdf,png")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--font-size", type=float, default=16)
    a = ap.parse_args(argv)
    from .structure import structure_report
    sep = "\t" if a.results.lower().endswith((".tsv", ".txt")) else ","
    try:
        res = pd.read_csv(a.results, sep=sep, index_col=0)
        out = structure_report(res, a.protein, a.structure, a.domains, a.out, a.chain, a.fdr,
                               a.estimate_col, a.padj_col,
                               tuple(f.strip() for f in a.formats.split(",") if f.strip()),
                               a.dpi, a.font_size)
    except (ValueError, KeyError, StopIteration, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    t = out["sites"]
    print(f"{a.protein}: {len(t)} site(s) mapped onto chain {out['chain']}")
    for m in out["messages"]:
        print(f"  note: {m}")
    if len(t):
        cols = [c for c in ("site", "estimate", "padj", "plddt", "in_domain",
                            "nearest_domain_3d", "distance_3d_angstrom") if c in t]
        with pd.option_context("display.width", 160, "display.max_colwidth", 30):
            print(t[cols].round(3).to_string(index=False))
    print(f"\nPyMOL: open {out['pymol']} (File > Run Script); scene F1 = domains, "
          f"F2 = confidence. Files in {a.out}")


if __name__ == "__main__":
    main()
