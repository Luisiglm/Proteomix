"""
Synthetic phosphoproteomics results for an AKT1 demo of proteomix-structure.

Scenario (synthetic): growth-factor stimulation (10 min) vs unstimulated cells, analysed with
the paired protein-adjusted phospho model. AKT1 is activated: the activation-loop site T308
(PDK1) and the hydrophobic-motif site S473 (mTORC2) rise strongly; the turn-motif site T450
is constitutively phosphorylated, so it barely changes; other sites change modestly.
Two other proteins are included to show that --protein selects one protein.

Writes, in the current folder:
  akt1_demo_results.csv   results table in the proteomix phospho format
  akt1_domains.tsv        domain file (name / start / end / type)

THE NUMBERS ARE SYNTHETIC. Site positions and domain boundaries are from memory of the
literature and UniProt P31749 (canonical isoform): check them against the UniProt entry
before presenting. proteomix-structure checks every site against the structure and reports
any position that is not S / T / Y there.

Then download the AlphaFold model of P31749 (AlphaFold database entry P31749, PDB format)
into the same folder and run:
  proteomix-structure akt1_demo_results.csv --protein AKT1 --structure AF-P31749-F1-model_v4.pdb --domains akt1_domains.tsv -o akt1_structure
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)

# site: (log2 FC, adjusted p-value, note for the presenter)
akt1 = {
    "AKT1_T308": (2.6, 2e-5, "activation loop, PDK1: canonical activation"),
    "AKT1_S473": (2.2, 8e-5, "hydrophobic motif, mTORC2: canonical activation"),
    "AKT1_T450": (0.1, 0.85, "turn motif: constitutive, unchanged"),
    "AKT1_S129": (0.6, 0.04, "PH-kinase linker (CK2): modest"),
    "AKT1_S124": (0.3, 0.40, "linker: likely low pLDDT -> 3D distances unreliable"),
    "AKT1_Y326": (0.9, 0.03, "kinase domain tyrosine: moderate"),
    "AKT1_S477": (1.1, 0.01, "C-terminal tail: moderate"),
    "AKT1_T479": (1.0, 0.02, "C-terminal tail: moderate"),
}
others = {"GSK3B_S9": (-1.8, 1e-4), "MTOR_S2448": (1.2, 0.002)}   # not shown with --protein

rows = []
for site, (lfc, padj, note) in akt1.items():
    rows.append({"site_name": site, "Gene": "AKT1", "protein_key": "P31749",
                 "estimate_adjusted": lfc, "padj_adjusted": padj,
                 "pvalue_adjusted": padj / 4,
                 "estimate_unadjusted": lfc + rng.normal(0, 0.15),
                 "protein_estimate": 0.05, "adjustment": "paired",
                 "demo_note": note})
for site, (lfc, padj) in others.items():
    gene = site.split("_")[0]
    rows.append({"site_name": site, "Gene": gene, "protein_key": gene,
                 "estimate_adjusted": lfc, "padj_adjusted": padj, "pvalue_adjusted": padj / 4,
                 "estimate_unadjusted": lfc, "protein_estimate": 0.0, "adjustment": "paired",
                 "demo_note": ""})
pd.DataFrame(rows).set_index("site_name").to_csv("akt1_demo_results.csv")

# Domain boundaries: approximate, CHECK against UniProt P31749 (Features > Domain / Region)
domains = pd.DataFrame([
    ("PH domain", 5, 108, "domain"),
    ("Protein kinase", 150, 408, "domain"),
    ("AGC-kinase C-terminal", 409, 480, "domain"),
], columns=["name", "start", "end", "type"])
domains.to_csv("akt1_domains.tsv", sep="\t", index=False)

print("Wrote akt1_demo_results.csv and akt1_domains.tsv")
print("Presenter notes:")
for site, (lfc, padj, note) in akt1.items():
    print(f"  {site:10s} log2FC {lfc:+.1f}  padj {padj:.0e}  {note}")
