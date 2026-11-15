# proteomix

Differential abundance analysis for label-free proteomics with abundance-dependent missing
values (selection model), plus QC, phospho (protein-adjusted PTM), enrichment, kinase
activity, time courses, study planning, structure views and machine learning.

**Status: development version (0.14.0), written with AI assistance and awaiting human code
review. See `HANDOVER.md` for what has been validated and what is open.**

## Install

```bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .            # from this folder; add ".[test]" for the test dependencies
```

## Commands

| Command | Purpose |
|---|---|
| `proteomix data.tsv -m metadata.tsv -o results -t ip\|proteome\|phospho` | full analysis: QC, testing, figures, enrichment, PPI, time course, kinases, export |
| `proteomix-plan power\|interpret -t ...` | samples needed for an effect size; how reliable hits are with given settings |
| `proteomix-ml train\|predict ...` | lasso / elastic net / kNN / MLP with nested cross-validation |
| `proteomix-structure results.csv --protein GENE --structure model.pdb` | phosphosites on a structure: lollipop, distance heatmaps, PyMOL script |

Each command has `--help`. Examples are in `examples/`.

## Tests

```bat
pytest                 # all tests (about 30 s)
pytest -m "not slow"   # quick subset
```
