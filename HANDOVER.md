# proteomix 0.14.0: developer handover

Differential abundance for label-free proteomics with abundance-dependent missingness,
plus QC, PTM, enrichment, time course and supervised-learning modules. Written with AI
assistance (Claude) and validated as described below; **it has not yet had a line-by-line
human review**. This note lists what exists, what was checked and how, and what is open.

## Layout

| Module | Purpose |
|---|---|
| `missingness.py` | Per-batch logistic missingness curve, IRLS in concave parameterisation |
| `model.py` | Selection-model regression (sigmoid approx. of probit marginal), Newton–Raphson, optional per-sample offsets (PTM) |
| `moderation.py` | limma-style empirical Bayes (fitFDist, trend via B-spline, simplified robust) |
| `normalisation.py`, `replicates.py`, `imputation.py` | median-of-ratios / total / quantile; replicate collapsing; 6 imputations |
| `pipeline.py` | `run_differential_abundance` end to end |
| `qc.py` | Sample / run outlier scores (robust z, leave-one-out), exclusions |
| `qc_extended.py`, `qc_figures.py` | Identifications, contaminants, CVs, replicate pairs, mean–variance, sample correlation and mislabels, run-order drift, confounding; IP bait recovery and IgG background; phospho residues, localisation, enrichment efficiency |
| `ptm.py` | Paired protein-offset PTM analysis, Welch–Satterthwaite alternative, PTM benchmark |
| `enrichment.py`, `ppi.py` | CAMERA-style competitive test, ORA; STRING / edge-list PPI, degree-matched null |
| `timecourse.py` | Temporal profiles from model estimates; k-means / fuzzy c-means; silhouette k |
| `export.py` | Complete matrices; model-based conditional-mean / draw imputation |
| `fragpipe.py`, `metadata.py` | FragPipe readers (also wide gene-level tables with `<sample> Area` / `Unshared Area` columns), run metadata, contrasts by group name |
| `figures.py`, `plotting.py` | Publication figures; benchmark plots |
| `planning.py`, `plan_cli.py` | Simulation calibrated to a dataset, per experiment type (proteome: `simulate_dataset`; IP: `simulate_ip_dataset`; phospho: `simulate_paired_ptm` + paired analysis); power / sample size for a chosen effect and contrast; interpretation of hits for given settings (`proteomix-plan`) |
| `structure.py`, `structure_cli.py` | Phosphosites on a user-provided structure (PDB / mmCIF) and domain file: residue checks, pLDDT, site–domain and site–site distances, lollipop plot, heatmaps, PyMOL script (`proteomix-structure`) |
| `kinases.py` | FASTA windows and residue checks; Kinase Library-style predictions (long / wide; matched by site name, gene + position or sequence window) to kinase sets; KSEA and CAMERA kinase activity; motif and kinase-set enrichment of site clusters (`proteomix -t phospho --kinase-predictions ... --fasta ... --timecourse`) |
| `benchmarking.py`, `simulation.py` | Simulation studies (type I / II error, FDR) |
| `cli.py`, `ml/cli.py` | `proteomix` and `proteomix-ml` commands |
| `ml/` | Coordinate-descent elastic net (gaussian / binomial / multinomial / Cox), kNN, MLP, fold-safe preprocessing, nested CV |
| `progress.py`, `runlog.py` | Progress bars; detailed run log with parameters, versions, input checksums |

## What was validated (and how)

These checks are automated in `tests/` (95 tests, ~30 s; `pytest -m "not slow"` for the
fast subset). Install with `pip install -e ".[test]"`; CI runs on Linux, Windows and macOS
via `.github/workflows/tests.yml`. Reference comparisons use scikit-learn and lifelines,
which are test-only dependencies.

- **Model derivatives**: gradient / Hessian of the missing-value term checked against
  finite differences (~1e-9); sigmoid approximation compared with exact integrals.
- **Missingness fit**: recovers simulated (w, r); log-likelihood monotone.
- **Calibration** (simulations, 10 runs/scenario): type I error 0.048–0.053 after the
  n/(n−p) small-sample correction (without it 0.077); KS distance of null p-values ~0.02.
- **Moderation**: recovers d0 ≈ 9.7, s0² ≈ 0.089 (true 10, 0.09); trend and robust tested.
- **Paired PTM**: paired model calibrated (FPR 0.049–0.057) with power 0.62–0.66;
  Welch–Satterthwaite becomes conservative as shared variance grows (power → 0).
- **Elastic net**: matches scikit-learn (gaussian ≤ 1e-4, binomial ≤ 2e-4) and lifelines
  Cox (≤ 1e-4 at small penalty). Cox "full" Hessian reaches lower objectives than the
  glmnet diagonal variant at small λ (both available: `cox_hessian=`).
- **Model-based export imputation**: bias +0.02, RMSE 0.42 log2 vs 0.88–1.46 for kNN /
  mean / min_prob (simulation with known truth).
- **QC**: flags injected degraded sample and single bad injection, no false flags in clean
  data after normalisation (one true loading shift flagged when unnormalised).
- **Time course**: four known profile shapes recovered with purity 1.0, k = 4 chosen by
  silhouette, both methods.
- **End-to-end CLI** runs on synthetic data mimicking the lab's FragPipe / metadata format
  for `--type ip`, `proteome`, `phospho`, and `proteomix-ml train` (all three tasks).

## Known limitations and open issues

1. **Test coverage is behavioural**, not line-by-line: plotting functions are exercised
   only through the end-to-end CLI tests, and statistical tolerances were chosen from the
   validation runs. Add coverage reporting (`pytest-cov`) to find untested paths.
2. **Not tested on real FragPipe output.** The site-table reader's column names
   (`Index`, localisation probability, `<run> Intensity`) are guesses; confirm with a real
   `combined_site_*.tsv`.
3. **Degrees of freedom** for proteins with missing values use n_obs − p (heuristic);
   calibration slightly above nominal in some scenarios (≈0.053).
4. **Robust moderation** is a simplified version of limma's (outliers get d0 = 0).
5. **PTM**: sites are adjusted only when the protein is measured in ≥ 2 samples per group;
   filled offsets are treated as known (uncertainty not propagated).
6. **Enrichment**: gene-level reduction by max |z| favours genes with many sites (PTM).
7. **ML**: new data in `predict` is normalised against its own reference; the pooled
   out-of-fold C-index mixes risk scales across fold models (use per-fold means); MLPs
   are rarely competitive at typical proteomics sample sizes.
8. **Time course**: group-mean dynamics only (independent cultures per time point);
   time is confounded with acquisition order / tray in the PD1 design.
9. **Extended QC thresholds** (robust z 3.5, minimum scales, 10% contaminant limit, drift
   p < 0.01) were set on synthetic data; the phospho enrichment check assumes FragPipe
   modified-peptide notation (S[167], T[181], Y[243] or 79.966) and needs a real file to
   confirm.
10. **Planning tools** simulate from the package's own model, so they show what the method
    can see in data *like* the user's; real data with outliers or strongly correlated
    proteins can do worse. Calibration matches the observed missingness by shifting the
    abundance distribution (sparse proteins are excluded from the direct estimate); this
    matching uses the proteome simulator for every type. Phospho calibration uses the
    protein file only; site-level settings (stoichiometry, enrichment noise, site
    missingness) keep their defaults unless set (--set).
11. **Structure module**: the mmCIF reader is minimal (the `_atom_site` loop, first model);
    distances use one atom per site (Ser OG / Thr OG1 / Tyr OH, else C-alpha); domain
    readers cover simple tables, UniProt feature-column exports and InterPro TSV (formats
    from documentation, to be confirmed on real exports); the PyMOL script has not yet been
    opened in PyMOL by us.
12. **Kinase activity** is competitive (relative to all quantified sites): when a large share
    of sites changes in one direction, unchanged kinases can appear changed in the opposite
    direction (seen in the synthetic test with 20% of sites up 4-fold). The Kinase Library
    export format was implemented from its description, not from a real export file.
13. **Performance**: pure Python coordinate descent; ML runs of ~1 min for 56 samples ×
   ~800 proteins. Profile before optimising (Numba is the obvious next step).

## Suggested next steps

1. Run the test suite in CI and add coverage reporting; extend tests as bugs are found.
2. Line-by-line review of the statistical core (`model.py`, `missingness.py`,
   `moderation.py`, `ptm.py`, `ml/lasso.py`, `ml/cv.py`).
3. Real-data benchmarks: spike-in (UPS1 / mixed-proteome designs) and real-null
   (replicate splits) against proDA, limma, MSstats, MSstatsPTM.
4. Documentation (installation, tutorial notebook, API reference), PyPI / Bioconda
   packaging (check the name is free), licence, Zenodo DOI.
5. Planned features: omnibus multi-contrast test (time × treatment interaction),
   fused-lasso trend filtering for time courses.
