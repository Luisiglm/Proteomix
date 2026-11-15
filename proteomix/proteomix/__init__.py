"""
proteomix: differential abundance for proteomics with abundance-dependent missingness.

Modules
    normalisation  median-of-ratios, total-intensity, quantile normalisation
    replicates     collapsing technical replicates
    missingness    per-batch missingness parameters (MissingParametersNR)
    imputation     zero, min_prob, mean, mean_noise, mean_group, kNN imputation
    model          selection-model / observed-only regression (DifferentialAbundanceNR)
    moderation     limma-style empirical-Bayes variance moderation
    testing        multiple-testing corrections
    pipeline       run_differential_abundance (end to end)
    simulation     simulated datasets
    benchmarking   type I / II error benchmarking study
    qc             sample / run outlier detection, exclusions
    qc_extended    identifications, contaminants, CVs, replicates, mean-variance, sample
                   correlation / mislabels, run order, confounding, IP and phospho checks
    enrichment     gene-set enrichment (CAMERA-style competitive test, ORA) from GMT files
    ppi            protein-protein interaction networks (STRING / edge lists), PPI enrichment
    planning       data-calibrated simulation: power / sample size, interpretation aid
                   (command line: proteomix-plan)
    timecourse     clustering of temporal profiles (k-means / fuzzy c-means)
    export         normalised / imputed matrices for downstream ML (model-based imputation)
    progress       progress bars;  runlog: detailed run logging
    ml             supervised learning: lasso / elastic net (incl. Cox), kNN, MLP, nested CV
    kinases        kinase activity inference (KSEA / CAMERA) from kinase predictions,
                   sequence windows from FASTA, motif and kinase enrichment of site clusters
    structure      phosphosites on user-provided structures / domains: lollipop plots,
                   distance heatmaps, PyMOL scripts (command line: proteomix-structure)
    cli            command-line pipeline (`proteomix data.tsv -m meta.tsv -o out -t ip`)
    plotting       benchmarking plots by method component
    fragpipe       reading FragPipe combined_protein files into pipeline inputs
    metadata       run -> condition files, condition factors, contrasts by group name
    ptm            paired protein-adjusted PTM (phosphosite) analysis and its benchmark
    figures        publication figures (PCA, QC, missingness curve, volcano, heatmap)
"""
from .figures import (make_report_figures, plot_enrichment, plot_heatmap, plot_protein_pca,
                      plot_intensity_distributions, plot_missingness_curve, plot_network,
                      plot_timecourse_clusters,
                      plot_sample_qc,
                      plot_missingness_per_sample, plot_pca, plot_total_intensity,
                      plot_volcano, publication_style, save_figure)
from .enrichment import camera, ora, read_gmt, run_enrichment
from .export import export_matrix, write_export
from .fragpipe import build_inputs, read_fragpipe
from .imputation import IMPUTATIONS, impute
from .metadata import ap_ms_contrasts, contrast_vector, read_run_metadata
from .missingness import MissingParametersNR
from .model import DifferentialAbundanceNR
from .moderation import fit_variance_prior, moderate_variances
from .normalisation import NORMALISATIONS, median_of_ratios, normalise
from .qc import run_quality_control, sample_qc, subset_inputs
from .ppi import ppi_analysis, read_network
from .ptm import (combine_independent, read_site_table, run_paired_ptm,
                  run_ptm_study)
from .pipeline import MISSING_HANDLING, run_differential_abundance
from .replicates import collapse_replicates
from .simulation import paired_ptm_inputs, simulate_dataset, simulate_paired_ptm
from .planning import SimulationSettings, calibrate, interpret, power_analysis
from .kinases import (kinase_activity, kinase_sets, ksea, read_fasta,
                      read_kinase_predictions)
from .structure import read_domains, read_structure, structure_report
from .testing import benjamini_hochberg
from .timecourse import build_profiles, cluster_profiles, time_design

__all__ = [
    "MissingParametersNR",
    "DifferentialAbundanceNR",
    "normalise",
    "median_of_ratios",
    "NORMALISATIONS",
    "collapse_replicates",
    "impute",
    "IMPUTATIONS",
    "MISSING_HANDLING",
    "fit_variance_prior",
    "moderate_variances",
    "benjamini_hochberg",
    "run_differential_abundance",
    "simulate_dataset",
    "plot_pca",
    "plot_missingness_per_sample",
    "plot_total_intensity",
    "plot_missingness_curve",
    "plot_volcano",
    "plot_heatmap",
    "make_report_figures",
    "publication_style",
    "save_figure",
    "read_fragpipe",
    "build_inputs",
    "read_run_metadata",
    "ap_ms_contrasts",
    "contrast_vector",
    "read_site_table",
    "run_paired_ptm",
    "combine_independent",
    "run_ptm_study",
    "simulate_paired_ptm",
    "paired_ptm_inputs",
    "run_quality_control",
    "sample_qc",
    "subset_inputs",
    "plot_sample_qc",
    "read_gmt",
    "run_enrichment",
    "camera",
    "ora",
    "read_network",
    "ppi_analysis",
    "export_matrix",
    "write_export",
    "plot_intensity_distributions",
    "plot_enrichment",
    "plot_network",
    "plot_protein_pca",
    "time_design",
    "build_profiles",
    "cluster_profiles",
    "plot_timecourse_clusters",
    "SimulationSettings",
    "calibrate",
    "power_analysis",
    "interpret",
    "read_structure",
    "read_domains",
    "structure_report",
    "read_fasta",
    "read_kinase_predictions",
    "kinase_sets",
    "ksea",
    "kinase_activity",
]

__version__ = "0.14.0"
