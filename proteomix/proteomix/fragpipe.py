"""
Reading FragPipe combined_protein.tsv / .csv output into pipeline inputs.

    data = read_fragpipe("combined_protein.tsv", quantity="MaxLFQ Intensity")
    inputs = build_inputs(data, metadata, key="sample", group_col="condition",
                          batch_col="batch")
    results, model, miss, sf = run_differential_abundance(**inputs.pipeline_kwargs(),
                                                          contrast=(1, 0))

Run names are parsed with a regular expression into a run table (data.runs). The default
pattern fits names such as
    230801kw_MKreileder_117_PD1_A10_S3_B5_1_16061
      -> sample "A10", tray "S3", well "B5", injection "1", acquisition 16061
so that the two runs of sample A10 (wells B5 and B6) become technical replicates. Pass your own
pattern if your naming differs; named groups become columns of data.runs.
"""
import logging
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

def _info(msg: str):
    """Message to the run log when logging is set up (CLI), otherwise to the console."""
    log = logging.getLogger("proteomix")
    if log.handlers:
        log.info(msg)
    else:
        print(msg)


DEFAULT_RUN_PATTERN = (r"(?P<sample>[A-Z]\d{1,2})_(?P<tray>S\d+)_(?P<well>[A-Z]\d{1,2})_"
                       r"(?P<injection>\d+)_(?P<acquisition>\d+)$")
QUANTITIES = ("MaxLFQ Intensity", "Intensity", "Spectral Count", "Unique Spectral Count",
              "Total Spectral Count", "Area", "Unshared Area")
SUMMARY_ROWS = {"total", "mean", "median", "stddev", "sum"}   # summary rows in some reports


@dataclass
class FragPipeData:
    """intensities: proteins x runs (raw scale, NaN = missing); proteins: annotation; runs: run table."""
    intensities: pd.DataFrame
    proteins: pd.DataFrame
    runs: pd.DataFrame
    quantity: str
    contaminants: pd.DataFrame | None = None      # contaminant proteins x runs (raw scale)


@dataclass
class ProteomixInputs:
    Y: torch.Tensor                    # (runs, proteins) log2, NaN = missing
    X: torch.Tensor                    # (runs, groups) one-hot
    batch_ids: torch.Tensor            # (runs,)
    replicate_ids: torch.Tensor        # (runs,)
    run_names: list
    group_names: list
    batch_names: list
    feature_names: list
    sample_names: list                 # biological samples, in replicate_ids order
    runs: pd.DataFrame                 # aligned run table incl. metadata
    proteins: pd.DataFrame             # protein annotation, rows match Y's columns
    extra: dict = field(default_factory=dict)

    def sample_table(self) -> pd.DataFrame:
        """One row per biological sample (its first run's metadata), in sample_names order."""
        codes = self.replicate_ids.numpy()
        first = [int(np.where(codes == k)[0][0]) for k in range(len(self.sample_names))]
        t = self.runs.iloc[first].reset_index(drop=True)
        t.insert(0, "sample_name", self.sample_names)
        return t

    def pipeline_kwargs(self) -> dict:
        """Arguments for run_differential_abundance (add contrast and options yourself)."""
        return dict(Y=self.Y, X=self.X, batch_ids=self.batch_ids,
                    replicate_ids=self.replicate_ids, feature_names=self.feature_names,
                    group_names=self.group_names, run_names=self.run_names,
                    sample_names=self.sample_names, log_scale=True)


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------
def _run_names(columns, quantity: str | None = None):
    """
    Run names from the per-run quantity columns "<run> <quantity>". Candidates that are
    another run's name plus a word (e.g. "S1 Unshared" next to "S1" when the quantity is
    "Area") belong to a different quantity and are dropped. Without a quantity: MaxLFQ
    Intensity, else Intensity, else Spectral Count.
    """
    if quantity:
        suffix = " " + quantity
        cand = [c[: -len(suffix)] for c in columns if c.endswith(suffix)]
        cand = [r for r in cand if r.strip() and r != "Combined"]
        runs = [r for r in cand if not any(r != r2 and r.startswith(r2 + " ") for r2 in cand)]
        if runs:
            return runs
    for suffix in (" MaxLFQ Intensity", " Intensity", " Spectral Count"):
        runs = [c[: -len(suffix)] for c in columns if c.endswith(suffix)]
        runs = [r for r in runs if not r.endswith((" MaxLFQ", " Unique", " Total"))
                and r != "Combined"]
        if runs:
            return runs
    raise ValueError("no per-run quantity columns found (e.g. '<run> MaxLFQ Intensity')")


def parse_run_names(runs, pattern: str = DEFAULT_RUN_PATTERN) -> pd.DataFrame:
    """
    Run table from run names. Named groups of `pattern` become columns; runs that do not match
    keep NaN there and get their own run name as 'sample' (i.e. no replicate collapsing).
    """
    rx = re.compile(pattern)
    rows = []
    for r in runs:
        m = rx.search(r)
        rows.append({"run": r, **(m.groupdict() if m else {})})
    table = pd.DataFrame(rows)
    if "sample" not in table:
        table["sample"] = table["run"]
    unmatched = table["sample"].isna()
    if unmatched.any():
        if unmatched.all():
            _info(f"Run names do not follow the lab naming pattern (e.g. "
                  f"{table['run'].iloc[0]!r}): each run is treated as its own biological "
                  f"sample (no technical replicates). Use a sample-ID metadata column if "
                  f"there are replicate injections.")
        else:
            warnings.warn(f"{unmatched.sum()} run names did not match the pattern; each is "
                          f"treated as its own sample: {table.loc[unmatched, 'run'].tolist()[:3]}")
        table.loc[unmatched, "sample"] = table.loc[unmatched, "run"]
    if "acquisition" in table and table["acquisition"].notna().all():
        table["acquisition"] = table["acquisition"].astype(int)
    return table


def read_fragpipe(path: str, quantity: str = "MaxLFQ Intensity",
                  run_pattern: str = DEFAULT_RUN_PATTERN, remove_contaminants: bool = True,
                  remove_decoys: bool = True, min_probability: float | None = None,
                  min_observed: int = 1, order_by_acquisition: bool = True) -> FragPipeData:
    """
    Reads a FragPipe combined_protein file (tab- or comma-separated, detected automatically),
    or any similar wide table with one row per protein / gene and one "<run> <quantity>"
    column per run, e.g. gene-level precursor-area summaries with "<sample> Area" and
    "<sample> Unshared Area" columns. Without a "Protein" / "Protein ID" / "Gene" column, the
    first column is taken as the gene identifier; summary rows (Total, Mean, Median, ...)
    are dropped.

    quantity: one of QUANTITIES (or any column suffix). MaxLFQ Intensity is recommended for
        FragPipe tables; for area tables "Unshared Area" (peptides unique to the gene) avoids
        counting shared peptides for several genes, "Area" is more complete. Spectral counts
        are discrete and fit the log-normal model poorly.
    Zeros are treated as missing. Proteins whose 'Protein' starts with 'contam_' (contaminants)
    or 'rev_' (decoys) are removed by default; min_probability filters on 'Protein Probability';
    min_observed keeps proteins quantified in at least that many runs.
    """
    with open(path, encoding="utf-8-sig") as fh:
        header = fh.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)

    if not {"Protein", "Protein ID", "Gene"} & set(df.columns):
        first = df.columns[0]
        df = df.rename(columns={first: "Gene"})
        df["Gene"] = df["Gene"].astype(str).str.strip()
        df = df[~df["Gene"].str.lower().isin(SUMMARY_ROWS)]
        df["Protein ID"] = df["Gene"]
    runs = _run_names(df.columns, quantity)
    cols = [f"{r} {quantity}" for r in runs]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"quantity {quantity!r} not found for all runs, e.g. {missing[0]!r}")

    keep = pd.Series(True, index=df.index)
    prot = df["Protein"].astype(str) if "Protein" in df else pd.Series("", index=df.index)
    if remove_contaminants:
        keep &= ~prot.str.startswith("contam_")
    if remove_decoys:
        keep &= ~prot.str.startswith("rev_")
    if min_probability is not None and "Protein Probability" in df:
        keep &= df["Protein Probability"] >= min_probability

    # contaminant intensities are kept aside for QC before they are removed
    contam = None
    cmask = prot.str.startswith("contam_")
    if cmask.any():
        contam = df.loc[cmask, cols].apply(pd.to_numeric, errors="coerce")
        contam = contam.where(contam > 0)
        contam.columns = runs
        cname = df.loc[cmask, "Gene"] if "Gene" in df else prot[cmask]
        contam.index = [str(g) if pd.notna(g) and str(g).strip() else str(p_)
                        for g, p_ in zip(cname, prot[cmask])]

    mat = df.loc[keep, cols].apply(pd.to_numeric, errors="coerce")
    mat = mat.where(mat > 0)
    obs = mat.notna().sum(1)
    mat = mat[obs >= min_observed]

    ann_cols = [c for c in ["Protein", "Protein ID", "Entry Name", "Gene", "Protein Length",
                            "Organism", "Protein Existence", "Description",
                            "Protein Probability", "Combined Total Peptides"] if c in df]
    proteins = df.loc[mat.index, ann_cols].copy()
    proteins["n_observed"] = mat.notna().sum(1)
    proteins["feature_name"] = _feature_names(proteins)
    mat.columns = runs
    mat.index = proteins["feature_name"].to_numpy()
    proteins.index = mat.index

    table = parse_run_names(runs, run_pattern)
    if order_by_acquisition and "acquisition" in table and table["acquisition"].notna().all():
        table = table.sort_values("acquisition").reset_index(drop=True)
        mat = mat[table["run"].tolist()]
    table["missing_pct"] = mat.isna().mean(0).to_numpy() * 100

    n_removed = int((~keep).sum())
    _info(f"Read {len(runs)} runs ({table['sample'].nunique()} samples), "
          f"{mat.shape[0]} proteins ({n_removed} contaminants/decoys/low-probability removed, "
          f"{int(keep.sum()) - mat.shape[0]} below min_observed); "
          f"{mat.isna().to_numpy().mean():.1%} missing.")
    return FragPipeData(intensities=mat, proteins=proteins, runs=table, quantity=quantity,
                        contaminants=contam)


def _feature_names(proteins: pd.DataFrame) -> pd.Series:
    """Gene symbol where available, else protein ID; duplicates get the protein ID appended."""
    pid = proteins.get("Protein ID", proteins.get("Protein", pd.Series(
        [f"protein_{i}" for i in range(len(proteins))], index=proteins.index))).astype(str)
    gene = proteins.get("Gene", pd.Series(np.nan, index=proteins.index))
    name = gene.where(gene.notna() & (gene.astype(str).str.strip() != ""), pid).astype(str)
    dup = name.duplicated(keep=False)
    name[dup] = name[dup] + "_" + pid[dup]
    return name


# --------------------------------------------------------------------------------------
# Combining with metadata
# --------------------------------------------------------------------------------------
def build_inputs(data: FragPipeData, metadata: pd.DataFrame | None = None, key: str = "sample",
                 metadata_key: str | None = None, group_col: str | None = None,
                 group_order=None, batch_col: str | None = None,
                 replicate_col: str = "sample", drop_unmatched: bool = False) -> ProteomixInputs:
    """
    Aligns the run table with sample metadata and builds the pipeline inputs.

    metadata: one row per sample (or per run), with a column matching data.runs[key]
        (metadata_key, default the same name). Columns are added to the run table.
    group_col: condition column -> one-hot X, levels in group_order (default: order of first
        appearance; the first level is a natural reference). Without group_col, X is a single
        intercept column (no group comparison possible).
    batch_col: column (from metadata or the parsed run table, e.g. "tray") -> batch_ids.
        None puts every run in one batch.
    replicate_col: column identifying biological samples; runs sharing a value are technical
        replicates and are collapsed by the pipeline.
    drop_unmatched: drop runs without metadata instead of raising an error.
    """
    runs = data.runs.copy()
    if metadata is not None:
        mk = metadata_key or key
        meta = metadata.copy()
        meta[mk] = meta[mk].astype(str)
        runs[key] = runs[key].astype(str)
        if meta[mk].duplicated().any() and key != "run":
            raise ValueError(f"metadata column {mk!r} has duplicated values")
        overlap = [c for c in meta.columns if c in runs.columns and c != mk]
        runs = runs.merge(meta.drop(columns=overlap).rename(columns={mk: key}), on=key,
                          how="left", indicator=True)
        unmatched = runs["_merge"] == "left_only"
        if unmatched.any():
            msg = (f"{unmatched.sum()} runs have no metadata (e.g. "
                   f"{runs.loc[unmatched, key].iloc[0]!r}).")
            if not drop_unmatched:
                raise ValueError(msg + " Fix the metadata or pass drop_unmatched=True.")
            warnings.warn(msg + " They are dropped.")
            runs = runs[~unmatched]
        runs = runs.drop(columns="_merge").reset_index(drop=True)

    mat = data.intensities[runs["run"].tolist()]
    Y = torch.tensor(np.log2(mat.to_numpy(dtype=float).T))

    if group_col is not None:
        if runs[group_col].isna().any():
            raise ValueError(f"missing values in group column {group_col!r}")
        levels = list(group_order) if group_order is not None else list(
            dict.fromkeys(runs[group_col].astype(str)))
        g = runs[group_col].astype(str)
        unknown = set(g) - set(levels)
        if unknown:
            raise ValueError(f"groups {unknown} are not in group_order")
        X = torch.tensor(np.stack([(g == lev).to_numpy(float) for lev in levels], 1))
        group_names = levels
    else:
        X = torch.ones(len(runs), 1, dtype=torch.float64)
        group_names = ["all"]

    if batch_col is not None:
        b_codes, b_levels = pd.factorize(runs[batch_col].astype(str), sort=True)
        batch_names = list(b_levels)
    else:
        b_codes, batch_names = np.zeros(len(runs), dtype=int), ["all"]
    rep_codes, rep_levels = pd.factorize(runs[replicate_col].astype(str))

    _check_replicates(runs, replicate_col, group_col, batch_col)
    if group_col is not None:
        counts = runs.groupby(group_col)[replicate_col].nunique().reindex(group_names)
        _info("Biological samples per group: " + ", ".join(f"{k}: {v}" for k, v in
                                                           counts.items()))

    return ProteomixInputs(
        Y=Y, X=X, batch_ids=torch.tensor(b_codes, dtype=torch.long),
        replicate_ids=torch.tensor(rep_codes, dtype=torch.long),
        run_names=runs["run"].tolist(), group_names=group_names, batch_names=batch_names,
        feature_names=list(mat.index), sample_names=list(rep_levels), runs=runs,
        proteins=data.proteins.loc[mat.index],
        extra={"contaminants": None if data.contaminants is None
               else data.contaminants[runs["run"].tolist()]})


def _check_replicates(runs, replicate_col, group_col, batch_col):
    for col in [c for c in (group_col, batch_col) if c is not None]:
        bad = runs.groupby(replicate_col)[col].nunique() > 1
        if bad.any():
            kind = "group" if col == group_col else "batch"
            msg = (f"replicates of {bad.sum()} samples span several {kind}s "
                   f"(e.g. {bad[bad].index[0]!r}).")
            if col == group_col:
                raise ValueError(msg + " Check the metadata.")
            warnings.warn(msg + " The pipeline uses each sample's first run's batch.")


def short_run_names(inputs: ProteomixInputs, fields=("sample", "well")) -> list:
    """Compact labels for figures, e.g. 'A10_B5', from columns of the run table."""
    cols = [f for f in fields if f in inputs.runs]
    if not cols:
        return inputs.run_names
    return inputs.runs[cols].astype(str).agg("_".join, axis=1).tolist()
