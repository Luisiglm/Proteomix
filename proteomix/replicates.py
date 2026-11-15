"""Collapsing technical replicates into biological samples."""
import warnings

import torch


def collapse_replicates(y: torch.Tensor, x: torch.Tensor, batch_ids: torch.Tensor,
                        replicate_ids, method: str = "mean", min_present: int = 1):
    """
    Collapses technical replicates (several runs of one biological sample) into one row.

    y: (N_runs, F) normalised log2 intensities, NaN = missing.
    x: (N_runs, p) design; must be identical across replicates of a sample.
    batch_ids: (N_runs,); if replicates span batches, the first replicate's batch is used.
    replicate_ids: (N_runs,) biological-sample label of each run (any integers).
    method: "mean" or "median" of the observed replicate values.
    min_present: the collapsed value is NaN unless at least this many replicates are observed.

    Note: averaging only the observed replicates is biased upwards when missingness depends on
    abundance (the low replicates are the ones that go missing). Raising min_present (e.g. to
    the number of replicates) trades that bias for more missing values, which the selection
    model then handles.

    Returns (y_c, x_c, batch_c, sample_ids, n_present), with S = number of biological samples.
    """
    rep = torch.as_tensor(replicate_ids, device=y.device)
    sample_ids, inv = torch.unique(rep, sorted=True, return_inverse=True)
    N, S = y.shape[0], len(sample_ids)
    run_idx = torch.arange(N, device=y.device)

    R = torch.zeros(S, N, dtype=y.dtype, device=y.device)
    R[inv, run_idx] = 1.0
    obs = (~torch.isnan(y)).to(y.dtype)
    n_present = R @ obs                                          # (S, F)

    if method == "mean":
        y_c = (R @ torch.nan_to_num(y, nan=0.0)) / n_present.clamp(min=1.0)
    elif method == "median":
        y_c = torch.stack([torch.nanmedian(y[inv == s], dim=0).values for s in range(S)])
    else:
        raise ValueError("method must be 'mean' or 'median'")
    y_c = torch.where(n_present >= min_present, y_c, torch.full_like(y_c, torch.nan))

    first = torch.full((S,), N, dtype=torch.long, device=y.device)
    first = first.scatter_reduce(0, inv, run_idx, reduce="amin")
    x_c = x[first]
    if not torch.equal(x, x_c[inv]):
        raise ValueError("Replicates of the same sample have different design rows in X.")
    batch_c = batch_ids[first]
    if not torch.equal(batch_ids, batch_c[inv]):
        warnings.warn("Some replicates span batches; using the batch of each sample's first run.")
    return y_c, x_c, batch_c, sample_ids, n_present
