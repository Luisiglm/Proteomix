"""
Run-level normalisation of log2 intensities (NaN = missing).

All methods take an (N_runs, F) tensor and return (y_norm_log2, log2_offsets or None):
    median_ratio  median-of-ratios (DESeq2-style) against the per-feature log geometric mean
    total         total-intensity scaling: equal summed raw intensity per run
    quantile      quantile normalisation, NaN-aware
    none          no normalisation

Caveat: with abundance-dependent missingness, runs with more missing values have observed
distributions shifted upwards. Methods that use complete features (median_ratio, total) are
largely unaffected; quantile normalisation forces the observed distributions to be identical
and therefore pulls those runs down.
"""
import torch

NORMALISATIONS = ("median_ratio", "total", "quantile", "none")


def _to_log2(y, log_scale):
    return y.clone() if log_scale else torch.log2(torch.where(y > 0, y, torch.nan))


def median_of_ratios(y: torch.Tensor, log_scale: bool = True, min_complete: int = 10):
    """
    Median-of-ratios normalisation (DESeq2-style) done in log2 space, NaN-aware.

    Reference = per-feature mean of log2 values (log of the geometric mean). Features observed in
    every run are used if there are at least `min_complete` of them; otherwise every feature is
    used with NaNs ignored. Offsets are centred so their geometric mean is 1.
    """
    ly = _to_log2(y, log_scale)
    complete = ~torch.isnan(ly).any(dim=0)
    use = ly[:, complete] if complete.sum() >= min_complete else ly
    ref = torch.nanmean(use, dim=0, keepdim=True)
    log_sf = torch.nanmedian(use - ref, dim=1).values
    log_sf = log_sf - log_sf.mean()
    return ly - log_sf[:, None], log_sf


def total_intensity(y: torch.Tensor, log_scale: bool = True, min_complete: int = 10):
    """
    Total-intensity normalisation: every run is scaled to the same summed raw intensity.
    Sums are taken over features observed in every run (if at least min_complete), so runs
    with more missing values are not penalised for it.
    """
    ly = _to_log2(y, log_scale)
    complete = ~torch.isnan(ly).any(dim=0)
    use = ly[:, complete] if complete.sum() >= min_complete else ly
    log_tot = torch.logsumexp(torch.nan_to_num(use, nan=-torch.inf) * torch.log(
        torch.tensor(2.0, dtype=ly.dtype)), dim=1) / torch.log(torch.tensor(2.0, dtype=ly.dtype))
    log_sf = log_tot - log_tot.mean()
    return ly - log_sf[:, None], log_sf


def quantile_normalise(y: torch.Tensor, log_scale: bool = True, n_grid: int | None = None):
    """
    NaN-aware quantile normalisation. The reference distribution is the average of the runs'
    empirical quantile functions on a common probability grid; each observed value is replaced
    by the reference quantile at its plotting position (rank - 0.5) / n_observed in its run.
    """
    ly = _to_log2(y, log_scale)
    N, Fn = ly.shape
    obs = ~torch.isnan(ly)
    n_obs = obs.sum(1)
    n_grid = int(n_obs.max()) if n_grid is None else n_grid
    grid = (torch.arange(n_grid, dtype=ly.dtype) + 0.5) / n_grid
    q = torch.stack([torch.quantile(ly[i, obs[i]], grid) for i in range(N)])
    ref = q.mean(0)
    out = torch.full_like(ly, torch.nan)
    for i in range(N):
        vals = ly[i, obs[i]]
        ranks = torch.empty_like(vals)
        ranks[torch.argsort(vals)] = torch.arange(len(vals), dtype=ly.dtype)
        pos = (ranks + 0.5) / len(vals)
        idx = pos * n_grid - 0.5                       # fractional index into the grid
        lo = idx.floor().clamp(0, n_grid - 1).long()
        hi = (lo + 1).clamp(max=n_grid - 1)
        frac = (idx - lo).clamp(0, 1)
        out[i, obs[i]] = ref[lo] * (1 - frac) + ref[hi] * frac
    return out, None


def normalise(y: torch.Tensor, method: str = "median_ratio", log_scale: bool = True):
    """Dispatch to a normalisation method; see NORMALISATIONS."""
    if method == "median_ratio":
        return median_of_ratios(y, log_scale=log_scale)
    if method == "total":
        return total_intensity(y, log_scale=log_scale)
    if method == "quantile":
        return quantile_normalise(y, log_scale=log_scale)
    if method == "none":
        return _to_log2(y, log_scale), None
    raise ValueError(f"unknown normalisation {method!r}; choose from {NORMALISATIONS}")
