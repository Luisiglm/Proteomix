"""Multiple-testing corrections."""
import numpy as np


def benjamini_hochberg(pvals) -> np.ndarray:
    """BH-adjusted p-values; NaNs are ignored (and stay NaN) and do not count towards m."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    m = ok.sum()
    if m == 0:
        return out
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out
