"""Normalisation, replicate collapsing, imputation, multiple testing, moderation, metadata."""
import numpy as np
import pytest
import torch
from scipy import stats
from scipy.special import polygamma

from proteomix import benjamini_hochberg, collapse_replicates, impute, normalise
from proteomix.metadata import (clean_condition, condition_order, contrast_vector,
                                parse_condition, read_run_metadata)
from proteomix.moderation import fit_variance_prior, moderate_variances, trigamma_inverse


# ---- normalisation -------------------------------------------------------------------
@pytest.mark.parametrize("method", ["median_ratio", "total"])
def test_normalisation_removes_known_run_offsets(method, rng):
    base = rng.normal(20, 2, 500)
    offsets = np.array([0.0, 0.5, -0.3, 1.0])
    y = base + offsets[:, None] + rng.normal(0, 0.05, (4, 500))
    yn, sf = normalise(torch.tensor(y), method)
    np.testing.assert_allclose(sf.numpy() - sf.numpy().mean(), offsets - offsets.mean(),
                               atol=0.05 if method == "median_ratio" else 0.15)
    assert np.std(np.median(yn.numpy(), 1)) < 0.05


def test_quantile_normalisation_equalises_distributions(rng):
    y = rng.normal(20, 2, (3, 400)) * np.array([[1.0], [1.3], [0.8]])
    yn, _ = normalise(torch.tensor(y), "quantile")
    q = np.sort(yn.numpy(), 1)
    np.testing.assert_allclose(q[0], q[1], atol=1e-8)


def test_normalisation_keeps_missing_values_missing(rng):
    y = rng.normal(20, 1, (4, 100))
    y[0, :10] = np.nan
    for method in ("median_ratio", "total", "quantile", "none"):
        yn, _ = normalise(torch.tensor(y), method)
        assert torch.isnan(yn).sum() == 10


# ---- replicates ------------------------------------------------------------------------
def test_collapse_replicates_means_and_design():
    y = torch.tensor([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0], [7.0, np.nan]], dtype=torch.float64)
    x = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float64)
    yc, xc, bc, ids, n = collapse_replicates(y, x, torch.zeros(4, dtype=torch.long),
                                             [0, 0, 1, 1])
    np.testing.assert_allclose(yc.numpy(), [[2.0, 4.0], [6.0, 6.0]])
    np.testing.assert_allclose(n.numpy(), [[2, 1], [2, 1]])
    assert torch.equal(xc, x[[0, 2]])


def test_collapse_replicates_rejects_inconsistent_design():
    y = torch.zeros(2, 3, dtype=torch.float64)
    x = torch.tensor([[1, 0], [0, 1]], dtype=torch.float64)
    with pytest.raises(ValueError):
        collapse_replicates(y, x, torch.zeros(2, dtype=torch.long), [0, 0])


# ---- imputation ------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["zero", "min_prob", "mean", "mean_noise", "mean_group", "knn"])
def test_imputation_fills_everything_and_keeps_observed(method, rng):
    y = rng.normal(20, 1, (10, 60))
    y[rng.random(y.shape) < 0.2] = np.nan
    y[:, 0] = np.nan                                         # a fully missing protein
    x = torch.tensor(np.eye(2)[np.repeat([0, 1], 5)])
    out = impute(torch.tensor(y), method, x=x, seed=0).numpy()
    assert not np.isnan(out).any()
    obs = ~np.isnan(y)
    np.testing.assert_allclose(out[obs], y[obs])


# ---- multiple testing ------------------------------------------------------------------
def test_benjamini_hochberg_matches_scipy_and_ignores_nan(rng):
    p = rng.random(200) ** 3
    np.testing.assert_allclose(benjamini_hochberg(p), stats.false_discovery_control(p))
    p2 = p.copy()
    p2[:20] = np.nan
    out = benjamini_hochberg(p2)
    assert np.isnan(out[:20]).all()
    np.testing.assert_allclose(out[20:], stats.false_discovery_control(p[20:]))


# ---- moderation ------------------------------------------------------------------------
def test_trigamma_inverse():
    x = np.array([0.05, 1.0, 10.0, 1e3])
    np.testing.assert_allclose(polygamma(1, trigamma_inverse(x)), x, rtol=1e-8)


def test_variance_prior_recovered(rng):
    G, d, d0, s0 = 20000, np.full(20000, 6.0), 10.0, 0.3
    sig2 = s0 ** 2 * d0 / rng.chisquare(d0, G)
    s2 = sig2 * rng.chisquare(d, G) / d
    pr = fit_variance_prior(s2, d)
    assert abs(pr["d0_global"] - d0) < 1.5
    assert abs(pr["s0_2"][0] - s0 ** 2) < 0.005
    post, df_total = moderate_variances(s2, d, pr)
    assert np.all(df_total > d)
    assert np.var(np.log(post)) < np.var(np.log(s2))


def test_variance_trend_and_constant_variances(rng):
    G, d = 10000, np.full(10000, 6.0)
    mu = rng.normal(22, 2, G)
    s0 = 0.3 * np.exp(0.35 * (22 - mu) / 2)
    s2 = s0 ** 2 * 10 / rng.chisquare(10, G) * rng.chisquare(d, G) / d
    pr = fit_variance_prior(s2, d, covariate=mu)
    assert np.median(np.abs(pr["s0_2"] / s0 ** 2 - 1)) < 0.05
    const = 0.09 * rng.chisquare(d, G) / d
    assert fit_variance_prior(const, d)["d0_global"] > 100


# ---- metadata ---------------------------------------------------------------------------
def test_condition_cleaning_and_parsing():
    assert clean_condition("IgG TCS PDL1 5min ") == "IgG TCS PDL1 5 min"
    assert clean_condition("IP  TCS Ctrl 4h") == "IP TCS Ctrl 4 h"
    assert parse_condition("IP TCS PDL1 20 min") == {"pulldown": "IP", "treatment": "PDL1",
                                                     "time_min": 20}
    assert parse_condition("IgG TCS Ctrl 4 h")["time_min"] == 240
    assert parse_condition("IgG unstimulated")["treatment"] == "unstimulated"


def test_contrast_vector_by_name_and_weights():
    g = ["A", "B", "C"]
    np.testing.assert_array_equal(contrast_vector(("C", "A"), g), [-1, 0, 1])
    np.testing.assert_array_equal(contrast_vector({"A": 0.5, "B": 0.5, "C": -1}, g),
                                  [0.5, 0.5, -1])
    with pytest.raises(KeyError):
        contrast_vector(("A", "D"), g)


@pytest.mark.parametrize("header", [True, False])
def test_read_run_metadata(tmp_path, header):
    path = tmp_path / "m.tsv"
    lines = ["run\tcondition"] if header else []
    lines += ["x_PD1_A1_S1_A1_1_100 MaxLFQ Intensity\tIP unstimulated",
              "x_PD1_A2_S1_A2_1_101 MaxLFQ Intensity\tIgG TCS Ctrl 5min "]
    path.write_text("\n".join(lines) + "\n")
    m = read_run_metadata(str(path))
    assert list(m["run"]) == ["x_PD1_A1_S1_A1_1_100", "x_PD1_A2_S1_A2_1_101"]
    assert list(m["condition"]) == ["IP unstimulated", "IgG TCS Ctrl 5 min"]
    assert condition_order(m) == ["IP unstimulated", "IgG TCS Ctrl 5 min"]


def test_read_run_metadata_sample_ids_without_header(tmp_path):
    path = tmp_path / "m.tsv"
    path.write_text("A10\tControl\nB10\tTreated\n")
    m = read_run_metadata(str(path), parse_factors=False)
    assert list(m["run"]) == ["A10", "B10"]                 # first row is data, not header
