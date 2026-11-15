"""Statistical core: likelihood derivatives, missingness fit, model fitting and inference."""
import math

import numpy as np
import pytest
import torch
from scipy import integrate, stats
from scipy.special import expit

from proteomix import DifferentialAbundanceNR, MissingParametersNR, run_differential_abundance


def _terms_for(model, y, x, w, r, beta, tau):
    """Evaluate model._terms on a tiny problem (sets the private fit state)."""
    obs = (~torch.isnan(y)).double()
    model._X, model._y0, model._obs = x, torch.nan_to_num(y), obs
    model._batch = torch.zeros(y.shape[0], dtype=torch.long)
    model._mis_n, model._mis_f = torch.nonzero(obs == 0, as_tuple=True)
    model._offset = torch.zeros_like(y)
    model._prior_mean = torch.zeros(y.shape[1], dtype=torch.float64)
    return model._terms(beta, tau)


@pytest.mark.parametrize("include_sigma", [True, False])
def test_gradient_and_hessian_match_finite_differences(include_sigma):
    torch.manual_seed(0)
    y = torch.tensor([[20.1], [19.4], [float("nan")], [21.0], [float("nan")], [20.5]],
                     dtype=torch.float64)
    x = torch.tensor([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]], dtype=torch.float64)
    m = DifferentialAbundanceNR(torch.tensor([-1.4]), torch.tensor([19.8]),
                                include_sigma=include_sigma, beta_prior_var=None)
    theta = np.array([19.9, 20.4, math.log(0.7)])

    def ll(th):
        b = torch.tensor(np.array([th[:2]]), dtype=torch.float64)
        t = torch.tensor([th[2]], dtype=torch.float64)
        return _terms_for(m, y, x, None, None, b, t)[0].item()

    _, g, H = _terms_for(m, y, x, None, None, torch.tensor(np.array([theta[:2]])),
                         torch.tensor([theta[2]]))
    h = 1e-5
    E = np.eye(3) * h
    g_num = np.array([(ll(theta + E[i]) - ll(theta - E[i])) / (2 * h) for i in range(3)])
    H_num = np.array([[(ll(theta + E[i] + E[j]) - ll(theta + E[i] - E[j])
                        - ll(theta - E[i] + E[j]) + ll(theta - E[i] - E[j])) / (4 * h * h)
                       for j in range(3)] for i in range(3)])
    np.testing.assert_allclose(g.numpy()[0], g_num, atol=1e-6)
    np.testing.assert_allclose(H.numpy()[0], H_num, atol=1e-4)


def test_sigmoid_approximation_close_to_exact_integral():
    w, r, s = -1.3, 9.0, 0.8
    for mu in (7.0, 9.0, 10.5):
        exact = integrate.quad(lambda v: stats.norm.pdf(v, mu, s) * expit(w * (v - r)),
                               mu - 10 * s, mu + 10 * s)[0]
        approx = expit(w * (mu - r) / math.sqrt(1 + math.pi / 8 * w * w * s * s))
        assert abs(exact - approx) < 0.01


def test_missingness_parameters_recovered():
    rng = np.random.default_rng(1)
    n, F = 12, 2000
    g = np.repeat([0, 1], n // 2)
    true = rng.normal(21, 2, F) + rng.normal(0, 0.2, (n, F))
    miss = rng.random((n, F)) < expit(-1.5 * (true - 19.5))
    y = torch.tensor(np.where(miss, np.nan, true))
    m = MissingParametersNR(1).double()
    hist = m.fit(y, torch.tensor(np.eye(2)[g]), torch.zeros(n, dtype=torch.long))
    assert np.all(np.diff(hist) >= -1e-8)
    assert abs(m.w.item() + 1.5) < 0.3
    assert abs(m.r.item() - 19.5) < 0.3


def test_fit_is_monotone_and_converges(two_group_data):
    d = two_group_data
    res, model, miss, _ = run_differential_abundance(d["Y"], d["X"], d["batch"], (1, 0))
    assert np.all(np.diff(model.ll_history) >= -1e-6 * abs(model.ll_history[0]))
    assert np.nanmax(model.grad_norm.numpy()) < 1e-3
    assert {"estimate", "se", "pvalue", "padj", "fitted"} <= set(res.columns)


def test_detects_de_and_estimates_unbiased(two_group_data):
    d = two_group_data
    res, *_ = run_differential_abundance(d["Y"], d["X"], d["batch"], (1, 0), moderate=True)
    sig = (res["padj"] < 0.05).to_numpy()
    tested = res["fitted"].to_numpy()
    assert sig[d["de"] & tested].mean() > 0.8                      # power
    assert (sig & ~d["de"]).sum() / max(sig.sum(), 1) < 0.1        # FDP
    err = (res["estimate"].to_numpy() - d["lfc"])[d["de"] & tested]
    assert abs(np.nanmean(err * np.sign(d["lfc"][d["de"] & tested]))) < 0.2


def test_complete_data_matches_ordinary_t_test():
    """With no missing values and no moderation, the Wald t-test is the pooled t-test."""
    rng = np.random.default_rng(2)
    y = rng.normal(10, 1, (10, 40))
    y[5:, :5] += 2
    g = np.repeat([0, 1], 5)
    res, *_ = run_differential_abundance(torch.tensor(y), torch.tensor(np.eye(2)[g]),
                                         torch.zeros(10, dtype=torch.long), (1, 0),
                                         normalisation="none", missing="none",
                                         beta_prior_var=None)
    t_ref = stats.ttest_ind(y[5:], y[:5])
    np.testing.assert_allclose(res["stat"].to_numpy(), t_ref.statistic, rtol=1e-4)
    np.testing.assert_allclose(res["pvalue"].to_numpy(), t_ref.pvalue, rtol=1e-3)


def test_offset_equals_subtracting_it():
    """Without missing values, a known offset is the same as analysing y - offset."""
    rng = np.random.default_rng(4)
    y = rng.normal(10, 1, (8, 30))
    off = rng.normal(0, 1, (8, 30))
    x = torch.tensor(np.eye(2)[np.repeat([0, 1], 4)])
    b = torch.zeros(8, dtype=torch.long)
    m1 = DifferentialAbundanceNR(missingness="ignore", beta_prior_var=None)
    m1.fit(torch.tensor(y), x, b, offset=torch.tensor(off))
    m2 = DifferentialAbundanceNR(missingness="ignore", beta_prior_var=None)
    m2.fit(torch.tensor(y - off), x, b)
    np.testing.assert_allclose(m1.beta.numpy(), m2.beta.numpy(), atol=1e-6)
    np.testing.assert_allclose(m1.sigma.numpy(), m2.sigma.numpy(), rtol=1e-5)


def test_filters_mark_features_not_fitted(two_group_data):
    d = two_group_data
    r0, *_ = run_differential_abundance(d["Y"], d["X"], d["batch"], (1, 0))
    r1, *_ = run_differential_abundance(d["Y"], d["X"], d["batch"], (1, 0),
                                        min_valid_per_group=8, min_groups=2)
    assert r1["fitted"].sum() < r0["fitted"].sum()
    assert r1.loc[~r1["fitted"], "pvalue"].isna().all()


@pytest.mark.slow
def test_type_one_error_calibrated_under_null():
    """Pooled over a few null simulations, the per-test type I error is close to 5%."""
    from proteomix import simulate_dataset
    rates = []
    for seed in range(4):
        dd = simulate_dataset(seed=seed, prop_de=0.0, n_features=800)
        res, *_ = run_differential_abundance(
            torch.tensor(dd["Y"]), torch.tensor(dd["X"]), torch.tensor(dd["batch_ids"]),
            (1, 0), replicate_ids=torch.tensor(dd["replicate_ids"]), moderate=True)
        p = res["pvalue"].dropna().to_numpy()
        rates.append(np.mean(p < 0.05))
    assert 0.03 < np.mean(rates) < 0.07
