"""Machine-learning module: solvers vs reference implementations, metrics, CV, persistence."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from proteomix.ml import (KNN, MLP, ElasticNet, Preprocessor, TrainedModel, cross_validate,
                          cv_elastic_net, make_folds, make_learner, train_final)
from proteomix.ml.metrics import classification_metrics, concordance_index, roc_auc


@pytest.fixture
def xy(rng):
    n, p = 80, 120
    X = rng.normal(size=(n, p))
    b = np.zeros(p)
    b[:6] = [1.5, -1.2, 1.0, -0.8, 0.7, 0.6]
    return X, b, X @ b


# ---- elastic net against references -----------------------------------------------------
def test_gaussian_lasso_matches_sklearn(xy):
    sk = pytest.importorskip("sklearn.linear_model")
    X, b, s = xy
    y = s + np.random.default_rng(0).normal(0, 1, len(s))
    m = ElasticNet("gaussian", standardize=False, lambdas=[0.1]).fit(X, y)
    ref = sk.Lasso(alpha=0.1, tol=1e-12, max_iter=10 ** 6).fit(X, y)
    np.testing.assert_allclose(m.coef_[0], ref.coef_, atol=2e-4)
    m = ElasticNet("gaussian", alpha=0.5, standardize=False, lambdas=[0.1]).fit(X, y)
    ref = sk.ElasticNet(alpha=0.1, l1_ratio=0.5, tol=1e-12, max_iter=10 ** 6).fit(X, y)
    np.testing.assert_allclose(m.coef_[0], ref.coef_, atol=2e-4)


def test_logistic_lasso_matches_sklearn(xy):
    sk = pytest.importorskip("sklearn.linear_model")
    X, b, s = xy
    y = (s + np.random.default_rng(1).logistic(size=len(s)) > 0).astype(int)
    lam = 0.05
    m = ElasticNet("binomial", standardize=False, lambdas=[lam]).fit(X, y)
    ref = sk.LogisticRegression(l1_ratio=1.0, C=1 / (len(y) * lam), solver="liblinear",
                                tol=1e-10, max_iter=10 ** 5, intercept_scaling=1e6).fit(X, y)
    np.testing.assert_allclose(m.coef_[0], ref.coef_[0], atol=1e-3)


@pytest.mark.parametrize("hessian", ["full", "diagonal"])
def test_cox_matches_lifelines(hessian):
    ll = pytest.importorskip("lifelines")
    rng = np.random.default_rng(2)
    n, p = 150, 6
    X = rng.normal(size=(n, p))
    t = rng.exponential(1 / np.exp(X @ np.array([1.0, -0.8, 0.5, 0, 0, 0])))
    c = rng.exponential(2, n)
    time, ev = np.minimum(t, c), (t <= c).astype(int)
    m = ElasticNet("cox", alpha=0.0, standardize=False, lambdas=[1e-4],
                   cox_hessian=hessian).fit(X, time, ev)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)]).assign(T=time, E=ev)
    ref = ll.CoxPHFitter(penalizer=1e-4, l1_ratio=0.0).fit(df, "T", "E")
    np.testing.assert_allclose(m.coef_[0], ref.params_.to_numpy(), atol=1e-3)


def test_path_properties_and_cv(xy):
    X, b, s = xy
    y = s + np.random.default_rng(3).normal(0, 0.5, len(s))
    m = ElasticNet("gaussian").fit(X, y)
    assert np.all(np.diff(m.lambdas_) < 0)
    assert (m.coef_path_[0] == 0).all()                       # lambda_max: empty model
    cvm = cv_elastic_net(X, y, "gaussian", n_folds=5)
    assert cvm.lambda_1se_ >= cvm.lambda_min_
    assert set(range(6)) <= set(cvm.selected_features())


def test_multinomial_predicts_three_classes(rng):
    X = rng.normal(size=(90, 30))
    y = np.argmax(X[:, :3] * 2 + rng.gumbel(size=(90, 3)), 1)
    m = ElasticNet("multinomial", lambdas=[0.02]).fit(X, y)
    P = m.predict_proba(X)
    np.testing.assert_allclose(P.sum(1), 1)
    assert (m.predict(X) == y).mean() > 0.7


# ---- metrics ----------------------------------------------------------------------------
def test_auc_and_cindex_match_references(rng):
    skm = pytest.importorskip("sklearn.metrics")
    y = rng.integers(0, 2, 200)
    s = y + rng.normal(0, 1, 200)
    assert np.isclose(roc_auc(y, s), skm.roc_auc_score(y, s))
    ll = pytest.importorskip("lifelines.utils")
    t, e, r = rng.exponential(1, 100), rng.integers(0, 2, 100), rng.normal(0, 1, 100)
    assert np.isclose(concordance_index(t, e, r), ll.concordance_index(t, -r, e), atol=1e-6)


def test_classification_metrics_perfect():
    m = classification_metrics(["a", "b", "b"], np.array([[.9, .1], [.2, .8], [.1, .9]]),
                               np.array(["a", "b"]))
    assert m["accuracy"] == 1 and m["auc"] == 1 and m["balanced_accuracy"] == 1


# ---- folds, preprocessing, learners -----------------------------------------------------
def test_folds_partition_and_stratify():
    y = np.array(["a"] * 30 + ["b"] * 10)
    folds = make_folds(y, 5, "classification", seed=0)
    allidx = np.sort(np.concatenate(folds))
    np.testing.assert_array_equal(allidx, np.arange(40))
    assert all((y[f] == "b").sum() == 2 for f in folds)


def test_preprocessor_learns_from_training_only(rng):
    Y = pd.DataFrame(rng.normal(20, 1, (20, 50)))
    Y.iloc[rng.random((20, 50)) < 0.1] = np.nan
    pre = Preprocessor(impute="model").fit(Y.iloc[:15])
    before = (pre.center_.copy(), pre.scale_.copy())
    X_new = pre.transform(Y.iloc[15:] + 100)                   # wildly shifted new samples
    np.testing.assert_array_equal(before[0], pre.center_)       # parameters unchanged
    assert X_new.mean() > 10 and not np.isnan(X_new).any()


@pytest.mark.parametrize("task", ["regression", "classification"])
def test_knn_and_mlp_learn_signal(task, rng):
    X = rng.normal(size=(120, 4))                           # 1 informative of 4 features
    s = X[:, 0] * 2
    y = s + rng.normal(0, .3, 120) if task == "regression" else np.where(s > 0, "p", "n")
    for model in (KNN(task, k=5), MLP(task, hidden=(16,), epochs=300, seed=0)):
        model.fit(X[:90], y[:90])
        pred = model.predict(X[90:])
        if task == "regression":
            assert np.corrcoef(pred, y[90:])[0, 1] > 0.7
        else:
            assert (pred == y[90:]).mean() > 0.8


def test_mlp_cox_loss_equals_partial_likelihood(rng):
    import torch
    from proteomix.ml.lasso import _cox_loglik, _cox_prepare
    from proteomix.ml.mlp import cox_loss
    t = np.round(rng.exponential(1, 50), 1)                     # with ties
    e = rng.integers(0, 2, 50)
    eta = rng.normal(0, 1, 50)
    order, _, d, first, _ = _cox_prepare(t, e)
    ref = -_cox_loglik(eta[order], d, first) / max(e.sum(), 1)
    got = cox_loss(torch.tensor(eta), torch.tensor(t), torch.tensor(e, dtype=torch.float64))
    assert np.isclose(float(got), ref)


@pytest.mark.slow
def test_nested_cv_and_saved_model(rng, tmp_path):
    n, p = 50, 80
    mu = rng.normal(21, 1.5, p)
    Z = rng.normal(size=(n, p))
    true = mu + 0.5 * Z
    Y = pd.DataFrame(np.where(rng.random((n, p)) < expit(-1.5 * (true - 19)), np.nan, true),
                     index=[f"S{i}" for i in range(n)], columns=[f"P{i}" for i in range(p)])
    y = np.where(Z[:, 0] - Z[:, 1] + rng.normal(0, .5, n) > 0, "R", "N")
    learners = {nm: make_learner(nm, "classification") for nm in ("lasso", "knn")}
    cv = cross_validate(Y, y, "classification", learners, n_folds=4)
    assert set(cv.fold_metrics["model"]) == {"lasso", "knn"}
    assert cv.summary().loc["lasso", "auc_pooled"] > 0.7      # chance = 0.5
    assert {"P0", "P1"} <= set(cv.selection["lasso"].head(5).index)
    tm = train_final(Y, y, "classification", make_learner("lasso", "classification"))
    path = tmp_path / "m.pkl"
    tm.save(str(path))
    pred = TrainedModel.load(str(path)).predict(Y.iloc[:5])
    np.testing.assert_allclose(pred.filter(like="p_").sum(1), 1)
