"""
Publication-style figures for the machine-learning module (same style as proteomix.figures).

    plot_model_comparison   primary metric per model: folds (dots), mean +/- SD, pooled OOF
    plot_roc                out-of-fold ROC curves (binary classification)
    plot_confusion          out-of-fold confusion matrix (classification)
    plot_pred_vs_obs        out-of-fold predicted vs observed (regression)
    plot_km_risk            Kaplan-Meier curves of high- vs low-risk halves (survival)
    plot_selection          how often each protein was selected across outer folds
    plot_lasso_cv           cross-validated deviance along the lambda path
"""
import numpy as np
import matplotlib.pyplot as plt

from ..figures import OKABE_ITO, publication_style
from .metrics import HIGHER_IS_BETTER, PRIMARY, kaplan_meier, logrank_test, roc_auc, roc_curve

LABELS = {"rmse": "RMSE", "mae": "MAE", "r2": "R²", "pearson": "Pearson r", "accuracy":
          "Accuracy", "balanced_accuracy": "Balanced accuracy", "auc": "ROC AUC",
          "log_loss": "Log loss", "c_index": "C-index"}


def _colours(names):
    return {n: OKABE_ITO[i % len(OKABE_ITO)] for i, n in enumerate(names)}


def plot_model_comparison(cv, metric: str | None = None, font_size: float = 16,
                          figsize=None, title: str | None = None):
    metric = metric or PRIMARY[cv.task]
    fm = cv.fold_metrics
    models = list(dict.fromkeys(fm["model"]))
    col = _colours(models)
    figsize = figsize or (max(5.5, 1.6 * len(models) + 2.5), 5.5)
    rng = np.random.default_rng(0)
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        for i, m in enumerate(models):
            v = fm.loc[fm["model"] == m, metric].to_numpy(float)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=30, color=col[m], alpha=0.6,
                       edgecolor="none", zorder=2)
            ax.errorbar(i + 0.25, np.nanmean(v), yerr=np.nanstd(v), fmt="o", color="0.15",
                        capsize=5, ms=8, zorder=3)
            pooled = cv.pooled_metrics.loc[cv.pooled_metrics["model"] == m, metric].mean()
            ax.scatter(i + 0.25, pooled, marker="D", s=70, facecolor="white",
                       edgecolor="0.15", zorder=4)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models)
        ax.set_xlim(-0.6, len(models) - 0.3)
        if metric == "auc" or metric == "c_index":
            ax.axhline(0.5, color="0.6", ls="--", lw=1.2)
        ax.set_ylabel(LABELS.get(metric, metric))
        better = "higher is better" if HIGHER_IS_BETTER.get(metric, True) else "lower is better"
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color="0.5", label="outer folds"),
                           Line2D([], [], marker="o", ls="", color="0.15",
                                  label="mean ± SD"),
                           Line2D([], [], marker="D", ls="", markerfacecolor="white",
                                  markeredgecolor="0.15", color="0.15",
                                  label="pooled out-of-fold")],
                  loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.set_title(title or f"Nested cross-validation ({better})", loc="left")
        fig.tight_layout()
    return fig


def plot_roc(cv, font_size: float = 16, figsize=(7.5, 6.5), repeat: int = 0):
    if cv.classes is None or len(cv.classes) != 2:
        raise ValueError("ROC curves are for binary classification")
    y01 = cv.y == cv.classes[1]
    col = _colours(list(cv.oof))
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        for m, outs in cv.oof.items():
            score = outs[repeat][:, 1]
            fpr, tpr = roc_curve(y01, score)
            ax.plot(fpr, tpr, color=col[m], lw=2.2, label=f"{m} (AUC {roc_auc(y01, score):.2f})")
        ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_aspect("equal")
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
        ax.legend(loc="lower right")
        ax.set_title(f"Out-of-fold ROC ({cv.classes[1]} vs {cv.classes[0]})", loc="left")
        fig.tight_layout()
    return fig


def plot_confusion(cv, model: str, font_size: float = 16, repeat: int = 0, figsize=None):
    K = len(cv.classes)
    pred = cv.oof[model][repeat].argmax(1)
    yi = np.searchsorted(cv.classes, cv.y)
    C = np.zeros((K, K), int)
    np.add.at(C, (yi, pred), 1)
    frac = C / np.maximum(C.sum(1, keepdims=True), 1)
    figsize = figsize or (1.2 * K + 3.5, 1.1 * K + 2.5)
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{C[i, j]}", ha="center", va="center",
                        color="white" if frac[i, j] > 0.6 else "0.1", fontsize=font_size - 2)
        ax.set_xticks(range(K))
        ax.set_xticklabels(cv.classes, rotation=45, ha="right")
        ax.set_yticks(range(K))
        ax.set_yticklabels(cv.classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("Fraction of observed class")
        cb.outline.set_visible(False)
        ax.set_title(f"Out-of-fold confusion matrix: {model}", loc="left")
        fig.tight_layout()
    return fig


def plot_pred_vs_obs(cv, model: str, font_size: float = 16, repeat: int = 0,
                     figsize=(6.5, 6.5), outcome: str = "outcome"):
    pred = cv.oof[model][repeat]
    y = cv.y.astype(float)
    lo, hi = np.nanmin(np.r_[y, pred]), np.nanmax(np.r_[y, pred])
    pad = 0.05 * (hi - lo)
    r = np.corrcoef(y, pred)[0, 1]
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.6", ls="--", lw=1.2)
        ax.scatter(y, pred, s=45, color=OKABE_ITO[0], edgecolor="white", linewidth=0.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Observed {outcome}")
        ax.set_ylabel(f"Predicted {outcome}")
        ax.text(0.04, 0.96, f"r = {r:.2f}\nRMSE = {rmse:.2f}", transform=ax.transAxes,
                va="top", fontsize=font_size - 2)
        ax.set_title(f"Out-of-fold predictions: {model}", loc="left")
        fig.tight_layout()
    return fig


def plot_km_risk(cv, model: str, font_size: float = 16, repeat: int = 0, figsize=(7.5, 6),
                 time_label: str = "Time"):
    risk = cv.oof[model][repeat]
    high = risk > np.median(risk)
    t, e = cv.y.astype(float), cv.event
    _, p = logrank_test(t, e, high)
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        for grp, lab, c in ((~high, "Low predicted risk", OKABE_ITO[0]),
                            (high, "High predicted risk", OKABE_ITO[1])):
            tt, ss = kaplan_meier(t[grp], e[grp])
            tt, ss = np.r_[tt, t[grp].max()], np.r_[ss, ss[-1]]      # extend to last follow-up
            ax.step(tt, ss, where="post", color=c, lw=2.2, label=f"{lab} (n = {grp.sum()})")
            cens = grp & (e == 0)
            if cens.any():
                s_at = np.array([ss[np.searchsorted(tt, x, side="right") - 1] for x in t[cens]])
                ax.plot(t[cens], s_at, "|", color=c, ms=10, mew=1.5)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, None)
        ax.set_xlabel(time_label)
        ax.set_ylabel("Survival probability")
        ax.legend(loc="lower left")
        ax.text(0.97, 0.96, f"log-rank p = {p:.2g}", transform=ax.transAxes, ha="right",
                va="top", fontsize=font_size - 2)
        ax.set_title(f"Out-of-fold risk groups: {model}", loc="left")
        fig.tight_layout()
    return fig


def plot_selection(cv, model: str, top: int = 25, font_size: float = 16, figsize=None):
    sel = cv.selection[model].head(top)[::-1]
    figsize = figsize or (7.5, 0.33 * len(sel) + 2)
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(np.arange(len(sel)), sel.to_numpy(), color=OKABE_ITO[2], edgecolor="none")
        ax.set_yticks(np.arange(len(sel)))
        ax.set_yticklabels(sel.index, fontsize=max(font_size - 4, 8))
        ax.set_xlim(0, 1)
        ax.set_xlabel("Selection frequency (outer folds)")
        ax.set_title(f"Most frequently selected proteins: {model}", loc="left")
        fig.tight_layout()
    return fig


def plot_lasso_cv(model, font_size: float = 16, figsize=(7.5, 5.5), title: str | None = None):
    """model: ElasticNet returned by cv_elastic_net."""
    ll = np.log10(model.lambdas_)
    m, se = model.cv_mean_[: len(ll)], model.cv_se_[: len(ll)]
    nz = [(c != 0).any(0).sum() for c in model.coef_path_]
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        ax.fill_between(ll, m - se, m + se, color=OKABE_ITO[5], alpha=0.3, lw=0)
        ax.plot(ll, m, color=OKABE_ITO[0], lw=2)
        for lam, lab, ls in ((model.lambda_min_, "λ min", "--"), (model.lambda_1se_, "λ 1SE", ":")):
            ax.axvline(np.log10(lam), color="0.3", ls=ls, lw=1.4, label=lab)
        ax.set_xlabel("log$_{10}$ λ")
        ax.set_ylabel("Cross-validated deviance")
        ax.invert_xaxis()
        top = ax.twiny()
        top.set_xlim(ax.get_xlim())
        idx = np.linspace(0, len(ll) - 1, min(6, len(ll))).astype(int)
        top.set_xticks(ll[idx])
        top.set_xticklabels([str(nz[i]) for i in idx], fontsize=font_size - 4)
        top.set_xlabel("Proteins selected", fontsize=font_size - 2)
        top.spines["top"].set_visible(True)
        ax.legend(loc="upper left")
        ax.set_title(title or "Lasso: choice of λ", loc="left", pad=50)
        fig.tight_layout()
    return fig
