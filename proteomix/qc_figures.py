"""Figures for qc_extended.py (same publication style as figures.py)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .figures import (OKABE_ITO, _colour_map, _group_legend, _np, _order_by_group,
                      _width_and_tick_size, publication_style)

FLAG = "#111111"


def _bars(ax, values, groups, names, flags, colours, ylabel, tick_fs):
    order = _order_by_group(groups, names)
    cols = [FLAG if flags[i] else colours[groups[i]] for i in order]
    ax.bar(np.arange(len(order)), np.asarray(values)[order], color=cols, width=0.8,
           edgecolor="none")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(np.asarray(names)[order], rotation=90, fontsize=tick_fs)
    ax.set_xlim(-0.6, len(order) - 0.4)


def plot_identifications_contaminants(runtab, font_size: float = 16, figsize=None):
    """Proteins quantified per run and % contaminant intensity (flagged runs in black)."""
    g = runtab["group"].astype(str).to_numpy()
    names = runtab["run"].astype(str).to_numpy()
    colours = _colour_map(g[_order_by_group(g, names)])
    has_c = runtab["contaminant_pct"].notna().any()
    width, tick_fs = _width_and_tick_size(len(names), font_size)
    figsize = figsize or (width + 3, 4.3 * (2 if has_c else 1) + 1)
    with publication_style(font_size):
        fig, axes = plt.subplots(2 if has_c else 1, 1, figsize=figsize, sharex=True,
                                 squeeze=False)
        _bars(axes[0, 0], runtab["n_quantified"], g, names,
              runtab["low_identifications"].to_numpy(), colours, "Features quantified", tick_fs)
        axes[0, 0].set_title("Identifications per run", loc="left", fontsize=font_size + 1)
        if has_c:
            _bars(axes[1, 0], runtab["contaminant_pct"], g, names,
                  runtab["high_contaminants"].to_numpy(), colours, "Contaminant intensity (%)",
                  tick_fs)
            axes[1, 0].set_title("Contaminants", loc="left", fontsize=font_size + 1)
        _group_legend(axes[0, 0], colours, extra=[Patch(facecolor=FLAG, label="Flagged")])
        fig.tight_layout()
    return fig


def plot_cv(cv_long, font_size: float = 16, figsize=None, max_cv: float = 150):
    """Distribution of protein CVs (%) per group (biological) and between injections."""
    groups = list(dict.fromkeys(cv_long["group"]))
    colours = _colour_map([g for g in groups if g != "technical"])
    colours["technical"] = "0.55"
    data = [np.clip(cv_long.loc[cv_long["group"] == g, "cv_pct"].dropna().to_numpy(), 0, max_cv)
            for g in groups]
    figsize = figsize or (max(6, 0.7 * len(groups) + 3), 5.5)
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        parts = ax.violinplot(data, showextrema=False, widths=0.85)
        for b, g in zip(parts["bodies"], groups):
            b.set_facecolor(colours[g])
            b.set_alpha(0.6)
        meds = [np.median(d) if len(d) else np.nan for d in data]
        ax.scatter(np.arange(1, len(groups) + 1), meds, color="0.1", zorder=3, s=30)
        for i, m in enumerate(meds):
            ax.text(i + 1, m, f" {m:.0f}%", va="center", fontsize=font_size - 5)
        ax.set_xticks(np.arange(1, len(groups) + 1))
        ax.set_xticklabels([("Technical\n(injections)" if g == "technical" else g)
                            for g in groups], rotation=45, ha="right")
        ax.set_ylabel("Coefficient of variation (%)")
        ax.set_title("Variability within groups (dot = median)", loc="left")
        fig.tight_layout()
    return fig


def plot_replicate_pairs(Y_runs, pairs, run_names, n_show: int = 8, font_size: float = 16,
                         figsize=None):
    """Scatter of the injections of the least concordant samples (lowest correlation)."""
    Y = _np(Y_runs)
    idx = {r: i for i, r in enumerate(run_names)}
    show = pairs.sort_values("pearson_r").head(n_show)
    ncols = min(4, len(show))
    nrows = int(np.ceil(len(show) / ncols))
    figsize = figsize or (3.6 * ncols, 3.6 * nrows + 0.8)
    lo, hi = np.nanquantile(Y, [0.001, 0.999])
    with publication_style(font_size):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        for k, (_, r) in enumerate(show.iterrows()):
            ax = axes[k // ncols, k % ncols]
            a, b = Y[idx[r["run_a"]]], Y[idx[r["run_b"]]]
            ok = ~np.isnan(a) & ~np.isnan(b)
            ax.scatter(a[ok], b[ok], s=6, alpha=0.4, color=OKABE_ITO[0], edgecolor="none",
                       rasterized=True)
            ax.plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_title(f"{r['sample']}  r = {r['pearson_r']:.3f}", loc="left",
                         fontsize=font_size - 3)
            ax.text(0.04, 0.92, f"median Δ {r['median_log2_diff']:+.2f}", transform=ax.transAxes,
                    fontsize=font_size - 6)
            ax.tick_params(labelsize=font_size - 5)
        for k in range(len(show), nrows * ncols):
            axes[k // ncols, k % ncols].axis("off")
        fig.supxlabel("Injection 1 (log$_2$)", fontsize=font_size)
        fig.supylabel("Injection 2 (log$_2$)", fontsize=font_size)
        fig.suptitle("Least concordant technical replicates", x=0.01, ha="left",
                     fontweight="bold", fontsize=font_size + 1)
        fig.tight_layout()
    return fig


def plot_mean_variance(mv, font_size: float = 16, figsize=(7, 5.5), n_bins: int = 20):
    """Pooled within-group SD against mean log2 intensity with a binned median trend."""
    t = mv.dropna(subset=["mean_log2", "pooled_sd"])
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(t["mean_log2"], t["pooled_sd"], s=8, alpha=0.35, color=OKABE_ITO[0],
                   edgecolor="none", rasterized=True)
        edges = np.quantile(t["mean_log2"], np.linspace(0, 1, n_bins + 1))
        mids, meds = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (t["mean_log2"] >= lo) & (t["mean_log2"] <= hi)
            if m.sum() >= 5:
                mids.append(t.loc[m, "mean_log2"].median())
                meds.append(t.loc[m, "pooled_sd"].median())
        ax.plot(mids, meds, color=OKABE_ITO[1], lw=2.5, label="binned median")
        ax.set_xlabel("Mean log$_2$ intensity")
        ax.set_ylabel("Within-group SD (log$_2$)")
        ax.set_ylim(0, np.nanquantile(t["pooled_sd"], 0.995) * 1.1)
        ax.legend(loc="upper right")
        ax.set_title("Mean–variance relationship", loc="left")
        fig.tight_layout()
    return fig


def plot_sample_correlation(corr, order, sample_groups, font_size: float = 16, figsize=None):
    """Clustered sample-sample correlation heatmap with a group colour bar."""
    C = corr.to_numpy()[np.ix_(order, order)]
    names = np.asarray(corr.index)[order]
    g = np.asarray(sample_groups).astype(str)[order]
    colours = _colour_map(np.asarray(sample_groups).astype(str))
    n = len(names)
    side, tick_fs = _width_and_tick_size(n, font_size, per_item=0.3, extra=4, max_width=16)
    figsize = figsize or (side + 3, side)
    off = C[~np.eye(n, dtype=bool)]
    vmin = np.nanquantile(off, 0.02)
    with publication_style(font_size):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.03], height_ratios=[0.03, 1],
                              wspace=0.03, hspace=0.02)
        axb = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0])
        axc = fig.add_subplot(gs[1, 1])
        levels = list(colours)
        from matplotlib.colors import ListedColormap
        axb.imshow([[levels.index(x) for x in g]], aspect="auto",
                   cmap=ListedColormap([colours[x] for x in levels]), vmin=-0.5,
                   vmax=len(levels) - 0.5, interpolation="nearest")
        axb.axis("off")
        im = ax.imshow(C, cmap="viridis", vmin=vmin, vmax=1, interpolation="nearest")
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=90, fontsize=tick_fs)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=tick_fs)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, cax=axc)
        cb.set_label("Pearson correlation")
        cb.outline.set_visible(False)
        handles = [Patch(facecolor=colours[x], label=x) for x in levels]
        axb.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.2),
                   ncol=min(4, len(handles)), fontsize=font_size - 4, frameon=False)
        axb.set_title("Sample correlation (clustered)", loc="left",
                      pad=(1.5 + 1.2 * np.ceil(len(handles) / 4)) * font_size)
    return fig


def plot_run_order(order_tab, drift, font_size: float = 16, figsize=None):
    """Run metrics against acquisition order, coloured by group, with drift tests."""
    g = order_tab["group"].astype(str).to_numpy()
    colours = _colour_map(g)
    metrics = [("median_log2", "Median log$_2$ intensity"),
               ("n_quantified", "Features quantified"), ("missing_pct", "Missing (%)")]
    figsize = figsize or (max(8, len(order_tab) * 0.12 + 5), 3.2 * len(metrics) + 1)
    with publication_style(font_size):
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
        for ax, (m, lab) in zip(axes, metrics):
            for lev, c in colours.items():
                sel = g == lev
                ax.scatter(order_tab.loc[sel, "order"], order_tab.loc[sel, m], color=c, s=40,
                           edgecolor="white", linewidth=0.6, label=lev)
            r = drift.set_index("metric").loc[m]
            ax.text(0.99, 0.95, f"within-group ρ = {r['spearman_rho_within_group']:+.2f}, "
                    f"p = {r['pvalue_within_group']:.2g}" + ("  DRIFT" if r["drift"] else ""),
                    transform=ax.transAxes, ha="right", va="top", fontsize=font_size - 4,
                    color="#B2182B" if r["drift"] else "0.3")
            ax.set_ylabel(lab, fontsize=font_size - 1)
        axes[-1].set_xlabel("Acquisition order")
        handles = [Line2D([], [], marker="o", ls="", color=c, label=lev)
                   for lev, c in colours.items()]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 0.95),
                   fontsize=font_size - 4, frameon=False)
        fig.suptitle("Run order", x=0.01, ha="left", fontweight="bold", fontsize=font_size + 1)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_confounding(xtabs: dict, conf, font_size: float = 16, figsize=None):
    """Counts of runs per condition and factor level, one panel per factor."""
    facs = list(xtabs)
    n_cond = max(t.shape[0] for t in xtabs.values())
    figsize = figsize or (4 + 3.2 * len(facs), 0.42 * n_cond + 2.5)
    with publication_style(font_size):
        fig, axes = plt.subplots(1, len(facs), figsize=figsize, squeeze=False)
        for k, (ax, f) in enumerate(zip(axes[0], facs)):
            t = xtabs[f]
            ax.imshow(t.to_numpy() > 0, cmap="Greys", vmin=0, vmax=1.6, aspect="auto")
            for i in range(t.shape[0]):
                for j in range(t.shape[1]):
                    if t.iat[i, j]:
                        ax.text(j, i, str(t.iat[i, j]), ha="center", va="center",
                                fontsize=font_size - 6)
            ax.set_xticks(range(t.shape[1]))
            ax.set_xticklabels(t.columns.astype(str), rotation=45, ha="right",
                               fontsize=font_size - 4)
            ax.set_yticks(range(t.shape[0]))
            ax.set_yticklabels(t.index if k == 0 else [], fontsize=font_size - 4)
            r = conf.set_index("factor").loc[f]
            ax.set_title(f"{f}\nV = {r['cramers_v']:.2f}" + (" (confounded)" if r["confounded"]
                                                            else ""),
                         loc="left", fontsize=font_size - 2,
                         color="#B2182B" if r["confounded"] else "0.1")
        fig.suptitle("Runs per condition and run-level factor", x=0.01, ha="left",
                     fontweight="bold", fontsize=font_size + 1)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_bait(bt, bait, font_size: float = 16, figsize=None):
    """Bait intensity per sample; missing values shown at the bottom, failed pull-downs black."""
    g = bt["group"].astype(str).to_numpy()
    names = bt["sample"].astype(str).to_numpy()
    colours = _colour_map(g[_order_by_group(g, names)])
    v = bt["bait_log2"].to_numpy(float)
    floor = np.nanmin(v) - 1 if np.isfinite(v).any() else 0
    width, tick_fs = _width_and_tick_size(len(names), font_size)
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize or (width + 3, 5.5))
        order = _order_by_group(g, names)
        for k, i in enumerate(order):
            miss = np.isnan(v[i])
            c = FLAG if bt["failed_pulldown"].iat[i] else colours[g[i]]
            ax.scatter(k, floor if miss else v[i], s=70, marker="x" if miss else "o",
                       color=c, zorder=3)
        ax.axhline(floor + 0.5, color="0.8", lw=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(names[order], rotation=90, fontsize=tick_fs)
        ax.set_ylabel(f"{bait} log$_2$ intensity")
        ax.set_title(f"Bait recovery: {bait} (× = not detected, black = flagged)", loc="left")
        _group_legend(ax, colours)
        fig.tight_layout()
    return fig


def plot_pulldown_counts(per, font_size: float = 16, figsize=(6, 5)):
    """Proteins quantified per sample, IP vs control pull-downs."""
    levels = list(dict.fromkeys(per["pulldown"]))
    data = [per.loc[per["pulldown"] == lev, "n_quantified"].to_numpy() for lev in levels]
    rng = np.random.default_rng(0)
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        for i, (lev, d) in enumerate(zip(levels, data)):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=40,
                       color=OKABE_ITO[i % len(OKABE_ITO)], edgecolor="white")
            ax.hlines(np.median(d), i - 0.3, i + 0.3, color="0.1", lw=2)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels)
        ax.set_xlim(-0.6, len(levels) - 0.4)
        ax.set_ylabel("Proteins quantified per sample")
        ax.set_title("Pull-down depth", loc="left")
        fig.tight_layout()
    return fig


def plot_phospho_qc(res, font_size: float = 16, figsize=None):
    """S/T/Y composition, localised share and (if available) phosphopeptide enrichment."""
    panels = [("residues", "Sites\n(%)")]
    if "localised_pct" in res and res["localised_pct"].notna().any():
        panels.append(("localised_pct", "Localised\nsites (%)"))
    if "phospho_intensity_pct" in res and res["phospho_intensity_pct"].notna().any():
        panels.append(("phospho_intensity_pct", "Phospho-\npeptide\nintensity (%)"))
    names = res["run"].astype(str).to_numpy()
    g = res["group"].astype(str).to_numpy()
    order = _order_by_group(g, names)
    colours = _colour_map(g[order])
    width, tick_fs = _width_and_tick_size(len(names), font_size)
    figsize = figsize or (width + 3, 3.6 * len(panels) + 1)
    with publication_style(font_size):
        fig, axes = plt.subplots(len(panels), 1, figsize=figsize, sharex=True, squeeze=False)
        for ax, (col, lab) in zip(axes[:, 0], panels):
            if col == "residues":
                bottom = np.zeros(len(order))
                for aa, c in zip(("S", "T", "Y", "other"),
                                 ("#3B3B3B", "#8C8C8C", "#7B3294", "#DDDDDD")):
                    v = res[f"{aa}_pct"].to_numpy()[order]
                    ax.bar(np.arange(len(order)), v, bottom=bottom, color=c, width=0.85,
                           edgecolor="none", label=aa)
                    bottom += v
                ax.set_ylim(0, 100)
                ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Residue")
            else:
                ax.bar(np.arange(len(order)), res[col].to_numpy()[order],
                       color=[colours[g[i]] for i in order], width=0.8, edgecolor="none")
            ax.set_ylabel(lab, fontsize=font_size - 1)
        axes[-1, 0].set_xticks(np.arange(len(order)))
        axes[-1, 0].set_xticklabels(names[order], rotation=90, fontsize=tick_fs)
        if len(panels) > 1:
            _group_legend(axes[1, 0], colours)
        fig.suptitle("Phospho QC", x=0.01, ha="left", fontweight="bold", fontsize=font_size + 1)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
