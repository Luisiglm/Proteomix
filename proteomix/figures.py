"""
Publication-quality figures for a differential abundance analysis.

    plot_pca                      PCA of samples (complete features or a supplied complete matrix)
    plot_missingness_per_sample   % missing values per run / sample
    plot_total_intensity          total intensity per run, before and after normalisation
    plot_missingness_curve        fitted per-batch missingness curves with binned observed rates
    plot_volcano                  volcano plot for one contrast
    plot_heatmap                  row z-scored heatmap of the top proteins, clustered
    make_report_figures           all of the above for one pipeline run, saved to a folder

Style: large sans-serif fonts (Arial/Helvetica if installed), colour-blind-safe Okabe-Ito
palette, no top/right spines, outward ticks. Figures are saved as vector PDF (text kept as
editable TrueType, fonttype 42) and as 600 dpi PNG; TIFF (LZW) and SVG are available too.
600 dpi meets the line-art requirement of most journals; photographs usually need only 300.
"""
import os
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442",
             "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
UP, DOWN, NS = "#B2182B", "#2166AC", "#BDBDBD"


# --------------------------------------------------------------------------------------
# Style and saving
# --------------------------------------------------------------------------------------
@contextmanager
def publication_style(font_size: float = 16):
    """Matplotlib settings for publication figures; font_size is the tick-label size."""
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size + 2,
        "axes.titlesize": font_size + 2,
        "axes.titleweight": "bold",
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size - 2,
        "legend.title_fontsize": font_size - 1,
        "legend.frameon": False,
        "axes.linewidth": 1.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
    with plt.rc_context(rc):
        yield


def save_figure(fig, path_stem: str, formats=("pdf", "png"), dpi: int = 600) -> list[str]:
    """Saves fig as path_stem.<fmt> for each format; returns the written paths."""
    os.makedirs(os.path.dirname(path_stem) or ".", exist_ok=True)
    paths = []
    for fmt in formats:
        path = f"{path_stem}.{fmt}"
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if fmt in ("tif", "tiff") else {}
        fig.savefig(path, dpi=dpi, bbox_inches="tight", **kw)
        paths.append(path)
    return paths


def _colour_map(labels):
    """Okabe-Ito for up to 8 groups; tab20 (paired light/dark hues) for more."""
    levels = list(dict.fromkeys(labels))
    if len(levels) <= len(OKABE_ITO):
        return {lev: OKABE_ITO[i] for i, lev in enumerate(levels)}
    from matplotlib.colors import to_hex
    cmap = plt.get_cmap("tab20")
    return {lev: to_hex(cmap(i % 20)) for i, lev in enumerate(levels)}


def _np(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def _group_legend(ax, colours, title=None, loc="upper left", anchor=(1.01, 1.0), extra=()):
    handles = [Patch(facecolor=c, edgecolor="none", label=str(g)) for g, c in colours.items()]
    handles += list(extra)
    ax.legend(handles=handles, title=title, loc=loc, bbox_to_anchor=anchor)


# --------------------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------------------
def _confidence_ellipse(ax, pts, colour, level=0.95):
    if len(pts) < 3:
        return
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = np.sqrt(stats.chi2.ppf(level, 2))
    w, h = 2 * scale * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse(pts.mean(0), w, h, angle=angle, facecolor=colour, alpha=0.12,
                         edgecolor=colour, lw=1.5))


def _levels(labels):
    """Level order: numeric if every label is a number (e.g. time 0, 5, 20, 240), else first
    appearance."""
    uniq = list(dict.fromkeys(labels))
    try:
        return sorted(uniq, key=lambda v: float(v))
    except (TypeError, ValueError):
        return uniq


def plot_pca(y, groups, batches=None, sample_names=None, components=(1, 2),
             ellipses: bool | None = None, label_samples: bool = False, title: str | None = None,
             font_size: float = 16, figsize=(7.5, 6), shapes=None, colour_title: str | None = None,
             shape_title: str | None = None, highlight=None):
    """
    y: (n_samples, n_features) log2 intensities. PCA uses features observed in every sample
    (pass an imputed matrix to use all features). Features are centred, not scaled.
    groups: colour labels (any factor: group, pulldown, treatment, time ...).
    shapes: optional marker-shape labels (another factor); batches is the same with the
        title "Batch". Ignored if it has only one level.
    highlight: optional bool per sample; those samples are labelled and ringed (e.g. outliers).
    ellipses: 95% normal-theory regions; default on for up to 6 colour levels.
    """
    Y = _np(y).astype(float)
    complete = ~np.isnan(Y).any(0)
    Z = Y[:, complete]
    Z = Z - Z.mean(0)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    scores = U * S
    var = S ** 2 / np.sum(S ** 2) * 100
    i, j = components[0] - 1, components[1] - 1
    groups = np.asarray(groups).astype(str)
    colours = _colour_map(_levels(list(groups)))
    if shapes is None and batches is not None:
        shapes, shape_title = batches, shape_title or "Batch"
    shapes = None if shapes is None else np.asarray(shapes).astype(str)
    if shapes is not None and len(set(shapes)) < 2:
        shapes = None
    if ellipses is None:
        ellipses = len(colours) <= 6
    smark = None if shapes is None else {b: MARKERS[k % len(MARKERS)]
                                         for k, b in enumerate(_levels(list(shapes)))}

    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        for g, c in colours.items():
            sel = groups == g
            if ellipses:
                _confidence_ellipse(ax, scores[sel][:, [i, j]], c)
            for b, mk in (smark or {None: "o"}).items():
                sb = sel if b is None else sel & (shapes == b)
                ax.scatter(scores[sb, i], scores[sb, j], s=110, color=c, marker=mk,
                           edgecolor="white", linewidth=1.0, zorder=3)
        names = None if sample_names is None else np.asarray(sample_names).astype(str)
        if highlight is not None:
            hl = np.asarray(highlight, dtype=bool)
            ax.scatter(scores[hl, i], scores[hl, j], s=320, facecolor="none",
                       edgecolor="#B2182B", linewidth=2.0, zorder=4)
            if names is not None:
                for k in np.where(hl)[0]:
                    ax.annotate(names[k], (scores[k, i], scores[k, j]), xytext=(9, 7),
                                textcoords="offset points", fontsize=font_size - 3,
                                color="#B2182B", fontweight="bold")
        if label_samples and names is not None:
            for k, name in enumerate(names):
                ax.annotate(name, (scores[k, i], scores[k, j]), xytext=(5, 5),
                            textcoords="offset points", fontsize=font_size - 5)
        ax.set_xlabel(f"PC{components[0]} ({var[i]:.1f}%)")
        ax.set_ylabel(f"PC{components[1]} ({var[j]:.1f}%)")
        ax.axhline(0, color="0.85", lw=1, zorder=0)
        ax.axvline(0, color="0.85", lw=1, zorder=0)
        handles = [Line2D([], [], marker="o", ls="", markersize=11, color=c, label=g)
                   for g, c in colours.items()]
        if smark is not None:
            handles += [Line2D([], [], ls="", label="")]
            if shape_title:
                handles += [Line2D([], [], ls="", label=shape_title)]
            handles += [Line2D([], [], marker=mk, ls="", markersize=11, color="0.4",
                               label=(f"Batch {b}" if shape_title == "Batch" else b))
                        for b, mk in smark.items()]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  title=colour_title)
        ax.set_title(title or f"PCA ({complete.sum()} complete proteins)", loc="left")
        fig.tight_layout()
    return fig


def plot_protein_pca(y, sample_groups, results=None, alpha: float = 0.05, scale: bool = False,
                     components=(1, 2), n_labels: int = 12, feature_names=None,
                     title: str | None = None, font_size: float = 16, figsize=(15, 6.5)):
    """
    PCA with proteins as points (the transpose of the sample PCA).
    Each protein's profile across samples is centred (and, with scale=True, scaled to unit
    variance), so components describe patterns of change across samples rather than overall
    abundance. Only proteins observed in every sample are used.
    Left: protein scores, coloured by the number of contrasts in which the protein is
    significant (padj < alpha) when results ({label: table}) are given; the proteins with
    the largest scores are labelled. Right: the sample loadings on the same components,
    coloured by group, showing which samples (conditions) define each component.
    """
    Y = _np(y).astype(float)
    names = np.asarray(feature_names if feature_names is not None
                       else (next(iter(results.values())).index if results else
                             np.arange(Y.shape[1]))).astype(str)
    complete = ~np.isnan(Y).any(0)
    P = Y[:, complete].T                                   # proteins x samples
    P = P - P.mean(1, keepdims=True)
    if scale:
        sd = P.std(1, keepdims=True)
        P = P / np.where(sd > 0, sd, 1)
    P = P - P.mean(0)                                      # column-centre for the PCA
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    scores = U * S
    var = S ** 2 / np.sum(S ** 2) * 100
    i, j = components[0] - 1, components[1] - 1
    pnames = names[complete]
    n_sig = np.zeros(len(pnames), dtype=int)
    if results:
        for res in results.values():
            padj = res["padj"].reindex(pnames).to_numpy(float)
            n_sig += np.nan_to_num(padj, nan=1.0) < alpha
    groups = np.asarray(sample_groups).astype(str)
    colours = _colour_map(groups)

    with publication_style(font_size):
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=figsize,
                                      gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.45})
        order = np.argsort(n_sig)
        if results:
            vmax = max(int(n_sig.max()), 1)
            cmap = plt.get_cmap("viridis", vmax + 1)
            sc = ax.scatter(scores[order, i], scores[order, j], c=n_sig[order], cmap=cmap,
                            vmin=-0.5, vmax=vmax + 0.5, s=np.where(n_sig[order] > 0, 26, 10),
                            alpha=0.85, edgecolor="none", rasterized=True)
            cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.01,
                              ticks=range(0, vmax + 1, max(1, (vmax + 1) // 8)))
            cb.set_label(f"Contrasts significant (FDR < {alpha:g})")
            cb.outline.set_visible(False)
        else:
            ax.scatter(scores[:, i], scores[:, j], s=12, color=OKABE_ITO[0], alpha=0.7,
                       edgecolor="none", rasterized=True)
        if n_labels:
            dist = np.hypot(scores[:, i] / scores[:, i].std(), scores[:, j] / scores[:, j].std())
            for k in np.argsort(-dist)[:n_labels]:
                ax.annotate(pnames[k], (scores[k, i], scores[k, j]), xytext=(4, 4),
                            textcoords="offset points", fontsize=font_size - 6)
        for a in (ax, ax2):
            a.axhline(0, color="0.85", lw=1, zorder=0)
            a.axvline(0, color="0.85", lw=1, zorder=0)
        ax.set_xlabel(f"PC{components[0]} ({var[i]:.1f}%)")
        ax.set_ylabel(f"PC{components[1]} ({var[j]:.1f}%)")
        ax.set_title(f"Proteins ({complete.sum()} complete)", loc="left")
        L = Vt.T                                             # samples x components
        for g, c in colours.items():
            m = groups == g
            ax2.scatter(L[m, i], L[m, j], s=70, color=c, edgecolor="white", linewidth=0.8,
                        label=g)
        ax2.set_xlabel(f"PC{components[0]} loading")
        ax2.set_ylabel(f"PC{components[1]} loading")
        ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=font_size - 4)
        ax2.set_title("Sample loadings", loc="left")
        fig.suptitle(title or "PCA of protein profiles", x=0.01, ha="left",
                     fontweight="bold", fontsize=font_size + 2)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --------------------------------------------------------------------------------------
# Outlier QC
# --------------------------------------------------------------------------------------
def plot_sample_qc(qc_table, threshold: float = 3.5, title: str | None = None,
                   font_size: float = 16, figsize=None):
    """
    One panel per QC metric (robust z-scores, see qc.sample_qc); bars beyond the threshold in
    the flagged direction are red. Samples are ordered by group.
    """
    t = qc_table
    metrics = [("z_correlation", "Within-group\ncorrelation (z)", -1),
               ("z_distance", "Deviation from\ngroup (z)", 1),
               ("z_missing", "Missingness (z)", 1)]
    if "z_size_factor" in t:
        metrics.append(("z_size_factor", "Size factor (z)", 0))
    names = t["sample"].astype(str).to_numpy()
    groups = t["group"].astype(str).to_numpy()
    order = _order_by_group(groups, names)
    colours = _colour_map(groups[order])
    width, tick_fs = _width_and_tick_size(len(names), font_size)
    figsize = figsize or (width + 3.5, 2.6 * len(metrics) + 1.5)

    with publication_style(font_size):
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True, squeeze=False)
        for ax, (col, lab, direction) in zip(axes[:, 0], metrics):
            z = t[col].to_numpy(float)[order]
            bad = (z * direction > threshold) if direction else (np.abs(z) > threshold)
            cols = np.where(bad, "#111111", [colours[g] for g in groups[order]])
            ax.bar(np.arange(len(z)), np.nan_to_num(z), color=cols, width=0.8, edgecolor="none")
            for sgn in ([direction] if direction else [-1, 1]):
                ax.axhline(sgn * threshold, color="#B2182B", ls="--", lw=1.2)
            ax.axhline(0, color="0.4", lw=0.8)
            ax.set_ylabel(lab, fontsize=font_size)
            lim = max(threshold * 1.3, np.nanmax(np.abs(z)) * 1.1 if np.isfinite(z).any() else 1)
            ax.set_ylim(-lim, lim)
        ax = axes[-1, 0]
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names[order], rotation=90, fontsize=tick_fs)
        ax.set_xlim(-0.6, len(names) - 0.4)
        handles = [Patch(facecolor=c, edgecolor="none", label=str(g)) for g, c in colours.items()]
        handles.append(Patch(facecolor="#111111", label=f"Flagged (|z| > {threshold:g})"))
        handles.append(Line2D([], [], color="#B2182B", ls="--", label="Threshold"))
        n_out = int(t["outlier"].sum())
        fig.suptitle(title or f"Sample QC: {n_out} flagged", x=0.01, ha="left",
                     fontweight="bold", fontsize=font_size + 2)
        fig.tight_layout(rect=(0, 0, 0.78, 0.97))
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.79, 0.93),
                   frameon=False)
    return fig


# --------------------------------------------------------------------------------------
# QC bar charts
# --------------------------------------------------------------------------------------
def _width_and_tick_size(n, font_size, per_item=0.38, extra=3.0, max_width=16.0):
    """Figure width for n bars/columns (capped for print) and a tick size that still fits."""
    width = min(max(6.5, per_item * n + extra), max_width)
    tick = min(font_size, 0.7 * (width - extra) * 72 / max(n, 1))
    return width, max(tick, 5)


def _order_by_group(groups, names):
    groups = np.asarray(groups).astype(str)
    levels = list(dict.fromkeys(groups))
    return np.argsort([levels.index(g) for g in groups], kind="stable")


def plot_missingness_per_sample(y, groups, sample_names, title: str | None = None,
                                font_size: float = 16, figsize=None):
    """Bar chart of the percentage of missing values in each run or sample, coloured by group."""
    Y = _np(y)
    pct = np.isnan(Y).mean(1) * 100
    names = np.asarray(sample_names).astype(str)
    groups = np.asarray(groups).astype(str)
    order = _order_by_group(groups, names)
    colours = _colour_map(groups[order])
    width, tick_fs = _width_and_tick_size(len(names), font_size)
    figsize = figsize or (width, 5.5)

    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(np.arange(len(order)), pct[order], color=[colours[g] for g in groups[order]],
               width=0.8, edgecolor="none")
        mean_line = ax.axhline(pct.mean(), color="0.25", ls="--", lw=1.5,
                               label=f"Mean ({pct.mean():.1f}%)")
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(names[order], rotation=90, fontsize=tick_fs)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylabel("Missing values (%)")
        ax.set_ylim(0, max(pct.max() * 1.15, 1))
        _group_legend(ax, colours, extra=[mean_line])
        ax.set_title(title or "Missingness per sample", loc="left")
        fig.tight_layout()
    return fig


def plot_total_intensity(y_before, groups, sample_names, y_after=None, log2_input: bool = True,
                         title: str | None = None, font_size: float = 16, figsize=None):
    """
    Total intensity (sum of observed raw-scale intensities) per run, before and, if given,
    after normalisation. Inputs are log2 matrices unless log2_input=False.
    """
    def totals(m):
        m = _np(m).astype(float)
        raw = np.exp2(m) if log2_input else m
        return np.nansum(raw, axis=1)

    names = np.asarray(sample_names).astype(str)
    groups = np.asarray(groups).astype(str)
    order = _order_by_group(groups, names)
    colours = _colour_map(groups[order])
    panels = [("Before normalisation", totals(y_before))]
    if y_after is not None:
        panels.append(("After normalisation", totals(y_after)))
    width, tick_fs = _width_and_tick_size(len(names), font_size)
    figsize = figsize or (width, 4.6 * len(panels))

    with publication_style(font_size):
        fig, axes = plt.subplots(len(panels), 1, figsize=figsize, sharex=True, squeeze=False)
        for ax, (lab, tot) in zip(axes[:, 0], panels):
            exp = int(np.floor(np.log10(np.nanmax(tot))))
            ax.bar(np.arange(len(order)), tot[order] / 10 ** exp,
                   color=[colours[g] for g in groups[order]], width=0.8, edgecolor="none")
            ax.set_ylabel(f"Total intensity\n(×10$^{{{exp}}}$)")
            ax.set_title(lab, loc="left", fontsize=font_size + 1)
            cv = np.std(tot) / np.mean(tot) * 100
            ax.text(1.0, 1.02, f"CV {cv:.1f}%", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=font_size - 3, color="0.25")
        ax = axes[-1, 0]
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(names[order], rotation=90, fontsize=tick_fs)
        ax.set_xlim(-0.6, len(order) - 0.4)
        _group_legend(axes[0, 0], colours)
        if title:
            fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=font_size + 2)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Intensity distributions
# --------------------------------------------------------------------------------------
def plot_intensity_distributions(y, groups, sample_names=None, bins: int = 40,
                                 per_sample: bool = True, title: str | None = None,
                                 font_size: float = 16, ncols: int | None = None, figsize=None):
    """
    Histogram of the (normalised) log2 values of each group, one panel per group, with the
    density of every run / sample of the group as thin lines, the group median as a dashed
    line and the group's % missing values. Panels share both axes, so shifts between groups
    and runs that do not overlay their group-mates stand out.
    y: (n_runs or n_samples, n_features) log2 matrix; groups: label per row.
    """
    from scipy.stats import gaussian_kde
    Y = _np(y).astype(float)
    groups = np.asarray(groups).astype(str)
    levels = list(dict.fromkeys(groups))
    colours = _colour_map(levels)
    finite = Y[np.isfinite(Y)]
    lo, hi = np.quantile(finite, [0.001, 0.999])
    pad = 0.05 * (hi - lo)
    edges = np.linspace(lo - pad, hi + pad, bins + 1)
    grid = np.linspace(edges[0], edges[-1], 300)
    n = len(levels)
    ncols = ncols or min(4, n)
    nrows = int(np.ceil(n / ncols))
    figsize = figsize or (3.9 * ncols + 0.6, 3.1 * nrows + 1.2)

    with publication_style(font_size):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True,
                                 squeeze=False)
        ymax = 0
        for k, g in enumerate(levels):
            ax = axes[k // ncols, k % ncols]
            rows = np.where(groups == g)[0]
            vals = Y[rows][np.isfinite(Y[rows])]
            c = colours[g]
            hist, _ = np.histogram(vals, bins=edges, density=True)
            ax.bar(edges[:-1], hist, width=np.diff(edges), align="edge", color=c, alpha=0.35,
                   edgecolor="none")
            ymax = max(ymax, hist.max())
            if per_sample:
                for r in rows:
                    v = Y[r][np.isfinite(Y[r])]
                    if len(v) > 10:
                        dens = gaussian_kde(v)(grid)
                        ax.plot(grid, dens, color=c, lw=1.0, alpha=0.9)
                        ymax = max(ymax, dens.max())
            med = np.median(vals)
            ax.axvline(med, color="0.25", ls="--", lw=1.2)
            miss = np.isnan(Y[rows]).mean() * 100
            ax.text(0.97, 0.95, f"median {med:.1f}\n{miss:.0f}% missing", transform=ax.transAxes,
                    ha="right", va="top", fontsize=font_size - 5, color="0.25")
            ax.set_title(g, loc="left", fontsize=font_size - 2, fontweight="bold")
        for k in range(n, nrows * ncols):
            axes[k // ncols, k % ncols].axis("off")
        axes[0, 0].set_ylim(0, ymax * 1.15)
        for ax in axes[:, 0]:
            ax.set_ylabel("Density")
        for k in range(n):
            if k + ncols >= n:
                axes[k // ncols, k % ncols].set_xlabel("log$_2$ intensity")
                axes[k // ncols, k % ncols].xaxis.set_tick_params(labelbottom=True)
        fig.suptitle(title or "Normalised log$_2$ intensities per group", x=0.01, ha="left",
                     fontweight="bold", fontsize=font_size + 2)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_intensity_overlay(y, groups, title: str | None = None, font_size: float = 16,
                           figsize=(8, 5.5)):
    """Density of the pooled log2 values of each group, all groups in one panel."""
    from scipy.stats import gaussian_kde
    Y = _np(y).astype(float)
    groups = np.asarray(groups).astype(str)
    levels = list(dict.fromkeys(groups))
    colours = _colour_map(levels)
    finite = Y[np.isfinite(Y)]
    lo, hi = np.quantile(finite, [0.001, 0.999])
    grid = np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 300)
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        for g in levels:
            v = Y[groups == g]
            v = v[np.isfinite(v)]
            ax.plot(grid, gaussian_kde(v)(grid), color=colours[g], lw=2, label=g)
        ax.set_xlabel("log$_2$ intensity")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.set_title(title or "Normalised intensity distributions", loc="left")
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Missingness curve
# --------------------------------------------------------------------------------------
def plot_missingness_curve(miss, y, x, batch_ids, batch_names=None, n_bins: int = 15,
                           title: str | None = None, font_size: float = 16, figsize=(7.5, 5.5)):
    """
    Fitted P(missing) = sigmoid(w_b (u - r_b)) per batch against the group-mean abundance u,
    with the observed fraction missing in quantile bins of u (points, area ~ bin size).
    miss: fitted MissingParametersNR; y, x, batch_ids: the data it was fitted on.
    """
    import torch
    y = torch.as_tensor(y).double()
    x = torch.as_tensor(x).double()
    b = _np(batch_ids).astype(int)
    u, defined = miss.compute_group_means(y, x)
    u, defined = _np(u), _np(defined)
    m = np.isnan(_np(y))
    w = _np(miss.w).ravel()
    r = _np(miss.r).ravel()
    batches = [k for k in range(len(w)) if (b == k).any()]
    names = batch_names or {k: f"Batch {k}" for k in batches}

    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        lo, hi = np.inf, -np.inf
        for idx, k in enumerate(batches):
            c = OKABE_ITO[idx % len(OKABE_ITO)]
            rows = b == k
            uu = u[rows][defined[rows]]
            mm = m[rows][defined[rows]]
            edges = np.unique(np.quantile(uu, np.linspace(0, 1, n_bins + 1)))
            which = np.clip(np.searchsorted(edges, uu, side="right") - 1, 0, len(edges) - 2)
            cnt = np.bincount(which, minlength=len(edges) - 1)
            frac = np.bincount(which, weights=mm, minlength=len(edges) - 1) / np.maximum(cnt, 1)
            centre = np.bincount(which, weights=uu, minlength=len(edges) - 1) / np.maximum(cnt, 1)
            ok = cnt > 0
            ax.scatter(centre[ok], frac[ok], s=25 + 175 * cnt[ok] / cnt.max(), color=c,
                       alpha=0.75, edgecolor="white", linewidth=0.8, zorder=3)
            grid = np.linspace(uu.min(), uu.max(), 300)
            ax.plot(grid, 1 / (1 + np.exp(-w[k] * (grid - r[k]))), color=c,
                    label=f"{names[k]}: w = {w[k]:.2f}, r = {r[k]:.1f}")
            lo, hi = min(lo, uu.min()), max(hi, uu.max())
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Group mean log$_2$ intensity")
        ax.set_ylabel("P(missing)")
        ax.legend(loc="upper right" if w.mean() < 0 else "upper left")
        ax.set_title(title or "Missingness model", loc="left")
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Volcano
# --------------------------------------------------------------------------------------
def plot_volcano(results, alpha: float = 0.05, lfc_threshold: float = 1.0, n_labels: int = 10,
                 label_col: str | None = None, contrast_label: str | None = None,
                 font_size: float = 16, figsize=(7, 6.5)):
    """
    Volcano plot of one contrast: estimate (log2 FC) against -log10 p-value.
    Significant: padj < alpha and |log2 FC| >= lfc_threshold (red up, blue down).
    The dashed horizontal line marks the largest p-value with padj < alpha.
    The n_labels most significant proteins are labelled with the results index or label_col.
    """
    est = results["estimate"].to_numpy(float)
    p = results["pvalue"].to_numpy(float)
    padj = results["padj"].to_numpy(float)
    names = (results[label_col] if label_col else results.index.to_series()).astype(str).to_numpy()
    ok = np.isfinite(est) & np.isfinite(p)
    est, p, padj, names = est[ok], p[ok], padj[ok], names[ok]
    nlp = -np.log10(np.maximum(p, 1e-300))
    sig = (padj < alpha) & (np.abs(est) >= lfc_threshold)
    up, down = sig & (est > 0), sig & (est < 0)

    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(est[~sig], nlp[~sig], s=14, color=NS, alpha=0.7, edgecolor="none",
                   rasterized=True, label=f"NS ({(~sig).sum()})")
        ax.scatter(est[down], nlp[down], s=26, color=DOWN, alpha=0.9, edgecolor="none",
                   label=f"Down ({down.sum()})")
        ax.scatter(est[up], nlp[up], s=26, color=UP, alpha=0.9, edgecolor="none",
                   label=f"Up ({up.sum()})")
        if lfc_threshold > 0:
            for v in (-lfc_threshold, lfc_threshold):
                ax.axvline(v, color="0.4", ls="--", lw=1.2)
        if (padj < alpha).any():
            ax.axhline(-np.log10(p[padj < alpha].max()), color="0.4", ls="--", lw=1.2)
        xmax = np.nanmax(np.abs(est)) * 1.08
        ymax = nlp.max() * 1.08
        ax.set_ylim(0, ymax)

        if n_labels and sig.any():
            top = np.where(sig)[0][np.argsort(p[sig])][:n_labels]
            gap = 0.055 * ymax
            for side in (1, -1):
                idx = top[np.sign(est[top]) == side]
                if not len(idx):
                    continue
                idx = idx[np.argsort(-nlp[idx])]
                # stack labels in a column beside the points, keeping a minimum vertical gap
                ys = nlp[idx].copy()
                for k in range(1, len(ys)):
                    ys[k] = min(ys[k], ys[k - 1] - gap)
                if ys[-1] < gap:
                    ys += gap - ys[-1]
                ys = np.minimum(ys, ymax - 0.5 * gap)
                for k in range(len(ys) - 2, -1, -1):
                    ys[k] = max(ys[k], ys[k + 1] + gap) if ys[k] - ys[k + 1] < gap else ys[k]
                x_text = side * (np.abs(est[idx]).max() + 0.12 * xmax)
                for k, i_ in enumerate(idx):
                    ax.annotate(names[i_], (est[i_], nlp[i_]), xytext=(x_text, ys[k]),
                                textcoords="data", ha="left" if side > 0 else "right",
                                va="center", fontsize=font_size - 5,
                                arrowprops=dict(arrowstyle="-", color="0.45", lw=0.8,
                                                shrinkA=2, shrinkB=3))
                xmax = max(xmax, abs(x_text) * 1.55)
        ax.set_xlim(-xmax, xmax)

        ax.set_xlabel("log$_2$ fold change")
        ax.set_ylabel("−log$_{10}$ p-value")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, markerscale=1.6,
                  handletextpad=0.1, columnspacing=1.0, borderaxespad=0.2)
        ax.set_title(contrast_label or "Volcano plot", loc="left", pad=42)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Heatmap
# --------------------------------------------------------------------------------------
def plot_heatmap(y, results, groups, sample_names, top_n: int = 50, alpha: float | None = None,
                 label_col: str | None = None, cluster_rows: bool = True,
                 cluster_cols: bool = False, show_row_labels: bool | None = None,
                 clip: float = 2.5, title: str | None = None, font_size: float = 16,
                 figsize=None):
    """
    Heatmap of the top_n proteins by p-value (only padj < alpha if alpha is given), row
    z-scored across samples, missing values in light grey. Rows are clustered (average linkage,
    Euclidean distance on z-scores); columns are ordered by group unless cluster_cols=True.
    y: (n_samples, n_features) log2 matrix whose columns match the rows of results.
    """
    Y = _np(y).astype(float)
    p = results["pvalue"].to_numpy(float)
    cand = np.isfinite(p)
    if alpha is not None:
        cand &= results["padj"].to_numpy(float) < alpha
    feats = np.where(cand)[0][np.argsort(p[cand])][:top_n]
    if len(feats) < 2:
        raise ValueError("fewer than 2 proteins to show; relax alpha or increase top_n")
    names = (results[label_col] if label_col else results.index.to_series()).astype(str).to_numpy()
    M = Y[:, feats].T
    with np.errstate(invalid="ignore"):
        Zs = (M - np.nanmean(M, 1, keepdims=True)) / np.nanstd(M, 1, ddof=1, keepdims=True)
    Zs = np.clip(Zs, -clip, clip)
    groups = np.asarray(groups).astype(str)
    snames = np.asarray(sample_names).astype(str)
    colours = _colour_map(groups)

    filled = np.nan_to_num(Zs, nan=0.0)
    row_link = linkage(filled, "average") if cluster_rows else None
    rows = leaves_list(row_link) if cluster_rows else np.arange(len(feats))
    if cluster_cols:
        cols = leaves_list(linkage(filled.T, "average"))
    else:
        cols = _order_by_group(groups, snames)
    show_row_labels = len(feats) <= 60 if show_row_labels is None else show_row_labels
    width, col_fs = _width_and_tick_size(len(snames), font_size, per_item=0.42, extra=5.0,
                                         max_width=18.0)
    figsize = figsize or (max(7, width), max(6, 0.24 * len(feats) + 3))

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E6E6E6")
    with publication_style(font_size):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, width_ratios=[0.18 if cluster_rows else 0.001, 1, 0.035],
                              height_ratios=[0.035 * 50 / max(len(feats), 20), 1],
                              wspace=0.02, hspace=0.02)
        ax_bar = fig.add_subplot(gs[0, 1])
        ax_hm = fig.add_subplot(gs[1, 1])
        ax_cb = fig.add_subplot(gs[1, 2])
        if cluster_rows:
            ax_d = fig.add_subplot(gs[1, 0])
            dendrogram(row_link, orientation="left", ax=ax_d, no_labels=True,
                       color_threshold=0, above_threshold_color="0.3")
            ax_d.invert_yaxis()
            ax_d.axis("off")

        levels = list(colours)
        bar = np.array([[levels.index(g) for g in groups[cols]]])
        ax_bar.imshow(bar, aspect="auto", cmap=ListedColormap([colours[g] for g in levels]),
                      vmin=-0.5, vmax=len(levels) - 0.5, interpolation="nearest")
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        for s in ax_bar.spines.values():
            s.set_visible(False)

        im = ax_hm.imshow(np.ma.masked_invalid(Zs[rows][:, cols]), aspect="auto", cmap=cmap,
                          vmin=-clip, vmax=clip, interpolation="nearest")
        ax_hm.set_xticks(np.arange(len(cols)))
        ax_hm.set_xticklabels(snames[cols], rotation=90, fontsize=col_fs)
        if show_row_labels:
            ax_hm.yaxis.tick_right()
            ax_hm.set_yticks(np.arange(len(rows)))
            ax_hm.set_yticklabels(names[feats][rows], fontsize=max(font_size - 6, 7))
        else:
            ax_hm.set_yticks([])
        for s in ax_hm.spines.values():
            s.set_visible(False)
        ax_hm.tick_params(length=0)
        if show_row_labels:
            ax_cb.set_position(ax_cb.get_position().translated(0.09, 0))
        cb = fig.colorbar(im, cax=ax_cb)
        cb.set_label("Row z-score")
        cb.outline.set_visible(False)
        handles = [Patch(facecolor=c, label=g) for g, c in colours.items()]
        if np.isnan(Zs).any():
            handles.append(Patch(facecolor="#E6E6E6", label="Missing"))
        ncol = min(len(handles), 4)
        nrow = int(np.ceil(len(handles) / ncol))
        ax_bar.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.15),
                      ncol=ncol, handlelength=1.2, columnspacing=1.0, borderaxespad=0.0)
        ax_bar.set_title(title or f"Top {len(feats)} proteins", loc="left",
                         pad=(1.0 + 1.4 * nrow) * font_size)
    return fig


# --------------------------------------------------------------------------------------
# PTM: site change against protein change
# --------------------------------------------------------------------------------------
def plot_site_vs_protein(table, alpha: float = 0.05, n_labels: int = 0,
                         title: str | None = None, font_size: float = 16, figsize=(7, 8)):
    """
    Unadjusted site log2 FC against the protein's log2 FC, one point per adjustable site.
    Sites on the diagonal change only because their protein does; points are coloured when
    the protein-adjusted test is significant (padj_adjusted < alpha). Axis limits cover the
    central 99.5% of points (outliers are drawn at the edge as open circles).
    table: a PTMResults.tables entry.
    """
    t = table[table["adjustment"] == "paired"]
    x = t["protein_estimate"].to_numpy(float)
    y = t["estimate_unadjusted"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    t, x, y = t[ok], x[ok], y[ok]
    padj = t["padj_adjusted"].to_numpy(float)
    est = t["estimate_adjusted"].to_numpy(float)
    sig = np.nan_to_num(padj, nan=1.0) < alpha
    up, down = sig & (est > 0), sig & (est < 0)
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        lim = np.nanquantile(np.abs(np.r_[x, y]), 0.995) * 1.1 if len(x) else 1
        out = (np.abs(x) > lim) | (np.abs(y) > lim)
        x, y = np.clip(x, -lim, lim), np.clip(y, -lim, lim)
        ax.plot([-lim, lim], [-lim, lim], color="0.4", ls="--", lw=1.2,
                label="Protein-driven (site = protein)")
        ax.axhline(0, color="0.85", lw=1, zorder=0)
        ax.axvline(0, color="0.85", lw=1, zorder=0)
        ax.scatter(x[~sig & ~out], y[~sig & ~out], s=14, color=NS, alpha=0.7,
                   edgecolor="none", rasterized=True, label=f"Adjusted NS ({(~sig).sum()})")
        if out.any():
            ax.scatter(x[out], y[out], s=30, facecolor="none", edgecolor="0.3", lw=1.0,
                       label=f"Beyond axis range ({out.sum()})")
        ax.scatter(x[down], y[down], s=28, color=DOWN, edgecolor="none",
                   label=f"Stoichiometry down ({down.sum()})")
        ax.scatter(x[up], y[up], s=28, color=UP, edgecolor="none",
                   label=f"Stoichiometry up ({up.sum()})")
        if n_labels and sig.any():
            top = np.where(sig)[0][np.argsort(padj[sig])][:n_labels]
            for k, i in enumerate(top):
                ax.annotate(str(t.index[i]), (x[i], y[i]), xytext=(6, 6 - 12 * (k % 2)),
                            textcoords="offset points", fontsize=font_size - 6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Protein log$_2$ fold change")
        ax.set_ylabel("Site log$_2$ fold change (unadjusted)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, markerscale=1.5,
                  columnspacing=1.0, handletextpad=0.3)
        ax.set_title(title or "Site vs protein", loc="left")
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Enrichment and networks
# --------------------------------------------------------------------------------------
def plot_enrichment(table, top: int = 20, alpha: float = 0.05, title: str | None = None,
                    font_size: float = 16, max_label: int = 55, figsize=None):
    """
    Dot plot of the top gene sets: x = -log10 FDR, colour = direction (camera) or a single
    colour (ORA), dot area = number of genes. Dashed line at FDR = alpha.
    """
    t = table.head(top).iloc[::-1]
    if not len(t):
        raise ValueError("no gene sets to plot")
    x = -np.log10(t["padj"].clip(lower=1e-300).to_numpy(float))
    names = [n if len(n) <= max_label else n[: max_label - 1] + "…" for n in t["term"]]
    names = [n.replace("_", " ") for n in names]
    if "direction" in t:
        col = np.where(t["direction"] == "Up", UP, DOWN)
    else:
        col = np.full(len(t), "#7B3294")
    size = t["n_genes"].to_numpy(float)
    area = 30 + 270 * (size - size.min()) / max(size.max() - size.min(), 1)
    figsize = figsize or (10, 0.42 * len(t) + 2.2)
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(x, np.arange(len(t)), s=area, c=col, edgecolor="white", linewidth=0.8,
                   zorder=3)
        ax.axvline(-np.log10(alpha), color="0.4", ls="--", lw=1.2)
        ax.set_yticks(np.arange(len(t)))
        ax.set_yticklabels(names, fontsize=max(font_size - 4, 8))
        ax.set_xlabel("−log$_{10}$ FDR")
        xr = max(x.max(), -np.log10(alpha))
        ax.set_xlim(-0.04 * xr, xr * 1.12)
        handles = []
        if "direction" in t:
            handles += [Line2D([], [], marker="o", ls="", color=UP, markersize=10, label="Up"),
                        Line2D([], [], marker="o", ls="", color=DOWN, markersize=10,
                               label="Down")]
        for v in sorted({int(size.min()), int(np.median(size)), int(size.max())}):
            a = 30 + 270 * (v - size.min()) / max(size.max() - size.min(), 1)
            handles.append(Line2D([], [], marker="o", ls="", color="0.6",
                                  markersize=np.sqrt(a), label=f"{v} genes"))
        ax.legend(handles=handles, loc="lower right", labelspacing=1.0, borderpad=0.8)
        ax.set_title(title or "Gene-set enrichment", loc="left")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
    return fig


def plot_network(ppi_result: dict, title: str | None = None, max_nodes: int = 150,
                 font_size: float = 16, figsize=(9, 9), seed: int = 0):
    """
    Interactions among the hits (spring layout); node colour = log2 FC, size = -log10 padj.
    Requires networkx. Only the max_nodes most connected hits are drawn.
    """
    import networkx as nx
    from matplotlib.colors import TwoSlopeNorm
    nodes = ppi_result["nodes"]
    edges = ppi_result["edges"]
    if nodes is None or not len(edges):
        raise ValueError("no interactions among the hits")
    nodes = nodes[nodes["degree_among_hits"] > 0].sort_values("degree_among_hits",
                                                              ascending=False)
    nodes = nodes.head(max_nodes)
    keep = set(nodes["gene"])
    edges = edges[edges["gene_a"].isin(keep) & edges["gene_b"].isin(keep)]
    G = nx.Graph()
    G.add_nodes_from(nodes["gene"])
    G.add_edges_from(zip(edges["gene_a"], edges["gene_b"]))
    layout = nx.spring_layout(G, seed=seed, k=1.2 / np.sqrt(max(len(G), 1)))
    lfc = nodes.set_index("gene")["estimate"]
    padj = nodes.set_index("gene")["padj"].clip(lower=1e-300)
    lim = max(np.nanmax(np.abs(lfc.to_numpy())), 0.5)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    sizes = 60 + 40 * np.clip(-np.log10(padj.reindex(list(G.nodes)).to_numpy()), 0, 10)
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx_edges(G, layout, ax=ax, edge_color="0.75", width=0.8)
        sc = nx.draw_networkx_nodes(G, layout, ax=ax, node_size=sizes,
                                    node_color=lfc.reindex(list(G.nodes)).to_numpy(),
                                    cmap="RdBu_r", vmin=-lim, vmax=lim, edgecolors="white",
                                    linewidths=0.8)
        sc.set_norm(norm)
        if len(G) <= 60:
            nx.draw_networkx_labels(G, {k: (v[0], v[1] + 0.035) for k, v in layout.items()},
                                    ax=ax, font_size=max(font_size - 4, 8),
                                    font_family=plt.rcParams["font.sans-serif"][0])
        ax.axis("off")
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
        cb.set_label("log$_2$ fold change")
        cb.outline.set_visible(False)
        s = ppi_result["summary"].iloc[0]
        note = (f"{int(s['observed_edges'])} interactions among {int(s['hits_in_network'])} hits "
                f"(expected {s['expected_edges']:.1f}; p = {s['pvalue']:.2g})")
        ax.set_title((title or "Interactions among hits") + "\n" + note, loc="left",
                     fontsize=font_size)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Temporal profile clusters
# --------------------------------------------------------------------------------------
def plot_timecourse_clusters(tc, ncols: int | None = None, max_lines: int = 300,
                             title: str | None = None, font_size: float = 16, figsize=None,
                             series_colours: dict | None = None):
    """
    One panel per cluster: z-scored profiles of its proteins (thin lines; opacity = membership
    for fuzzy clustering) and the cluster centre (bold), one colour per series. Time points
    are equally spaced and labelled with their times.
    tc: TimecourseResult from timecourse.cluster_profiles.
    """
    from .timecourse import time_label
    Z = tc.z
    series = list(dict.fromkeys(Z.columns.get_level_values(0)))
    times = sorted(set(Z.columns.get_level_values(1)))
    xpos = {t: i for i, t in enumerate(times)}
    colours = series_colours or {s_: OKABE_ITO[i % len(OKABE_ITO)] for i, s_ in enumerate(series)}
    k = tc.k
    ncols = ncols or min(4, k)
    nrows = int(np.ceil(k / ncols))
    figsize = figsize or (4.2 * ncols + 0.5, 3.4 * nrows + 1.4)
    rng = np.random.default_rng(0)
    with publication_style(font_size):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True,
                                 squeeze=False)
        for c in range(1, k + 1):
            ax = axes[(c - 1) // ncols, (c - 1) % ncols]
            members = tc.assignments.index[tc.assignments["cluster"] == c]
            show = members if len(members) <= max_lines else \
                rng.choice(members, max_lines, replace=False)
            memb = tc.assignments.loc[show, "membership"].to_numpy()
            for s_ in series:
                cols = [col for col in Z.columns if col[0] == s_]
                x = [xpos[col[1]] for col in cols]
                vals = Z.loc[show, cols].to_numpy()
                for row, mb in zip(vals, memb):
                    ax.plot(x, row, color=colours[s_], lw=0.6, alpha=0.08 + 0.25 * mb)
                ax.plot(x, tc.centroids.loc[c, cols].to_numpy(), color=colours[s_], lw=3,
                        marker="o", ms=6, label=s_)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_title(f"Cluster {c} (n = {len(members)})", loc="left",
                         fontsize=font_size - 1, fontweight="bold")
        for c in range(k, nrows * ncols):
            axes[c // ncols, c % ncols].axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("z-score")
        for c in range(k):
            ax = axes[c // ncols, c % ncols]
            if c + ncols >= k:
                ax.set_xticks(range(len(times)))
                ax.set_xticklabels([time_label(t) for t in times])
                ax.xaxis.set_tick_params(labelbottom=True)
                ax.set_xlabel("Time")
        handles = [Line2D([], [], color=colours[s_], lw=3, marker="o", label=s_) for s_ in series]
        fig.legend(handles=handles, loc="upper right", ncol=len(series), frameon=False,
                   bbox_to_anchor=(0.99, 0.995))
        method = "fuzzy c-means" if tc.method == "fuzzy" else "k-means"
        fig.suptitle(title or f"Temporal profiles: {k} clusters ({method})", x=0.01,
                     ha="left", fontweight="bold", fontsize=font_size + 2)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# --------------------------------------------------------------------------------------
# All figures for one pipeline run
# --------------------------------------------------------------------------------------
def make_report_figures(out_dir: str, y_raw_runs, y_norm_runs, model, results: dict,
                        run_groups, run_names, sample_groups, sample_names, sample_batches,
                        miss=None, log2_input: bool = True, formats=("pdf", "png"),
                        dpi: int = 600, font_size: float = 16, volcano_kwargs=None,
                        heatmap_kwargs=None, contrast_vectors: dict | None = None,
                        heatmap_samples: str = "contrast", pca_factors: dict | None = None,
                        pca_shape=None, highlight=None, progress=None) -> dict:
    """
    Writes every figure to out_dir and returns {figure name: [paths]}.
    results: {contrast label: results DataFrame}; one volcano and one heatmap per contrast.
    Run-level panels use y_raw_runs / y_norm_runs (before collapsing); sample-level panels use
    model.y_input (after collapsing, before imputation).
    contrast_vectors: {label: weights over X columns}; with heatmap_samples="contrast" each
    heatmap shows only samples from groups with non-zero weight in its contrast.
    pca_factors: {name: labels per sample}; one extra PCA coloured by each factor
        (e.g. {"pulldown": [...], "treatment": [...], "time_min": [...]}).
    pca_shape: (name, labels per sample) used as marker shape in every PCA.
    highlight: bool per sample (e.g. QC outliers), ringed and labelled in the PCAs.
    progress: optional Progress, advanced once per figure.
    """
    import logging
    log = logging.getLogger("proteomix")
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    if progress is not None:
        progress.add(n_report_figures(results, miss, pca_factors))

    def save(fig, name):
        out[name] = save_figure(fig, os.path.join(out_dir, name), formats=formats, dpi=dpi)
        plt.close(fig)
        log.debug(f"   figure {name}")
        if progress is not None:
            progress.step(f"figure {name}")

    y_s = _np(model.y_input)
    shape_title, shape_labels = (pca_shape if pca_shape is not None else (None, None))
    if shape_labels is None:
        shape_title, shape_labels = "Batch", sample_batches
    save(plot_pca(y_s, sample_groups, shapes=shape_labels, shape_title=shape_title,
                  sample_names=sample_names, highlight=highlight, colour_title="Group",
                  font_size=font_size), "pca")
    for name, labels in (pca_factors or {}).items():
        slug = "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_")
        save(plot_pca(y_s, labels, shapes=shape_labels, shape_title=shape_title,
                      sample_names=sample_names, highlight=highlight,
                      colour_title=str(name).replace("_", " ").capitalize(),
                      title=f"PCA coloured by {str(name).replace('_', ' ')}",
                      ellipses=False, font_size=font_size), f"pca_by_{slug}")
    save(plot_protein_pca(y_s, sample_groups, results=results, font_size=font_size,
                          feature_names=next(iter(results.values())).index if results else None),
         "pca_proteins")
    raw_log2 = _np(y_raw_runs) if log2_input else np.log2(_np(y_raw_runs))
    save(plot_missingness_per_sample(raw_log2, run_groups, run_names, font_size=font_size),
         "missingness_per_sample")
    save(plot_total_intensity(raw_log2, run_groups, run_names, y_after=y_norm_runs,
                              font_size=font_size), "total_intensity")
    save(plot_intensity_distributions(y_norm_runs, run_groups, run_names,
                                      font_size=font_size), "intensity_histograms")
    save(plot_intensity_overlay(y_norm_runs, run_groups, font_size=font_size),
         "intensity_overlay")
    if miss is not None:
        save(plot_missingness_curve(miss, model.y_input, model.x_input, model.batch_input,
                                    font_size=font_size), "missingness_curve")
    for k, (label, res) in enumerate(results.items()):
        slug = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_") or f"c{k}"
        save(plot_volcano(res, contrast_label=label, font_size=font_size,
                          **(volcano_kwargs or {})), f"volcano_{slug}")
        cols = np.arange(len(sample_names))
        if heatmap_samples == "contrast" and contrast_vectors and label in contrast_vectors:
            used = np.abs(np.asarray(contrast_vectors[label])) > 0
            cols = np.where((_np(model.x_input)[:, used] != 0).any(1))[0]
        try:
            save(plot_heatmap(y_s[cols], res, np.asarray(sample_groups)[cols],
                              np.asarray(sample_names)[cols], font_size=font_size,
                              title=f"Top proteins: {label}", **(heatmap_kwargs or {})),
                 f"heatmap_{slug}")
        except ValueError as exc:
            log.warning(f"heatmap for {label} skipped: {exc}")
            if progress is not None:
                progress.step("heatmap skipped")
    return out


def n_report_figures(results, miss, pca_factors) -> int:
    """Number of figures make_report_figures will write (for progress bars)."""
    return 6 + len(pca_factors or {}) + (miss is not None) + 2 * len(results)
