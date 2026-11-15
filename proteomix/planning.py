"""
Study planning and interpretation with simulations that look like the user's data.

calibrate(inputs)            estimates simulation settings from a real dataset: protein
                             abundance distribution, biological and technical variance,
                             missingness curve, run loading variation, design
power_analysis(...)          empirical power for a chosen effect size over a range of sample
                             sizes; the smallest n per group reaching the target power
interpret(...)               runs the user's analysis settings on simulated data of their
                             design and reports how trustworthy hits and non-hits are:
                             share of hits that are real (by adjusted p-value, fold change and
                             abundance), detection rate by true effect size and abundance, how
                             often a non-hit hides a real change, and fold-change bias

Power is always computed for an effect size the user chooses (biologically relevant or
from earlier studies), never from the effects observed in the data: "observed power" adds
nothing to the p-values.

Experiment types (the same simulators as the package's benchmarks):
    proteome  two groups (simulate_dataset); contrast group 1 vs group 0
    ip        IgG and IP, each Ctrl / Treatment (simulate_ip_dataset): interactors enriched
              in the IP, some responding to the treatment in the IP only; evaluated
              contrasts: IP vs IgG (Ctrl), treatment in IP, IgG-corrected treatment
    phospho   matched sites and proteins (simulate_paired_ptm), analysed with the paired
              protein-offset model; the target is the change in site stoichiometry, and
              sites that change only because their protein does count as false if called

The simulations follow the package's own model (log-normal abundances, abundance-dependent
missingness). Results are a guide to what the analysis can and cannot see in data like
yours; real data can deviate (outliers, correlated proteins, non-sigmoid missingness).
"""
import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch

_log = logging.getLogger("proteomix")
TIERS = ("low", "mid", "high")


@dataclass
class SimulationSettings:
    n_features: int = 2000
    n_per_group: int = 4
    n_tech_reps: int = 1
    n_batches: int = 1
    prop_de: float = 0.1
    logfc: float = 1.0
    effect_dist: str = "fixed"
    logfc_min: float = 0.25
    frac_up: float = 0.5
    w: float = -1.5
    r: float = 20.0
    mcar: float = 0.0
    mean_abund: float = 22.0
    sd_abund: float = 2.0
    bio_s0: float = 0.3
    bio_d0: float = 10.0
    tech_sd: float = 0.15
    sf_sd: float = 0.2
    group_loading_shift: float = 0.0
    effect_shape: float = 2.0
    # IP / IgG experiments
    enrichment_dist: str = "gamma"
    prop_interactors: float = 0.1
    enrichment: float = 3.0
    prop_responsive: float = 0.3
    # phospho (sites with matched proteins)
    mean_extra_sites: float = 1.0
    prop_protein_change: float = 0.15
    shared_sd: float = 0.3
    site_sd: float = 0.2
    stoich_mean: float = -3.5
    stoich_sd: float = 1.5
    enrichment_sd: float = 0.4
    site_mcar: float = 0.1
    site_r_offset: float = -2.0
    prop_unquantified: float = 0.1

    def simulate(self, seed=None, **override) -> dict:
        from .simulation import simulate_dataset
        kw = {k: v for k, v in {**asdict(self), **override}.items()
              if k in simulate_dataset.__code__.co_varnames}
        return simulate_dataset(batch_r_shift=0.0, seed=seed, **kw)

    def simulate_ip(self, seed=None, **override) -> dict:
        from .simulation import simulate_ip_dataset
        return simulate_ip_dataset(seed=seed, **{**asdict(self), **override})

    def simulate_phospho(self, seed=None, **override) -> dict:
        from .simulation import simulate_paired_ptm
        s = {**asdict(self), **override}
        return simulate_paired_ptm(
            n_proteins=s["n_features"], mean_extra_sites=s["mean_extra_sites"],
            n_per_group=s["n_per_group"], n_tech_reps=s["n_tech_reps"],
            prop_protein_change=s["prop_protein_change"], prop_site_change=s["prop_de"],
            logfc=s["logfc"], shared_sd=s["shared_sd"], site_sd=s["site_sd"],
            stoich_mean=s["stoich_mean"], stoich_sd=s["stoich_sd"], tech_sd=s["tech_sd"],
            loading_sd=s["sf_sd"], enrichment_sd=s["enrichment_sd"], w_protein=s["w"],
            r_protein=s["r"], w_site=s["w"], r_site=s["r"] + s["site_r_offset"],
            site_mcar=s["site_mcar"], prop_unquantified=s["prop_unquantified"],
            mean_abund=s["mean_abund"], sd_abund=s["sd_abund"], effect_dist=s["effect_dist"],
            logfc_min=s["logfc_min"], effect_shape=s["effect_shape"], seed=seed)

    def update(self, **kw):
        unknown = set(kw) - set(asdict(self))
        if unknown:
            raise ValueError(f"unknown simulation settings {sorted(unknown)}; valid: "
                             f"{sorted(asdict(self))}")
        for k, v in kw.items():
            setattr(self, k, type(getattr(self, k))(v))
        return self


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------
def calibrate(inputs, normalisation: str = "median_ratio", max_features: int = 2000,
              contrast=None) -> tuple:
    """
    Simulation settings estimated from a ProteomixInputs dataset. Returns (settings, report),
    where report compares the observed and simulated missingness and variance.
    """
    from .pipeline import run_differential_abundance
    from .qc_extended import replicate_pairs
    from .normalisation import normalise

    p = inputs.X.shape[1]
    contrast = contrast or ((1, 0) if p >= 2 else [1.0])
    res, model, miss, log_sf = run_differential_abundance(
        **inputs.pipeline_kwargs(), contrast=contrast, normalisation=normalisation,
        moderate=True)
    prior = model.var_prior
    s0_2 = float(np.median(prior["s0_2"]))
    d0 = prior["d0_global"]
    d0 = 50.0 if not np.isfinite(d0) else float(np.clip(d0, 2.0, 50.0))

    codes = inputs.replicate_ids.numpy()
    reps = int(np.median(np.bincount(codes)))
    tech_sd = 0.0
    if reps >= 2:
        Yn, _ = normalise(inputs.Y.double(), normalisation)
        rs = np.asarray(inputs.sample_names)[codes]
        pairs = replicate_pairs(Yn, rs, inputs.run_names)
        if len(pairs):
            tech_sd = float(np.median(pairs["sd_log2_diff"]) / np.sqrt(2))
    bio_s0 = float(np.sqrt(max(s0_2 - tech_sd ** 2 / reps, 0.01 * s0_2)))
    # loading variation from median-of-ratios offsets, whatever normalisation is analysed
    _, sf_mr = normalise(inputs.Y.double(), "median_ratio")
    sf_sd = float(np.std(sf_mr.numpy()))
    am = model.amean.numpy()[model.fitted.numpy()]
    group_sizes = np.bincount(model.x_input.numpy().argmax(1))
    st = SimulationSettings(
        n_features=int(min(len(inputs.feature_names), max_features)),
        n_per_group=int(np.median(group_sizes[group_sizes > 0])), n_tech_reps=reps,
        n_batches=1, w=float(miss.w.mean()), r=float(miss.r.mean()),
        mean_abund=float(np.mean(am)), sd_abund=float(np.std(am)), bio_s0=bio_s0,
        bio_d0=d0, tech_sd=tech_sd,
        sf_sd=sf_sd)
    # Proteins too sparse to fit are left out of the abundance estimate, which biases it
    # upwards; shift the abundance distribution until the simulated share of missing values
    # matches the observed one (bisection).
    obs_missing = float(torch.isnan(inputs.Y).double().mean())
    lo, hi = st.mean_abund - 6.0, st.mean_abund + 3.0
    for _ in range(14):
        mid = (lo + hi) / 2
        m = st.simulate(seed=0, prop_de=0.0, mean_abund=mid)["missing_rate"]
        lo, hi = (lo, mid) if m < obs_missing else (mid, hi)
    st.mean_abund = float((lo + hi) / 2)
    sim = st.simulate(seed=0, prop_de=0.0)
    report = {"observed_missing_pct": 100 * obs_missing,
              "simulated_missing_pct": 100 * float(sim["missing_rate"]),
              "prior_df_d0": prior["d0_global"], "prior_sd_s0": float(np.sqrt(s0_2)),
              "technical_sd": tech_sd, "proteins_in_data": len(inputs.feature_names)}
    return st, report


# --------------------------------------------------------------------------------------
# Running the analysis on simulated data
# --------------------------------------------------------------------------------------
EXPERIMENTS = ("proteome", "ip", "phospho")
IP_CONTRASTS = {
    "IP vs IgG (Ctrl)": {"IP Ctrl": 1, "IgG Ctrl": -1},
    "Treatment in IP": {"IP Treatment": 1, "IP Ctrl": -1},
    "Treatment (IP - IgG)": {"IP Treatment": 1, "IP Ctrl": -1, "IgG Treatment": -1,
                             "IgG Ctrl": 1},
}


def _tiers(mu):
    q = np.quantile(mu, [1 / 3, 2 / 3])
    return np.where(mu < q[0], "low", np.where(mu < q[1], "mid", "high"))


def _rows(i, contrast, true, real, tier, est, p, padj, tested, extra=None):
    d = pd.DataFrame({"sim": i, "contrast": contrast, "true_logfc": true, "is_de": real,
                      "abundance_tier": tier, "estimate": est, "pvalue": p, "padj": padj,
                      "tested": tested})
    if extra:
        for k, v in extra.items():
            d[k] = v
    return d


def _run_one(settings, analysis, experiment, seed, override):
    """One simulated dataset of the experiment type, analysed with the given settings."""
    from .pipeline import run_differential_abundance
    if experiment == "proteome":
        sim = settings.simulate(seed=seed, **override)
        res, *_ = run_differential_abundance(
            torch.tensor(sim["Y"]), torch.tensor(sim["X"]), torch.tensor(sim["batch_ids"]),
            (1, 0), replicate_ids=torch.tensor(sim["replicate_ids"]), **analysis)
        return [_rows(seed, "group 1 vs group 0", sim["true_logfc"], sim["is_de"],
                      _tiers(sim["mu"]), res["estimate"].to_numpy(), res["pvalue"].to_numpy(),
                      res["padj"].to_numpy(), res["fitted"].to_numpy())]
    if experiment == "ip":
        sim = settings.simulate_ip(seed=seed, **override)
        res, *_ = run_differential_abundance(
            torch.tensor(sim["Y"]), torch.tensor(sim["X"]), torch.tensor(sim["batch_ids"]),
            IP_CONTRASTS, replicate_ids=torch.tensor(sim["replicate_ids"]),
            group_names=sim["group_names"], **analysis)
        tier = _tiers(sim["mu"])
        out = []
        for label, r in res.items():
            if label.startswith("IP vs IgG"):
                true = sim["enrichment_logfc"]
            else:
                true = sim["treatment_logfc"]
            out.append(_rows(seed, label, true, true != 0, tier, r["estimate"].to_numpy(),
                             r["pvalue"].to_numpy(), r["padj"].to_numpy(),
                             r["fitted"].to_numpy()))
        return out
    if experiment == "phospho":
        from .ptm import run_paired_ptm
        from .simulation import paired_ptm_inputs
        sim = settings.simulate_phospho(seed=seed, **override)
        si, pi = paired_ptm_inputs(sim)
        kw = {k: v for k, v in analysis.items() if k != "normalisation"}
        norm = analysis.get("normalisation", "median_ratio")
        res = run_paired_ptm(si, pi, {"c": ("Treatment", "Control")}, site_normalisation=norm,
                             protein_normalisation=norm, **kw)
        t = res.tables["c"].reindex(si.feature_names)
        dS = sim["truth"]["site_logfc"]
        tested = (t["adjustment"] == "paired").to_numpy() & t["pvalue_adjusted"].notna().to_numpy()
        return [_rows(seed, "site stoichiometry (protein-adjusted)", dS, dS != 0,
                      _tiers(sim["truth"]["site_mean"]), t["estimate_adjusted"].to_numpy(),
                      t["pvalue_adjusted"].to_numpy(), t["padj_adjusted"].to_numpy(), tested,
                      {"protein_driven": sim["truth"]["category"] == "protein-driven"})]
    raise ValueError(f"experiment must be one of {EXPERIMENTS}")


def _collect(settings, analysis, n_sims, seed, progress=None, experiment="proteome",
             **override):
    rows = []
    for i in range(n_sims):
        rows += _run_one(settings, analysis, experiment, seed + i, override)
        if progress is not None:
            progress.step(f"simulation {i + 1}/{n_sims}")
    d = pd.concat(rows, ignore_index=True)
    d["sim"] = d["sim"] - seed
    return d


# --------------------------------------------------------------------------------------
# Power and sample size
# --------------------------------------------------------------------------------------
DEFAULT_CONTRAST = {"proteome": "group 1 vs group 0", "ip": "Treatment (IP - IgG)",
                    "phospho": "site stoichiometry (protein-adjusted)"}


def power_analysis(settings: SimulationSettings, effect: float, target: float = 0.8,
                   alpha: float = 0.05, n_grid=(2, 3, 4, 5, 6, 8, 10, 12), n_sims: int = 5,
                   analysis: dict | None = None, seed: int = 0, progress=None,
                   experiment: str = "proteome", contrast: str | None = None) -> tuple:
    """
    Power to detect |log2 FC| = effect at FDR alpha in the chosen contrast, for each n per
    group in n_grid (changed proteins / sites that cannot be tested count as missed).
    ip: effect is the treatment effect of responsive interactors; phospho: the change in
    site stoichiometry. Returns (table by n, table by n x abundance tier, required n or None).
    """
    analysis = analysis or {}
    contrast = contrast or DEFAULT_CONTRAST[experiment]
    rows, tier_rows = [], []
    for n in n_grid:
        d = _collect(settings, analysis, n_sims, seed, progress, experiment,
                     n_per_group=int(n), logfc=float(effect), effect_dist="fixed")
        if contrast not in set(d["contrast"]):
            raise ValueError(f"contrast {contrast!r} not available; choose from "
                             f"{sorted(set(d['contrast']))}")
        d = d[d["contrast"] == contrast]
        de = d[d["is_de"]]
        if de.empty:
            raise ValueError("the simulation contains no changed proteins for this contrast "
                             "(effect size 0, or no responsive / changed proteins); increase "
                             "the effect or the proportion of changed proteins")
        hit = de["padj"].fillna(1) < alpha
        per_sim = hit.groupby(de["sim"]).mean()
        rows.append({"n_per_group": n, "power": per_sim.mean(),
                     "power_sd": per_sim.std(ddof=1) if n_sims > 1 else np.nan,
                     "power_testable": (hit & de["tested"]).sum() / max(de["tested"].sum(), 1),
                     "fdp": _fdp(d, alpha)})
        for t in TIERS:
            m = de["abundance_tier"] == t
            tier_rows.append({"n_per_group": n, "abundance_tier": t,
                              "power": hit[m].mean() if m.any() else np.nan})
    tab = pd.DataFrame(rows)
    ok = tab[tab["power"] >= target]
    required = int(ok["n_per_group"].min()) if len(ok) else None
    return tab, pd.DataFrame(tier_rows), required


def _fdp(d, alpha):
    hit = d["padj"].fillna(1) < alpha
    per = (hit & ~d["is_de"]).groupby(d["sim"]).sum() / hit.groupby(d["sim"]).sum().clip(lower=1)
    return float(per.mean())


# --------------------------------------------------------------------------------------
# Interpretation aid
# --------------------------------------------------------------------------------------
PADJ_BINS = [0, 0.001, 0.01, 0.05, 0.1, 1.0000001]
PADJ_LABELS = ["< 0.001", "0.001-0.01", "0.01-0.05", "0.05-0.1", "> 0.1"]
FC_BINS = [0, 0.25, 0.5, 1, 1.5, 2, 3, np.inf]
FC_LABELS = ["< 0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "> 3"]


def _summaries(d, alpha, lfc_threshold):
    d = d.copy()
    d["hit"] = (d["padj"].fillna(1) < alpha) & (d["estimate"].abs() >= lfc_threshold)
    real = d["is_de"]
    n_hits = d.groupby("sim")["hit"].sum()
    overall = {
        "mean_hits": n_hits.mean(),
        "precision_share_of_hits_real": (d["hit"] & real).sum() / max(d["hit"].sum(), 1),
        "detection_rate_real_changes_found": (d["hit"] & real).sum() / max(real.sum(), 1),
        "false_hits_per_analysis": (d["hit"] & ~real).groupby(d["sim"]).sum().mean(),
        "missed_real_per_analysis": (~d["hit"] & real).groupby(d["sim"]).sum().mean(),
        "untestable_pct": 100 * (~d["tested"]).mean()}
    if "protein_driven" in d:
        pdv = d["protein_driven"].astype(bool)
        overall["protein_driven_sites_called"] = (d["hit"] & pdv).sum() / max(pdv.sum(), 1)
    out = {"overall": pd.DataFrame([overall])}
    band = pd.cut(d["padj"].fillna(1), PADJ_BINS, labels=PADJ_LABELS, right=False)
    out["by_padj"] = (d.assign(band=band).groupby("band", observed=False)
                      .agg(proteins=("is_de", "size"), share_real=("is_de", "mean"))
                      .reset_index())
    h = d[d["hit"]]
    if len(h):
        fc = pd.cut(h["estimate"].abs(), FC_BINS, labels=FC_LABELS, right=False)
        out["by_fold_change"] = (h.assign(fc=fc).groupby("fc", observed=False)
                                 .agg(hits=("is_de", "size"), share_real=("is_de", "mean"))
                                 .reset_index().rename(columns={"fc": "estimated_abs_log2fc"}))
    else:
        out["by_fold_change"] = pd.DataFrame()
    rows = []
    for t in TIERS:
        m = d["abundance_tier"] == t
        hm, rm = d["hit"] & m, real & m
        err = (d["estimate"] - d["true_logfc"])[rm & d["tested"]]
        sgn = np.sign(d["true_logfc"][rm & d["tested"]])
        rows.append({"abundance_tier": t,
                     "precision": (hm & real).sum() / max(hm.sum(), 1),
                     "detection_rate": (hm & real).sum() / max(rm.sum(), 1),
                     "non_hits_that_are_real_changes": (~d["hit"] & rm).sum()
                     / max((~d["hit"] & m).sum(), 1),
                     "untestable_pct": 100 * (~d["tested"][m]).mean(),
                     "fold_change_bias": float(np.mean(err * sgn)) if len(err) else np.nan})
    out["by_abundance"] = pd.DataFrame(rows)
    de = d[real].copy()
    if len(de) and de["true_logfc"].abs().nunique() > 1:
        de["effect"] = pd.cut(de["true_logfc"].abs(), FC_BINS, labels=FC_LABELS, right=False)
    else:
        de["effect"] = f"{de['true_logfc'].abs().iloc[0]:g}" if len(de) else "none"
    out["by_effect"] = (de.groupby(["effect", "abundance_tier"], observed=False)["hit"]
                        .agg(["mean", "size"]).reset_index()
                        .rename(columns={"mean": "detection_rate", "size": "proteins"}))
    out["proteins"] = d
    return out


def interpret(settings: SimulationSettings, analysis: dict | None = None, n_sims: int = 5,
              alpha: float = 0.05, lfc_threshold: float = 0.0, seed: int = 0,
              progress=None, experiment: str = "proteome") -> dict:
    """
    Runs the analysis settings on data simulated for the experiment type. A result is a "hit"
    when padj < alpha and |estimate| >= lfc_threshold. Returns {contrast: tables}, where the
    tables are:
      overall        hits, share of hits that are real (precision), detection, false hits;
                     for phospho also the share of protein-driven sites called
      by_padj        share real among results in each adjusted p-value band
      by_fold_change share real among hits, by estimated |log2 FC|
      by_abundance   precision, detection, missed real changes among non-hits, bias, by tier
      by_effect      detection rate by true |log2 FC| and abundance tier
      proteins       per-protein (or per-site) results of all simulations
    """
    d = _collect(settings, analysis or {}, n_sims, seed, progress, experiment)
    return {c: _summaries(g, alpha, lfc_threshold) for c, g in d.groupby("contrast", sort=False)}


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def plot_power_curve(tab, tiers, effect, target=0.8, required=None, font_size: float = 16,
                     figsize=(8, 5.5)):
    import matplotlib.pyplot as plt
    from .figures import OKABE_ITO, publication_style
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        ax.errorbar(tab["n_per_group"], tab["power"], yerr=tab["power_sd"], color="0.1",
                    marker="o", lw=2.5, capsize=4, label="all proteins")
        for k, t in enumerate(TIERS):
            s = tiers[tiers["abundance_tier"] == t]
            ax.plot(s["n_per_group"], s["power"], marker=".", lw=1.5, ls="--",
                    color=OKABE_ITO[k], label=f"{t} abundance")
        ax.axhline(target, color="#B2182B", ls=":", lw=1.5)
        if required:
            ax.axvline(required, color="#B2182B", ls=":", lw=1.5)
            ax.text(required, 0.03, f"  n = {required}", color="#B2182B",
                    fontsize=font_size - 2)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Biological samples per group")
        ax.set_ylabel("Power (FDR-controlled)")
        ax.legend(loc="lower right", fontsize=font_size - 3)
        ax.set_title(f"Power to detect |log$_2$ FC| = {effect:g} ({2 ** effect:.2g}-fold)",
                     loc="left")
        fig.tight_layout()
    return fig


def plot_interpretation(out, alpha=0.05, font_size: float = 16, figsize=(22, 5.5),
                        title: str | None = None):
    import matplotlib.pyplot as plt
    from .figures import OKABE_ITO, publication_style
    d = out["proteins"]
    with publication_style(font_size):
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        ax = axes[0]
        eff = d.loc[d["is_de"], "true_logfc"].abs().to_numpy()
        if len(eff):
            ax.hist(eff, bins=np.linspace(0, np.quantile(eff, 0.99) * 1.05, 30),
                    color=OKABE_ITO[5], edgecolor="white")
            med = np.median(eff)
            ax.axvline(med, color="0.2", ls="--", lw=1.2)
            ax.text(med, ax.get_ylim()[1] * 0.92, f" median {med:.2f}\n ({2 ** med:.2g}-fold)",
                    fontsize=font_size - 5, va="top")
        ax.set_xlabel("True |log$_2$ FC|")
        ax.set_ylabel("Changed proteins")
        ax.set_title("Simulated changes", loc="left", fontsize=font_size)
        b = out["by_padj"]
        axes[1].bar(range(len(b)), b["share_real"], color=OKABE_ITO[0])
        axes[1].set_xticks(range(len(b)))
        axes[1].set_xticklabels(b["band"].astype(str), rotation=30, ha="right")
        axes[1].set_ylim(0, 1)
        axes[1].set_xlabel("Adjusted p-value")
        axes[1].set_ylabel("Share that are real changes")
        axes[1].set_title("How often a result is real", loc="left", fontsize=font_size)
        e = out["by_effect"].copy()
        order = {lab: i for i, lab in enumerate(FC_LABELS)}
        e["x"] = e["effect"].astype(str).map(order)
        used = set()
        for k, t in enumerate(TIERS):
            s_ = e[(e["abundance_tier"] == t) & (e["proteins"] >= 5)].dropna(subset=["x"])
            s_ = s_.sort_values("x")
            used |= set(s_["x"].astype(int))
            axes[2].plot(s_["x"], s_["detection_rate"], marker="o", lw=2,
                         color=OKABE_ITO[k], label=f"{t} abundance")
        ticks = sorted(used)
        axes[2].set_xticks(ticks)
        axes[2].set_xticklabels([FC_LABELS[i] for i in ticks], rotation=30, ha="right")
        axes[2].set_ylim(0, 1.02)
        axes[2].set_xlabel("True |log$_2$ FC|")
        axes[2].set_ylabel("Detection rate")
        axes[2].legend(fontsize=font_size - 4)
        axes[2].set_title("How often a real change is found", loc="left", fontsize=font_size)
        de = d[d["is_de"] & d["tested"]]
        axes[3].scatter(de["true_logfc"], de["estimate"], s=8, alpha=0.3, color=OKABE_ITO[0],
                        edgecolor="none", rasterized=True)
        lim = np.nanmax(np.abs(np.r_[de["true_logfc"], de["estimate"]])) * 1.05 if len(de) else 1
        axes[3].plot([-lim, lim], [-lim, lim], color="0.5", ls="--")
        axes[3].set_xlim(-lim, lim)
        axes[3].set_ylim(-lim, lim)
        axes[3].set_xlabel("True log$_2$ FC")
        axes[3].set_ylabel("Estimated log$_2$ FC")
        axes[3].set_title("Fold-change accuracy", loc="left", fontsize=font_size)
        fig.suptitle(title or "What your settings can detect in data like yours (simulated)",
                     x=0.01, ha="left", fontweight="bold", fontsize=font_size + 1)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig
