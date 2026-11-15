"""
Simulated proteomics data with technical replicates, run-level size factors and
batch-specific abundance-dependent missingness.

Data-generating process (log2 scale), two groups:
    true sample abundance   y_sf  = mu_f + delta_f * [group == 1] + N(0, sigma_f^2)
    run value               y_rf  = y_s(r),f + N(0, tech_sd^2) + log2 size factor of run r
    missingness             P(miss) = sigmoid(w_b * (y_rf - r_b)),  b = batch of run r
Biological variances sigma_f^2 (var_dist):
    "inv_chi2"   scaled inverse chi-square (s0 = bio_s0, d0 = bio_d0): exactly limma's prior
    "lognormal"  log sigma_f ~ N(log bio_s0, lognormal_sd^2): a prior limma does not assume
    "trend"      inverse chi-square around a mean-variance trend, low abundance = noisier:
                 s0(mu) = bio_s0 * exp(trend_strength * (mean_abund - mu) / sd_abund)
Groups are balanced across batches. A fraction prop_de of features has delta_f = +/- logfc.
"""
import numpy as np
from scipy.special import expit

EFFECT_DISTS = ("fixed", "uniform", "normal", "gamma")


def draw_effects(rng, n, logfc, effect_dist="fixed", logfc_min=0.25, effect_shape=2.0):
    """
    |log2 FC| of n changed proteins:
        fixed    all equal to logfc
        uniform  Uniform(logfc_min, logfc)
        normal   |N(0, logfc^2)|
        gamma    Gamma(shape=effect_shape, mean=logfc): many modest changes and a long tail
                 of large ones, the typical shape of proteomics fold changes (shape 2:
                 mode logfc / 2, ~5% above 2.4 x logfc)
    """
    if effect_dist == "fixed":
        return np.full(n, float(logfc))
    if effect_dist == "uniform":
        return rng.uniform(logfc_min, logfc, n)
    if effect_dist == "normal":
        return np.abs(rng.normal(0, logfc, n))
    if effect_dist == "gamma":
        return rng.gamma(effect_shape, logfc / effect_shape, n)
    raise ValueError(f"effect_dist must be one of {EFFECT_DISTS}")


def simulate_dataset(n_features=2000, n_per_group=4, n_tech_reps=2, n_batches=2,
                     prop_de=0.2, logfc=1.0, w=-1.5, r=20.0, batch_r_shift=0.5,
                     mean_abund=22.0, sd_abund=2.0, bio_s0=0.3, bio_d0=10.0,
                     tech_sd=0.15, sf_sd=0.2, var_dist="inv_chi2", lognormal_sd=0.5,
                     trend_strength=0.35, effect_dist="fixed", logfc_min=0.25, frac_up=0.5,
                     mcar=0.0, group_loading_shift=0.0, effect_shape=2.0, seed=None) -> dict:
    """
    w, r: missingness slope and midpoint; scalars or one value per batch. With a scalar r,
    batch midpoints are spread by batch_r_shift around r. w < 0 means low abundances go missing.
    mcar: abundance-independent missingness floor, P(miss) = mcar + (1 - mcar) sigmoid(...).
    Effects of the DE proteins (effect_dist):
        "fixed"    |log2 FC| = logfc
        "uniform"  |log2 FC| ~ Uniform(logfc_min, logfc)
        "normal"   log2 FC ~ N(0, logfc^2) (many small effects, few large)
        "gamma"    |log2 FC| ~ Gamma(effect_shape, mean logfc) (see draw_effects)
    frac_up: fraction of DE proteins that go up (0.5 = symmetric).
    sf_sd: run-to-run loading variation (0 = perfectly balanced loading);
    group_loading_shift: systematic log2 loading difference of group 1 vs group 0.
    Returns numpy arrays: Y (runs x features, NaN = missing), X (runs x 2 one-hot),
    batch_ids, replicate_ids (biological sample per run), plus the truth.
    """
    rng = np.random.default_rng(seed)
    G = 2
    S = G * n_per_group
    group = np.repeat(np.arange(G), n_per_group)
    sample_batch = np.tile(np.arange(n_per_group) % n_batches, G)

    w_b = np.broadcast_to(np.asarray(w, dtype=float), (n_batches,)).copy()
    if np.ndim(r) == 0:
        r_b = r + batch_r_shift * (np.arange(n_batches) - (n_batches - 1) / 2)
    else:
        r_b = np.asarray(r, dtype=float)

    mu = rng.normal(mean_abund, sd_abund, n_features)
    if var_dist == "inv_chi2":
        bio_sd = np.sqrt(bio_s0 ** 2 * bio_d0 / rng.chisquare(bio_d0, n_features))
    elif var_dist == "lognormal":
        bio_sd = bio_s0 * np.exp(rng.normal(0.0, lognormal_sd, n_features))
    elif var_dist == "trend":
        s0 = bio_s0 * np.exp(trend_strength * (mean_abund - mu) / sd_abund)
        bio_sd = np.sqrt(s0 ** 2 * bio_d0 / rng.chisquare(bio_d0, n_features))
    else:
        raise ValueError("var_dist must be 'inv_chi2', 'lognormal' or 'trend'")
    n_de = int(round(prop_de * n_features))
    is_de = np.zeros(n_features, dtype=bool)
    is_de[rng.permutation(n_features)[:n_de]] = True
    true_logfc = np.zeros(n_features)
    sign = np.where(rng.random(n_de) < frac_up, 1.0, -1.0)
    true_logfc[is_de] = draw_effects(rng, n_de, logfc, effect_dist, logfc_min,
                                     effect_shape) * sign

    y_sample = mu + group[:, None] * true_logfc + rng.normal(size=(S, n_features)) * bio_sd

    replicate_ids = np.repeat(np.arange(S), n_tech_reps)
    N = len(replicate_ids)
    log_sf = rng.normal(0.0, sf_sd, N) + group_loading_shift * group[replicate_ids]
    y_run = (y_sample[replicate_ids] + rng.normal(0.0, tech_sd, (N, n_features))
             + log_sf[:, None])
    batch_ids = sample_batch[replicate_ids]
    p_miss = mcar + (1 - mcar) * expit(w_b[batch_ids][:, None] * (y_run - r_b[batch_ids][:, None]))
    missing = rng.random((N, n_features)) < p_miss

    return {
        "Y": np.where(missing, np.nan, y_run),
        "X": np.eye(G)[group[replicate_ids]],
        "batch_ids": batch_ids,
        "replicate_ids": replicate_ids,
        "is_de": is_de,
        "true_logfc": true_logfc,
        "w": w_b, "r": r_b, "log_sf": log_sf,
        "missing_rate": missing.mean(),
        "bio_sd": bio_sd,
        "mu": mu,
    }


# --------------------------------------------------------------------------------------
# Matched whole-proteome + PTM (e.g. phospho) data
# --------------------------------------------------------------------------------------
SITE_CATEGORIES = ("null", "protein-driven", "stoichiometry", "both")


def simulate_paired_ptm(n_proteins=1500, mean_extra_sites=1.0, n_per_group=4, n_tech_reps=2,
                        prop_protein_change=0.15, prop_site_change=0.10, logfc=1.0,
                        shared_sd=0.3, protein_sd=0.1, site_sd=0.2, stoich_mean=-3.5,
                        stoich_sd=1.5, tech_sd=0.15, loading_sd=0.2, enrichment_sd=0.4,
                        w_protein=-1.5, r_protein=20.0, w_site=-1.5, r_site=18.0,
                        site_mcar=0.1, prop_unquantified=0.1, mean_abund=22.0, sd_abund=2.0,
                        effect_dist="fixed", logfc_min=0.25, effect_shape=2.0,
                        seed=None) -> dict:
    """
    Two layers measured on the same biological samples (two groups, n_per_group each), each
    injected n_tech_reps times per layer.

    True log2 levels in biological sample s:
        protein  P[s, j] = mu_j + g_s * dP_j + b[s, j] + N(0, protein_sd^2)
        site     S[s, f] = mu_j + g_s * dP_j + b[s, j] + c_f + g_s * dS_f + N(0, site_sd^2)
    b[s, j] ~ N(0, shared_sd^2) is shared by a protein and its sites (what pairing cancels);
    c_f ~ N(stoich_mean, stoich_sd^2) is the site's log stoichiometry; dS_f its change.
    Runs add technical noise and a per-run factor: loading (protein runs) or enrichment
    efficiency (PTM runs, larger). Missingness: sigmoid(w (y - r)) per layer; sites also have
    an abundance-independent floor site_mcar (stochastic sampling).
    Each protein has 1 + Poisson(mean_extra_sites) sites. A fraction prop_unquantified of
    proteins is removed from the protein layer (their sites cannot be adjusted).

    Site categories (truth["category"]): null, protein-driven (dP != 0, dS = 0),
    stoichiometry (dS != 0, dP = 0), both. The protein-adjusted target is dS.
    """
    rng = np.random.default_rng(seed)
    G = 2
    S = G * n_per_group
    group = np.repeat(np.arange(G), n_per_group)
    mu = rng.normal(mean_abund, sd_abund, n_proteins)
    dP = np.zeros(n_proteins)
    chg = rng.random(n_proteins) < prop_protein_change
    dP[chg] = draw_effects(rng, chg.sum(), logfc, effect_dist, logfc_min,
                           effect_shape) * rng.choice([-1.0, 1.0], chg.sum())
    b = rng.normal(0, shared_sd, (S, n_proteins))
    P = mu + group[:, None] * dP + b + rng.normal(0, protein_sd, (S, n_proteins))

    n_sites = 1 + rng.poisson(mean_extra_sites, n_proteins)
    site_prot = np.repeat(np.arange(n_proteins), n_sites)
    F = len(site_prot)
    c = rng.normal(stoich_mean, stoich_sd, F)
    dS = np.zeros(F)
    sc = rng.random(F) < prop_site_change
    dS[sc] = draw_effects(rng, sc.sum(), logfc, effect_dist, logfc_min,
                          effect_shape) * rng.choice([-1.0, 1.0], sc.sum())
    Ssite = (mu[site_prot] + group[:, None] * dP[site_prot] + b[:, site_prot] + c
             + group[:, None] * dS + rng.normal(0, site_sd, (S, F)))

    rep = np.repeat(np.arange(S), n_tech_reps)
    N = len(rep)

    def runs(true, factor_sd, w, r, mcar):
        f = rng.normal(0, factor_sd, N)
        y = true[rep] + rng.normal(0, tech_sd, true[rep].shape) + f[:, None]
        p = mcar + (1 - mcar) / (1 + np.exp(-w * (y - r)))
        return np.where(rng.random(y.shape) < p, np.nan, y)

    Yp = runs(P, loading_sd, w_protein, r_protein, 0.0)
    Ys = runs(Ssite, enrichment_sd, w_site, r_site, site_mcar)
    quantified = rng.random(n_proteins) >= prop_unquantified
    keep_p = np.where(quantified)[0]

    category = np.where(dP[site_prot] != 0, np.where(dS != 0, "both", "protein-driven"),
                        np.where(dS != 0, "stoichiometry", "null"))
    return {
        "protein": {"Y": Yp[:, keep_p], "ids": [f"PROT{j:05d}" for j in keep_p]},
        "site": {"Y": Ys, "ids": [f"PROT{j:05d}_S{k}" for k, j in enumerate(site_prot)],
                 "protein_ids": [f"PROT{j:05d}" for j in site_prot]},
        "X": np.eye(G)[group[rep]], "replicate_ids": rep, "group": group,
        "truth": {"site_logfc": dS, "protein_logfc_of_site": dP[site_prot],
                  "site_mean": mu[site_prot] + c,
                  "category": category, "protein_quantified": quantified[site_prot],
                  "protein_logfc": dP[keep_p]},
        "missing_rate": {"protein": float(np.isnan(Yp).mean()), "site": float(np.isnan(Ys).mean())},
    }


def paired_ptm_inputs(sim: dict, group_names=("Control", "Treatment")):
    """ProteomixInputs (site, protein) for a simulate_paired_ptm result."""
    import pandas as pd
    import torch
    from .fragpipe import ProteomixInputs

    rep = sim["replicate_ids"]
    samples = [f"S{k + 1}" for k in range(rep.max() + 1)]
    gnames = list(group_names)

    def make(layer, prefix):
        Y = sim[layer]["Y"]
        ids = sim[layer]["ids"]
        runs = pd.DataFrame({"run": [f"{prefix}_{samples[s]}_r{i % 2 + 1}"
                                     for i, s in enumerate(rep)],
                             "sample": [samples[s] for s in rep],
                             "condition": [gnames[g] for g in sim["X"].argmax(1)]})
        prot = pd.DataFrame(index=ids)
        if layer == "site":
            prot["protein_key"] = sim["site"]["protein_ids"]
            prot["site"] = ids
        else:
            prot["Protein ID"] = ids
        return ProteomixInputs(
            Y=torch.tensor(Y), X=torch.tensor(sim["X"]),
            batch_ids=torch.zeros(len(rep), dtype=torch.long),
            replicate_ids=torch.tensor(rep, dtype=torch.long), run_names=runs["run"].tolist(),
            group_names=gnames, batch_names=["all"], feature_names=list(ids),
            sample_names=samples, runs=runs, proteins=prot)

    return make("site", "PH"), make("protein", "WP")


# --------------------------------------------------------------------------------------
# IP / IgG pull-downs with a treatment
# --------------------------------------------------------------------------------------
IP_GROUPS = ("IgG Ctrl", "IgG Treatment", "IP Ctrl", "IP Treatment")


def simulate_ip_dataset(n_features=2000, n_per_group=4, n_tech_reps=2, prop_interactors=0.1,
                        enrichment=3.0, prop_responsive=0.3, logfc=1.0, effect_dist="fixed",
                        logfc_min=0.25, frac_up=0.5, w=-1.5, r=20.0, mcar=0.0,
                        enrichment_dist="gamma", effect_shape=2.0,
                        mean_abund=21.0, sd_abund=2.0, bio_s0=0.3, bio_d0=10.0, tech_sd=0.15,
                        sf_sd=0.2, seed=None, **_ignored) -> dict:
    """
    Pull-down experiment: IgG and IP, each with Ctrl and Treatment, n_per_group biological
    samples per group, n_tech_reps injections each.
      background proteins   same level in IgG and IP (non-specific binders)
      interactors           (prop_interactors) enriched in every IP; log2 enrichment fixed
                            at `enrichment`, or (enrichment_dist="gamma") drawn with that
                            mean and shape 4, so interactors range from weak to strong
      responsive            a fraction prop_responsive of the interactors changes with the
                            treatment in the IP only (effect from effect_dist / logfc /
                            frac_up), i.e. a true IgG-corrected treatment effect
    Missingness, noise and loading as in simulate_dataset. Group order: IP_GROUPS.
    Truth: is_interactor, treatment_logfc (IP-specific treatment effect), mu.
    """
    rng = np.random.default_rng(seed)
    G = 4
    group = np.repeat(np.arange(G), n_per_group)
    is_ip = group >= 2
    is_treat = group % 2 == 1
    mu = rng.normal(mean_abund, sd_abund, n_features)
    bio_sd = np.sqrt(bio_s0 ** 2 * bio_d0 / rng.chisquare(bio_d0, n_features))
    inter = rng.random(n_features) < prop_interactors
    resp = inter & (rng.random(n_features) < prop_responsive)
    sign = np.where(rng.random(n_features) < frac_up, 1.0, -1.0)
    eff = draw_effects(rng, n_features, logfc, effect_dist, logfc_min, effect_shape)
    treat = np.where(resp, eff * sign, 0.0)
    enr = (np.full(n_features, float(enrichment)) if enrichment_dist == "fixed"
           else rng.gamma(4.0, enrichment / 4.0, n_features))
    enr = np.where(inter, enr, 0.0)
    y_sample = (mu + is_ip[:, None] * enr
                + (is_ip & is_treat)[:, None] * treat
                + rng.normal(size=(len(group), n_features)) * bio_sd)
    rep_ids = np.repeat(np.arange(len(group)), n_tech_reps)
    N = len(rep_ids)
    y_run = (y_sample[rep_ids] + rng.normal(0, tech_sd, (N, n_features))
             + rng.normal(0, sf_sd, N)[:, None])
    p_miss = mcar + (1 - mcar) * expit(w * (y_run - r))
    missing = rng.random(y_run.shape) < p_miss
    return {"Y": np.where(missing, np.nan, y_run), "X": np.eye(G)[group[rep_ids]],
            "batch_ids": np.zeros(N, dtype=int), "replicate_ids": rep_ids,
            "group_names": list(IP_GROUPS), "missing_rate": float(missing.mean()),
            "is_interactor": inter, "enrichment_logfc": enr, "treatment_logfc": treat,
            "mu": mu}
