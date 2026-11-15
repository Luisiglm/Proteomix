"""Planning tools: calibration, power, interpretation (planning.py, plan_cli.py)."""
import json

import numpy as np
import pytest

from proteomix.planning import SimulationSettings, calibrate, interpret, power_analysis


def test_settings_update_validates():
    st = SimulationSettings().update(n_per_group=6, prop_de=0.2)
    assert st.n_per_group == 6 and st.prop_de == 0.2
    with pytest.raises(ValueError):
        SimulationSettings().update(not_a_setting=1)


def test_simulation_options():
    st = SimulationSettings(n_features=2000, prop_de=0.5, frac_up=1.0, effect_dist="uniform",
                            logfc=2.0, mcar=0.2)
    sim = st.simulate(seed=1)
    lfc = sim["true_logfc"][sim["is_de"]]
    assert (lfc > 0).all() and lfc.min() >= 0.25 and lfc.max() <= 2.0
    assert sim["missing_rate"] > 0.2                          # MCAR floor adds missingness


@pytest.mark.slow
def test_power_increases_with_sample_size():
    st = SimulationSettings(n_features=600, prop_de=0.1)
    tab, tiers, req = power_analysis(st, effect=1.0, n_grid=(2, 4, 8), n_sims=2)
    assert tab["power"].is_monotonic_increasing
    assert tab["power"].iloc[-1] > tab["power"].iloc[0] + 0.3
    assert set(tiers["abundance_tier"]) == {"low", "mid", "high"}
    assert req is None or req in (2, 4, 8)


@pytest.mark.slow
def test_interpretation_precision_matches_fdr():
    st = SimulationSettings(n_features=800, prop_de=0.15, effect_dist="uniform", logfc=2.5)
    out = interpret(st, n_sims=3)["group 1 vs group 0"]
    o = out["overall"].iloc[0]
    assert o["precision_share_of_hits_real"] > 0.9              # FDR 5% controlled
    b = out["by_padj"].set_index("band")["share_real"]
    assert b["< 0.001"] > b["> 0.1"]
    ab = out["by_abundance"].set_index("abundance_tier")
    assert ab.loc["high", "detection_rate"] > ab.loc["low", "detection_rate"]


@pytest.mark.slow
def test_calibration_reproduces_missingness(fragpipe_files):
    from proteomix.fragpipe import build_inputs, read_fragpipe
    from proteomix.metadata import read_run_metadata
    data = read_fragpipe(str(fragpipe_files["protein"]))
    inp = build_inputs(data, read_run_metadata(str(fragpipe_files["metadata"])), key="run",
                       group_col="condition")
    st, rep = calibrate(inp)
    assert st.n_per_group == 3 and st.n_tech_reps == 2
    assert abs(rep["observed_missing_pct"] - rep["simulated_missing_pct"]) < 2
    assert 0.05 < st.tech_sd < 0.3


@pytest.mark.slow
def test_plan_cli_power_and_config(tmp_path):
    from proteomix.plan_cli import main
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"simulation": {"n_features": 400, "prop_de": 0.1},
                               "analysis": {"moderate": True}}))
    main(["power", "-o", str(tmp_path / "p"), "-t", "ip", "--config", str(cfg), "--fold-change", "2",
          "--n-range", "3,6", "--n-sims", "2", "--no-progress", "--no-figures"])
    saved = json.loads((tmp_path / "p" / "settings.json").read_text())
    assert saved["simulation"]["n_features"] == 400
    assert np.isclose(saved["simulation"]["logfc"], 1.0)
    assert saved["analysis"]["normalisation"] == "none"             # -t ip default
    assert (tmp_path / "p" / "power_by_n.csv").exists()
    with pytest.raises(SystemExit):
        main(["power", "-o", str(tmp_path / "q"), "--no-progress"])   # effect is required


@pytest.mark.slow
def test_interpret_ip_contrasts():
    st = SimulationSettings(n_features=600, prop_interactors=0.15, prop_responsive=0.4,
                            logfc=2.0)
    res = interpret(st, n_sims=2, experiment="ip")
    assert set(res) == {"IP vs IgG (Ctrl)", "Treatment in IP", "Treatment (IP - IgG)"}
    enr = res["IP vs IgG (Ctrl)"]["overall"].iloc[0]
    assert enr["detection_rate_real_changes_found"] > 0.8
    assert enr["precision_share_of_hits_real"] > 0.85


@pytest.mark.slow
def test_interpret_phospho_reports_protein_driven_sites():
    st = SimulationSettings(n_features=400, prop_de=0.15, logfc=1.5)
    res = interpret(st, n_sims=1, experiment="phospho")
    o = res["site stoichiometry (protein-adjusted)"]["overall"].iloc[0]
    assert "protein_driven_sites_called" in o and o["protein_driven_sites_called"] < 0.1
    assert o["precision_share_of_hits_real"] > 0.8


def test_gamma_effects_have_requested_mean_and_spread():
    from proteomix.simulation import draw_effects
    e = draw_effects(np.random.default_rng(0), 50000, 1.0, "gamma")
    assert abs(e.mean() - 1.0) < 0.02
    assert np.quantile(e, 0.1) < 0.4 and np.quantile(e, 0.9) > 1.8     # a real spread


@pytest.mark.slow
def test_interpret_cli_defaults_to_a_spread_of_effects(tmp_path):
    from proteomix.plan_cli import main
    main(["interpret", "-o", str(tmp_path / "i"), "--n-features", "500", "--n-sims", "1",
          "--no-progress", "--no-figures"])
    saved = json.loads((tmp_path / "i" / "settings.json").read_text())
    assert saved["simulation"]["effect_dist"] == "gamma"
    main(["interpret", "-o", str(tmp_path / "j"), "--effect-dist", "fixed", "--n-features",
          "500", "--n-sims", "1", "--no-progress", "--no-figures"])
    assert json.loads((tmp_path / "j" / "settings.json").read_text())["simulation"][
        "effect_dist"] == "fixed"
