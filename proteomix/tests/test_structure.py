"""Structure module: readers, site mapping, distances, figures and PyMOL script."""
import numpy as np
import pandas as pd
import pytest

from proteomix import structure as st

SPECIAL = {10: "SER", 30: "TYR", 50: "SER", 75: "THR", 100: "SER"}
SIDE = {"SER": "OG", "THR": "OG1", "TYR": "OH"}


def _atoms():
    rows = []
    for i in range(1, 121):
        res = SPECIAL.get(i, "ALA")
        t = i * 0.6
        ca = np.array([8 * np.cos(t), 8 * np.sin(t), 1.5 * i])
        names = ["N", "CA", "C", "O", "CB"] + ([SIDE[res]] if res in SIDE else [])
        for k, nm in enumerate(names):
            rows.append((i, res, nm, ca + 0.4 * k, 90.0 if i <= 80 else 40.0))
    return rows


@pytest.fixture
def files(tmp_path):
    rows = _atoms()
    pdb = tmp_path / "model.pdb"
    with open(pdb, "w") as fh:
        for n, (i, res, nm, xyz, b) in enumerate(rows, start=1):
            name = f" {nm:<3}" if len(nm) < 4 else nm
            fh.write(f"ATOM  {n:5d} {name} {res:3s} A{i:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}"
                     f"{xyz[2]:8.3f}  1.00{b:6.2f}           {nm[0]}\n")
        fh.write("END\n")
    cif = tmp_path / "model.cif"
    with open(cif, "w") as fh:
        fh.write("data_test\nloop_\n")
        for c in ("group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id",
                  "label_comp_id", "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y",
                  "Cartn_z", "occupancy", "B_iso_or_equiv", "auth_seq_id", "auth_asym_id",
                  "pdbx_PDB_model_num"):
            fh.write(f"_atom_site.{c}\n")
        for n, (i, res, nm, xyz, b) in enumerate(rows, start=1):
            fh.write(f"ATOM {n} {nm[0]} {nm} . {res} A {i} {xyz[0]:.3f} {xyz[1]:.3f} "
                     f"{xyz[2]:.3f} 1.00 {b:.2f} {i} A 1\n")
        fh.write("#\n")
    dom = tmp_path / "domains.tsv"
    dom.write_text("name\tstart\tend\ttype\nD1\t20\t60\tdomain\nD2\t70\t110\tdomain\n")
    res = pd.DataFrame({"Gene": "GENE1", "estimate_adjusted": [1.5, -0.8, 0.2, 2.0, 0.1, 0.5, 0.3],
                        "padj_adjusted": [0.001, 0.2, 0.9, 0.01, 0.5, 0.3, 0.4]},
                       index=["GENE1_S10", "GENE1_Y30", "GENE1_S50", "GENE1_T75",
                              "GENE1_S100", "GENE1_S12", "GENE1_S200"])
    res.loc["OTHER_S5"] = ["OTHER", 3.0, 0.001]
    return {"pdb": pdb, "cif": cif, "domains": dom, "results": res, "dir": tmp_path}


def test_pdb_and_mmcif_give_the_same_atoms(files):
    a, b = st.read_structure(str(files["pdb"])), st.read_structure(str(files["cif"]))
    assert len(a) == len(b) == len(_atoms())
    np.testing.assert_allclose(a[["x", "y", "z", "bfactor"]], b[["x", "y", "z", "bfactor"]],
                               atol=1e-3)
    assert st.structure_sequence(a, "A")[30] == "Y"


def test_domain_formats(tmp_path, files):
    d = st.read_domains(str(files["domains"]))
    assert d[["start", "end"]].values.tolist() == [[20, 60], [70, 110]]
    up = tmp_path / "uniprot.tsv"
    up.write_text('Entry\tDomain [FT]\nQ15116\tDOMAIN 35..145; /note="Ig-like V-type"; '
                  '/evidence="ECO:1"\n')
    d = st.read_domains(str(up))
    assert d.iloc[0][["name", "start", "end"]].tolist() == ["Ig-like V-type", 35, 145]


def test_site_mapping_checks_residues(files):
    atoms = st.read_structure(str(files["pdb"]))
    sites = st.map_sites(st.sites_from_results(files["results"], "GENE1"), atoms, "A")
    s = sites.set_index("site")
    assert s.loc["GENE1_S10", "residue_matches"] and s.loc["GENE1_S10", "atom"] == "OG"
    assert s.loc["GENE1_Y30", "atom"] == "OH" and s.loc["GENE1_T75", "atom"] == "OG1"
    assert not s.loc["GENE1_S12", "residue_matches"]            # residue 12 is Ala
    assert not s.loc["GENE1_S200", "in_structure"]
    assert s.loc["GENE1_S100", "plddt"] == 40.0
    assert "OTHER_S5" not in s.index


def test_distances_and_annotation(files):
    atoms = st.read_structure(str(files["pdb"]))
    sites = st.map_sites(st.sites_from_results(files["results"], "GENE1"), atoms, "A")
    sites = sites[sites["in_structure"] & sites["residue_matches"]].reset_index(drop=True)
    doms = st.read_domains(str(files["domains"]))
    D = st.domain_distances(sites, atoms, "A", doms)
    assert D.loc["GENE1_S50"].iloc[0] == 0                       # inside D1
    assert D.loc["GENE1_S10"].iloc[0] > 0
    ann = st.annotate_sites(sites, doms, D).set_index("site")
    assert ann.loc["GENE1_S50", "in_domain"] == "D1"
    assert ann.loc["GENE1_S10", "nearest_domain_sequence"] == "D1"
    assert ann.loc["GENE1_S10", "sequence_distance_residues"] == 10
    S = st.site_distances(sites)
    np.testing.assert_allclose(S.values, S.values.T)
    assert (np.diag(S.values) == 0).all()


def test_report_and_pymol_script(files):
    out = files["dir"] / "out"
    r = st.structure_report(files["results"], "GENE1", str(files["pdb"]), str(files["domains"]),
                            str(out), formats=("png",), dpi=40)
    assert len(r["sites"]) == 5
    assert any("mismatch" in m for m in r["messages"])
    assert any("outside" in m for m in r["messages"])
    pml = (out / "GENE1_view.pml").read_text()
    assert 'cmd.load(_f if os.path.exists(_f) else _g, "GENE1")' in pml
    assert '_f = r"model.pdb"' in pml and (out / "model.pdb").exists()
    assert "show spheres, site_S10 and name OG" in pml          # significant site
    assert "show spheres, site_S50" not in pml                   # not significant
    assert "scene F1, store" in pml and "scene F2, store" in pml
    for f in ("GENE1_sites.csv", "GENE1_lollipop.png", "GENE1_site_domain_heatmap.png",
              "GENE1_site_site_heatmap.png"):
        assert (out / f).exists(), f


def test_cli(files, tmp_path):
    from proteomix.structure_cli import main
    path = tmp_path / "res.csv"
    files["results"].to_csv(path)
    main([str(path), "--protein", "GENE1", "--structure", str(files["cif"]), "--domains",
          str(files["domains"]), "-o", str(tmp_path / "cli"), "--formats", "png", "--dpi", "40"])
    assert (tmp_path / "cli" / "GENE1_view.pml").exists()
    with pytest.raises(SystemExit):
        main([str(path), "--protein", "NOPE", "--structure", str(files["pdb"]), "-o",
              str(tmp_path / "x")])
