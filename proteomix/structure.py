"""
Phosphosites in their structural context, one protein at a time, from user-provided files.

Inputs
  structure   PDB (.pdb) or mmCIF (.cif) file, e.g. an AlphaFold model or an experimental
              structure; for AlphaFold models the B-factor column holds pLDDT (confidence)
  domains     table of regions: a simple .tsv / .csv with columns name, start, end
              (optional: type), a UniProt TSV export with feature columns such as
              "Domain [FT]" / "Region" / "Motif", or an InterPro TSV
  sites       a proteomix phospho results table (tables/<contrast>.csv), or any table with
              a site column like GENE_S123 / P12345_Y223 and estimate / padj columns

Outputs (structure_report)
  sites table      residue check, pLDDT, nearest domain in sequence and in 3D, distances
  lollipop plot    sites along the sequence over domain boxes, coloured by log2 FC, with
                   significant sites marked and a pLDDT track
  heatmaps         site x domain and site x site 3D distances (angstrom)
  PyMOL script     open in PyMOL (File > Run Script, or `pymol view.pml`): cartoon coloured
                   by domain, sites as sticks coloured by log2 FC, significant sites as spheres,
                   labels; scenes F1 (domains) and F2 (pLDDT); rotate and save images with
                   File > Export Image / `png name.png, dpi=300, ray=1`

Site positions must use the same numbering as the structure (usually UniProt's canonical
sequence): every site is checked to be S, T or Y in the structure, and mismatches (other
isoform, partial structure) are reported, not silently mapped.
Distances use the phosphorylatable oxygen (Ser OG, Thr OG1, Tyr OH), else the C-alpha;
distances in regions with low pLDDT (< 50, usually disordered) are unreliable.
"""
import os
import re
import shlex
import shutil

import numpy as np
import pandas as pd

THREE_TO_ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
                "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
                "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
                "TYR": "Y", "VAL": "V", "SEP": "S", "TPO": "T", "PTR": "Y", "MSE": "M"}
SITE_ATOM = {"S": "OG", "T": "OG1", "Y": "OH"}


# --------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------
def read_structure(path: str) -> pd.DataFrame:
    """Atoms of a PDB or mmCIF file: chain, resnum, resname, atom, x, y, z, bfactor."""
    if path.lower().endswith((".cif", ".mmcif", ".cif.gz")):
        return _read_mmcif(path)
    rows = []
    with _open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                if line[16] not in (" ", "A"):                  # alternate locations
                    continue
                rows.append((line[21].strip() or "A", int(line[22:26]), line[17:20].strip(),
                             line[12:16].strip(), float(line[30:38]), float(line[38:46]),
                             float(line[46:54]), float(line[60:66] or 0)))
            elif line.startswith("ENDMDL"):                     # first model only
                break
    if not rows:
        raise ValueError(f"no atoms found in {path}")
    return pd.DataFrame(rows, columns=["chain", "resnum", "resname", "atom", "x", "y", "z",
                                       "bfactor"])


def _open(path):
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt")
    return open(path, encoding="utf-8", errors="replace")


def _read_mmcif(path):
    """Minimal mmCIF reader for the _atom_site loop (first model, no alternate locations)."""
    with _open(path) as fh:
        lines = fh.read().splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith("_atom_site."):
        i += 1
    cols = []
    while i < len(lines) and lines[i].startswith("_atom_site."):
        cols.append(lines[i].split(".", 1)[1].strip())
        i += 1
    if not cols:
        raise ValueError(f"no _atom_site loop in {path}")
    c = {name: k for k, name in enumerate(cols)}

    def col(*names):
        for nm in names:
            if nm in c:
                return c[nm]
        raise ValueError(f"mmCIF lacks {names[0]}")
    ic = {"chain": col("auth_asym_id", "label_asym_id"), "res": col("auth_seq_id", "label_seq_id"),
          "resname": col("auth_comp_id", "label_comp_id"),
          "atom": col("auth_atom_id", "label_atom_id"),
          "x": col("Cartn_x"), "y": col("Cartn_y"), "z": col("Cartn_z"),
          "b": col("B_iso_or_equiv")}
    alt = c.get("label_alt_id")
    model = c.get("pdbx_PDB_model_num")
    rows, first_model = [], None
    for line in lines[i:]:
        if not line.strip() or line.startswith(("#", "loop_", "_")):
            if rows:
                break
            continue
        f = shlex.split(line)
        if len(f) < len(cols):
            continue
        if model is not None:
            first_model = first_model or f[model]
            if f[model] != first_model:
                break
        if alt is not None and f[alt] not in (".", "?", "A"):
            continue
        rows.append((f[ic["chain"]], int(f[ic["res"]]), f[ic["resname"]], f[ic["atom"]].strip('"'),
                     float(f[ic["x"]]), float(f[ic["y"]]), float(f[ic["z"]]), float(f[ic["b"]])))
    if not rows:
        raise ValueError(f"no atoms parsed from {path}")
    return pd.DataFrame(rows, columns=["chain", "resnum", "resname", "atom", "x", "y", "z",
                                       "bfactor"])


def structure_sequence(atoms: pd.DataFrame, chain: str) -> dict:
    """{residue number: one-letter code} of a chain."""
    r = atoms[atoms["chain"] == chain].drop_duplicates("resnum")
    return {int(n): THREE_TO_ONE.get(rn, "X") for n, rn in zip(r["resnum"], r["resname"])}


FT_PATTERN = re.compile(r"\b([A-Z_]+)\s+(\d+)\.\.(\d+);\s*/note=\"([^\"]+)\"")


def read_domains(path: str) -> pd.DataFrame:
    """
    Regions as name, start, end, type. Accepts a simple table (name/start/end[/type]),
    a UniProt TSV export (feature columns: 'DOMAIN 35..145; /note="Ig-like V-type"; ...'),
    or an InterPro TSV (signature description and start / end in columns 6-8).
    """
    sep = "\t" if open(path, encoding="utf-8").readline().count("\t") else ","
    raw = pd.read_csv(path, sep=sep, dtype=str, header=None)
    head = [str(x).strip().lower() for x in raw.iloc[0]]
    if {"start", "end"} <= set(head):
        t = raw.iloc[1:].copy()
        t.columns = head
        name = "name" if "name" in head else head[0]
        out = pd.DataFrame({"name": t[name].astype(str), "start": t["start"].astype(int),
                            "end": t["end"].astype(int),
                            "type": t["type"].astype(str) if "type" in t else "domain"})
        return out.sort_values("start").reset_index(drop=True)
    text = raw.astype(str).agg(" ".join, axis=1).str.cat(sep=" ")
    ft = FT_PATTERN.findall(text)
    if ft:
        out = pd.DataFrame([{"name": note, "start": int(a), "end": int(b), "type": kind.lower()}
                            for kind, a, b, note in ft]).drop_duplicates()
        return out.sort_values("start").reset_index(drop=True)
    if raw.shape[1] >= 8 and raw[6].str.isdigit().all() and raw[7].str.isdigit().all():
        out = pd.DataFrame({"name": raw[5].fillna(raw[4]), "start": raw[6].astype(int),
                            "end": raw[7].astype(int), "type": raw[3].str.lower()})
        return out.drop_duplicates(["name", "start", "end"]).sort_values("start") \
            .reset_index(drop=True)
    raise ValueError("domain file not recognised: use columns name, start, end (and "
                     "optionally type), a UniProt TSV export with feature columns, or an "
                     "InterPro TSV")


SITE_RE = re.compile(r"(?:^|[_\-\s])([STY])(\d+)(?:$|[_\-\s])")


def sites_from_results(table: pd.DataFrame, protein: str, estimate_col=None, padj_col=None):
    """
    Rows of a results table belonging to `protein` (matched against Gene, protein_key,
    Protein ID or the start of the site name), with residue and position parsed from the
    site name (…_S123). Returns DataFrame site, residue, position, estimate, padj.
    """
    t = table.copy()
    if "site_name" not in t:
        t["site_name"] = t.index.astype(str) if t.index.dtype == object else \
            t.iloc[:, 0].astype(str)
    p = str(protein).upper()
    mask = pd.Series(False, index=t.index)
    for col in ("Gene", "protein_key", "Protein ID", "gene"):
        if col in t:
            mask |= t[col].astype(str).str.upper() == p
    mask |= t["site_name"].str.upper().str.startswith(p + "_")
    t = t[mask]
    if t.empty:
        raise ValueError(f"no sites of {protein!r} in the results table")
    est = estimate_col or next(c for c in ("estimate_adjusted", "estimate") if c in t)
    pad = padj_col or next(c for c in ("padj_adjusted", "padj") if c in t)
    rows = []
    for name, e, q in zip(t["site_name"], t[est], t[pad]):
        m = SITE_RE.search(str(name)) or SITE_RE.search(str(name) + "_")
        if m:
            rows.append({"site": name, "residue": m.group(1), "position": int(m.group(2)),
                         "estimate": float(e), "padj": float(q)})
    if not rows:
        raise ValueError(f"could not read residue positions from the site names of {protein!r}"
                         " (expected names like GENE_S123)")
    return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Mapping and distances
# --------------------------------------------------------------------------------------
def map_sites(sites: pd.DataFrame, atoms: pd.DataFrame, chain: str) -> pd.DataFrame:
    """Adds structure_residue, residue_matches, plddt and the site-atom coordinates."""
    a = atoms[atoms["chain"] == chain]
    seq = structure_sequence(atoms, chain)
    out = sites.copy()
    xyz, res_in, plddt, atom_used = [], [], [], []
    for r, pos in zip(out["residue"], out["position"]):
        rr = a[a["resnum"] == pos]
        res_in.append(seq.get(int(pos), "-"))
        if rr.empty:
            xyz.append((np.nan,) * 3)
            plddt.append(np.nan)
            atom_used.append("")
            continue
        pick = rr[rr["atom"] == SITE_ATOM.get(r, "CA")]
        if pick.empty:
            pick = rr[rr["atom"] == "CA"]
        pick = pick if not pick.empty else rr
        xyz.append(tuple(pick[["x", "y", "z"]].iloc[0]))
        atom_used.append(pick["atom"].iloc[0])
        ca = rr[rr["atom"] == "CA"]
        plddt.append(float((ca if not ca.empty else rr)["bfactor"].iloc[0]))
    out["structure_residue"] = res_in
    out["in_structure"] = out["structure_residue"] != "-"
    out["residue_matches"] = out["structure_residue"] == out["residue"]
    out["plddt"] = plddt
    out["atom"] = atom_used
    out[["x", "y", "z"]] = np.array(xyz, dtype=float)
    return out


def domain_distances(sites: pd.DataFrame, atoms: pd.DataFrame, chain: str,
                     domains: pd.DataFrame) -> pd.DataFrame:
    """Minimum 3D distance (angstrom) from each site atom to any heavy atom of each domain."""
    a = atoms[(atoms["chain"] == chain) & ~atoms["atom"].str.startswith("H")]
    D = np.full((len(sites), len(domains)), np.nan)
    P = sites[["x", "y", "z"]].to_numpy(float)
    for j, (_, d) in enumerate(domains.iterrows()):
        sel = a[(a["resnum"] >= d["start"]) & (a["resnum"] <= d["end"])]
        if sel.empty:
            continue
        Q = sel[["x", "y", "z"]].to_numpy(float)
        D[:, j] = np.sqrt(((P[:, None, :] - Q[None]) ** 2).sum(-1)).min(1)
    labels = [f"{d['name']} ({d['start']}-{d['end']})" for _, d in domains.iterrows()]
    return pd.DataFrame(D, index=sites["site"], columns=labels)


def site_distances(sites: pd.DataFrame) -> pd.DataFrame:
    P = sites[["x", "y", "z"]].to_numpy(float)
    D = np.sqrt(((P[:, None, :] - P[None]) ** 2).sum(-1))
    return pd.DataFrame(D, index=sites["site"], columns=sites["site"])


def annotate_sites(sites, domains, dist_dom):
    """Domain containing each site, nearest domain in sequence and in 3D."""
    out = sites.copy()
    inside, near_seq, near_seq_d, near_3d, near_3d_d = [], [], [], [], []
    for i, pos in enumerate(out["position"]):
        cont = domains[(domains["start"] <= pos) & (domains["end"] >= pos)]
        inside.append("; ".join(cont["name"]) if len(cont) else "")
        if len(domains):
            sd = np.where((domains["start"] <= pos) & (domains["end"] >= pos), 0,
                          np.minimum(abs(domains["start"] - pos), abs(domains["end"] - pos)))
            k = int(np.argmin(sd))
            near_seq.append(domains["name"].iloc[k])
            near_seq_d.append(int(sd[k]))
            row = dist_dom.iloc[i].to_numpy(float)
            if np.isfinite(row).any():
                k3 = int(np.nanargmin(row))
                near_3d.append(domains["name"].iloc[k3])
                near_3d_d.append(round(float(row[k3]), 1))
            else:
                near_3d.append("")
                near_3d_d.append(np.nan)
        else:
            near_seq.append("")
            near_seq_d.append(np.nan)
            near_3d.append("")
            near_3d_d.append(np.nan)
    out["in_domain"] = inside
    out["nearest_domain_sequence"] = near_seq
    out["sequence_distance_residues"] = near_seq_d
    out["nearest_domain_3d"] = near_3d
    out["distance_3d_angstrom"] = near_3d_d
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def _lfc_colours(values, lim=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    v = np.asarray(values, float)
    lim = lim or max(np.nanmax(np.abs(v)) if np.isfinite(v).any() else 1.0, 0.5)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    cmap = plt.get_cmap("RdBu_r")
    return [cmap(norm(x))[:3] if np.isfinite(x) else (0.6, 0.6, 0.6) for x in v], cmap, norm


def plot_lollipop(sites, domains, protein, length=None, plddt=None, alpha=0.05,
                  font_size: float = 16, figsize=None):
    """Sites along the sequence (height and colour = log2 FC) above domain boxes and pLDDT."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from .figures import OKABE_ITO, publication_style
    L = length or int(max(sites["position"].max(), domains["end"].max() if len(domains) else 0)) + 10
    cols, cmap, norm = _lfc_colours(sites["estimate"])
    has_p = plddt is not None and len(plddt)
    figsize = figsize or (max(10, L / 60), 6.5 if has_p else 5.5)
    with publication_style(font_size):
        fig, axes = plt.subplots(3 if has_p else 2, 1, figsize=figsize, sharex=True,
                                 gridspec_kw={"height_ratios": [3, 0.8, 0.7] if has_p else [3, 0.8],
                                              "hspace": 0.08})
        ax = axes[0]
        sig = sites["padj"].fillna(1) < alpha
        for (_, s), c, sg in zip(sites.iterrows(), cols, sig):
            ax.plot([s["position"]] * 2, [0, s["estimate"]], color="0.5", lw=1.2, zorder=1)
            ax.scatter(s["position"], s["estimate"], s=120 if sg else 60, color=c,
                       edgecolor="black" if sg else "0.4", linewidth=1.6 if sg else 0.6,
                       zorder=3)
            if sg or len(sites) <= 15:
                ax.annotate(f"{s['residue']}{s['position']}", (s["position"], s["estimate"]),
                            xytext=(0, 9 if s["estimate"] >= 0 else -16),
                            textcoords="offset points", ha="center", fontsize=font_size - 5)
        ax.axhline(0, color="0.3", lw=1)
        ax.set_ylabel("log$_2$ FC")
        lim = max(np.nanmax(np.abs(sites["estimate"])) * 1.35, 0.5)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{protein}: phosphosites (black outline = FDR < {alpha:g})", loc="left")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.01)
        cb.set_label("log$_2$ FC")
        cb.outline.set_visible(False)
        axd = axes[1]
        axd.add_patch(Rectangle((1, 0.42), L - 1, 0.16, color="0.8", lw=0))
        for k, (_, d) in enumerate(domains.iterrows()):          # same colours as PyMOL
            axd.add_patch(Rectangle((d["start"], 0.1), d["end"] - d["start"], 0.8,
                                    color=OKABE_ITO[k % len(OKABE_ITO)], alpha=0.9, lw=0))
            if d["end"] - d["start"] > L * 0.04:
                axd.text((d["start"] + d["end"]) / 2, 0.5, d["name"], ha="center",
                         va="center", fontsize=font_size - 6, color="white", clip_on=True)
        for x in sites["position"]:
            axd.axvline(x, color="0.2", lw=0.6, alpha=0.5)
        axd.set_ylim(0, 1)
        axd.set_yticks([])
        for s_ in ("left", "right", "top"):
            axd.spines[s_].set_visible(False)
        if has_p:
            axp = axes[2]
            axp.fill_between(plddt.index, plddt.to_numpy(), color=OKABE_ITO[5], alpha=0.6,
                             step="mid", lw=0)
            axp.axhline(70, color="0.4", ls=":", lw=1)
            axp.set_ylim(0, 100)
            axp.set_ylabel("pLDDT", fontsize=font_size - 3)
        axes[-1].set_xlim(0, L)
        axes[-1].set_xlabel("Residue")
    return fig


def plot_distance_heatmap(D: pd.DataFrame, title: str, plddt=None, font_size: float = 16,
                          figsize=None, max_dist: float = 40):
    """Distance heatmap (angstrom); short distances dark. Optional pLDDT label per row."""
    import matplotlib.pyplot as plt
    from .figures import publication_style
    rows = [f"{r} (pLDDT {p:.0f})" if plddt is not None and np.isfinite(p) else str(r)
            for r, p in zip(D.index, plddt if plddt is not None else [np.nan] * len(D))]
    figsize = figsize or (max(6, 0.55 * D.shape[1] + 4), max(4, 0.42 * D.shape[0] + 2.5))
    with publication_style(font_size):
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(np.minimum(D.to_numpy(float), max_dist), cmap="magma", vmin=0,
                       vmax=max_dist, aspect="auto")
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                v = D.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=max(font_size - 7, 7),
                            color="white" if v < max_dist * 0.55 else "black")
        ax.set_xticks(range(D.shape[1]))
        ax.set_xticklabels(D.columns, rotation=45, ha="right", fontsize=font_size - 4)
        ax.set_yticks(range(D.shape[0]))
        ax.set_yticklabels(rows, fontsize=font_size - 4)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(f"Distance (Å, capped at {max_dist:g})")
        cb.outline.set_visible(False)
        ax.set_title(title, loc="left")
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# PyMOL script
# --------------------------------------------------------------------------------------
def write_pymol_script(path: str, structure_file: str, sites, domains, chain: str,
                       protein: str, alpha: float = 0.05) -> str:
    """
    PyMOL script: cartoon coloured by domain (scene F1) or pLDDT (scene F2), sites as
    sticks coloured by log2 FC, significant sites as spheres, labels. The structure file
    is referenced by name, so keep it in the same folder as the script.
    """
    from .figures import OKABE_ITO
    obj = re.sub(r"\W", "_", protein)
    cols, _, _ = _lfc_colours(sites["estimate"])
    lines = [f"# proteomix: {protein} phosphosites. Open in PyMOL (File > Run Script, or",
             "# `pymol this_file.pml`). Rotate, then File > Export Image or:",
             "#   png view.png, width=2400, height=1800, dpi=300, ray=1",
             "reinitialize",
             "# load the structure: from PyMOL's current folder, else from where it was written",
             "python",
             "import os",
             f"_f = r\"{os.path.basename(structure_file)}\"",
             f"_g = r\"{os.path.abspath(structure_file).replace(os.sep, '/')}\"",
             f"cmd.load(_f if os.path.exists(_f) else _g, \"{obj}\")",
             "python end",
             "bg_color white", "set ray_opaque_background, 0", "hide everything",
             f"show cartoon, {obj}", f"color grey90, {obj}",
             "set cartoon_transparency, 0.1", "set label_size, 18",
             "set label_color, black", "set label_outline_color, white",
             "set label_position, (0, 0, 4)", "set sphere_scale, 0.7"]
    hexes = [c.lstrip("#") for c in OKABE_ITO]
    for i, (_, d) in enumerate(domains.iterrows()):
        h = hexes[i % len(hexes)]
        lines += [f"set_color dom_{i}, [{int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}]",
                  f"color dom_{i}, {obj} and chain {chain} and resi {d['start']}-{d['end']}",
                  f"# domain {i}: {d['name']} ({d['start']}-{d['end']})"]
    sig = sites["padj"].fillna(1) < alpha
    for (_, s), c, sg in zip(sites.iterrows(), cols, sig):
        if not s.get("in_structure", True):
            continue
        sel = f"{obj} and chain {chain} and resi {s['position']}"
        name = f"site_{s['residue']}{s['position']}"
        lines += [f"set_color c_{name}, [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]",
                  f"select {name}, {sel}",
                  f"show sticks, {name} and not name N+C+O",
                  f"color c_{name}, {name}",
                  f"label {name} and name CA, \"{s['residue']}{s['position']}\""]
        if sg:
            atom = SITE_ATOM.get(s["residue"], "CA")
            lines.append(f"show spheres, {name} and name {atom}")
    lines += ["deselect", f"orient {obj}", "scene F1, store",
              "# scene F2: colour by B-factor = AlphaFold confidence (pLDDT; red < 50, "
              "green > 90).",
              "# Only meaningful for predicted models; experimental B-factors mean something else.",
              f"spectrum b, red_yellow_green, {obj}, minimum=30, maximum=100"]
    for _, s in sites.iterrows():
        if s.get("in_structure", True):
            name = f"site_{s['residue']}{s['position']}"
            lines.append(f"color c_{name}, {name} and not name N+CA+C+O")
    lines += ["scene F2, store", "scene F1, recall"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def structure_report(results: pd.DataFrame, protein: str, structure_path: str,
                     domains_path: str | None, out_dir: str, chain: str | None = None,
                     alpha: float = 0.05, estimate_col=None, padj_col=None,
                     formats=("pdf", "png"), dpi: int = 600, font_size: float = 16) -> dict:
    """Everything for one protein: tables, figures and the PyMOL script (in out_dir)."""
    import matplotlib.pyplot as plt
    from .figures import save_figure
    os.makedirs(out_dir, exist_ok=True)
    atoms = read_structure(structure_path)
    chain = chain or atoms["chain"].iloc[0]
    if chain not in set(atoms["chain"]):
        raise ValueError(f"chain {chain!r} not in the structure (chains: "
                         f"{sorted(set(atoms['chain']))})")
    domains = read_domains(domains_path) if domains_path else \
        pd.DataFrame(columns=["name", "start", "end", "type"])
    sites = map_sites(sites_from_results(results, protein, estimate_col, padj_col), atoms, chain)
    ok = sites["in_structure"] & sites["residue_matches"]
    messages = []
    if (~sites["in_structure"]).any():
        messages.append(f"{(~sites['in_structure']).sum()} site(s) outside the residues in the "
                        f"structure: {', '.join(sites.loc[~sites['in_structure'], 'site'])}")
    bad = sites["in_structure"] & ~sites["residue_matches"]
    if bad.any():
        messages.append("residue mismatch (different isoform or numbering?): " + ", ".join(
            f"{s} is {r} in the structure" for s, r in
            zip(sites.loc[bad, "site"], sites.loc[bad, "structure_residue"])))
    mapped = sites[ok].reset_index(drop=True)
    dd = domain_distances(mapped, atoms, chain, domains) if len(domains) and len(mapped) \
        else pd.DataFrame(index=mapped["site"])
    table = annotate_sites(mapped, domains, dd) if len(mapped) else mapped
    slug = re.sub(r"\W", "_", protein)
    table.drop(columns=["x", "y", "z"]).to_csv(os.path.join(out_dir, f"{slug}_sites.csv"),
                                                index=False)
    if not dd.empty:
        dd.to_csv(os.path.join(out_dir, f"{slug}_site_domain_distances.csv"))
    ca = atoms[(atoms["chain"] == chain) & (atoms["atom"] == "CA")]
    plddt = ca.set_index("resnum")["bfactor"]
    seq_len = int(ca["resnum"].max())

    def save(fig, name):
        save_figure(fig, os.path.join(out_dir, f"{slug}_{name}"), formats, dpi)
        plt.close(fig)
    fig = plot_lollipop(mapped, domains, protein, length=seq_len, plddt=plddt, alpha=alpha,
                        font_size=font_size)
    n_out = int((~ok).sum())
    if n_out:
        fig.text(0.01, -0.04, f"{n_out} site(s) not shown: outside the structure or residue "
                 "mismatch (see notes)", fontsize=font_size - 5, color="0.35")
    save(fig, "lollipop")
    if not dd.empty:
        save(plot_distance_heatmap(dd, f"{protein}: site to domain distance",
                                   plddt=mapped["plddt"].to_numpy(), font_size=font_size),
             "site_domain_heatmap")
    if len(mapped) >= 2:
        save(plot_distance_heatmap(site_distances(mapped), f"{protein}: site to site distance",
                                   plddt=mapped["plddt"].to_numpy(), font_size=font_size),
             "site_site_heatmap")
    struct_copy = os.path.join(out_dir, os.path.basename(structure_path))
    if os.path.abspath(struct_copy) != os.path.abspath(structure_path):
        shutil.copyfile(structure_path, struct_copy)
    pml = write_pymol_script(os.path.join(out_dir, f"{slug}_view.pml"), struct_copy, mapped,
                             domains, chain, protein, alpha)
    low = mapped["plddt"] < 50
    if low.any():
        messages.append(f"{low.sum()} site(s) in low-confidence regions (pLDDT < 50, likely "
                        f"disordered): their 3D distances are unreliable")
    return {"sites": table, "domain_distances": dd, "pymol": pml, "messages": messages,
            "chain": chain}
