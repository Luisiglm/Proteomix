"""
Reader for UniProtKB entries in text format (the ".txt" download of an entry, e.g.
P31749.txt; several entries in one file are allowed, separated by "//").

    entry = read_uniprot_text("P31749.txt")[0]
    entry.sequence, entry.gene, entry.accession
    entry.domains()            # name, start, end, type (the domain-file format)
    entry.modified_residues()  # curated PTMs: position, residue, modification, kinase, note
    entry.functional_sites()   # active sites, binding sites (with ligand), other sites

Feature lines ("FT") use the current UniProt layout:
    FT   DOMAIN          150..408
    FT                   /note="Protein kinase"
    FT   MOD_RES         308
    FT                   /note="Phosphothreonine; by PDPK1"
Positions refer to the canonical sequence of the entry. Features with uncertain or
open-ended positions (?, <, >) keep the known end; features located on other entries or
isoforms (e.g. "P12345-2:10..20") are skipped.
"""
import re
from dataclasses import dataclass, field

import pandas as pd

REGION_TYPES = ("DOMAIN", "REGION", "MOTIF", "REPEAT", "ZN_FING", "DNA_BIND", "COILED",
                "TOPO_DOM", "TRANSMEM", "COMPBIAS")
SITE_TYPES = ("ACT_SITE", "BINDING", "SITE")
PTM_TYPES = ("MOD_RES", "CARBOHYD", "LIPID", "CROSSLNK", "DISULFID")


@dataclass
class UniProtEntry:
    accession: str
    accessions: list
    entry_name: str
    gene: str
    organism: str
    sequence: str
    features: pd.DataFrame = field(repr=False)

    # ---- views ----------------------------------------------------------------------
    def domains(self, types=REGION_TYPES, include_disordered: bool = False) -> pd.DataFrame:
        """Regions as name, start, end, type (lower case UniProt feature type)."""
        f = self.features[self.features["type"].isin(types)].copy()
        if not include_disordered:
            f = f[~f["note"].fillna("").str.lower().str.startswith("disordered")]
        f["name"] = f["note"].fillna(f["type"].str.lower()).str.split(";").str[0]
        out = pd.DataFrame({"name": f["name"], "start": f["start"], "end": f["end"],
                            "type": f["type"].str.lower()})
        return out.dropna(subset=["start", "end"]).astype({"start": int, "end": int}) \
            .sort_values(["start", "end"]).reset_index(drop=True)

    def modified_residues(self, phospho_only: bool = False) -> pd.DataFrame:
        """Curated modified residues: position, residue, modification, kinase, evidence."""
        f = self.features[self.features["type"] == "MOD_RES"].dropna(subset=["start"])
        rows = []
        for _, r in f.iterrows():
            note = r["note"] or ""
            mod = note.split(";")[0].strip()
            m = re.search(r"\bby ([^;]+)", note)
            pos = int(r["start"])
            rows.append({"position": pos,
                         "residue": self.sequence[pos - 1] if 0 < pos <= len(self.sequence)
                         else "", "modification": mod,
                         "kinase": m.group(1).strip() if m else "", "note": note,
                         "evidence": r["evidence"]})
        out = pd.DataFrame(rows, columns=["position", "residue", "modification", "kinase",
                                          "note", "evidence"])
        if phospho_only:
            out = out[out["modification"].str.lower().str.startswith("phospho")]
        return out.reset_index(drop=True)

    def functional_sites(self) -> pd.DataFrame:
        """Active sites, binding sites (ligand in 'ligand') and other annotated sites."""
        f = self.features[self.features["type"].isin(SITE_TYPES)].dropna(subset=["start"])
        out = pd.DataFrame({"type": f["type"].str.lower(), "start": f["start"].astype(int),
                            "end": f["end"].astype(int), "ligand": f["ligand"].fillna(""),
                            "note": f["note"].fillna("")})
        out["label"] = [lig or note or t.replace("_", " ") for lig, note, t in
                        zip(out["ligand"], out["note"], out["type"])]
        return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------
_LOC = re.compile(r"^([<>?]?)(\d+|\?)(?:\.\.([<>?]?)(\d+|\?))?$")


def _location(text: str):
    text = text.strip()
    if ":" in text:                                   # feature on another entry / isoform
        return None, None
    m = _LOC.match(text)
    if not m:
        return None, None
    a = int(m.group(2)) if m.group(2) != "?" else None
    b = int(m.group(4)) if m.group(4) and m.group(4) != "?" else (a if not m.group(4) else None)
    a = a if a is not None else b
    b = b if b is not None else a
    return a, b


def _parse_features(ft_lines):
    feats, cur, key, buf = [], None, None, ""

    def close_qualifier():
        nonlocal key, buf
        if cur is not None and key:
            cur[key] = buf.strip().strip('"')
        key, buf = None, ""

    for line in ft_lines:
        body = line[5:].rstrip("\n")
        if body[:16].strip() and not body.startswith(" "):  # new feature: type + location
            close_qualifier()
            if cur:
                feats.append(cur)
            parts = body.split(None, 1)
            start, end = _location(parts[1] if len(parts) > 1 else "")
            cur = {"type": parts[0], "start": start, "end": end}
            continue
        q = body.strip()
        m = re.match(r"^/(\w+)=(.*)$", q)
        if m:
            close_qualifier()
            key, buf = m.group(1), m.group(2)
        elif key:                                            # continuation of a qualifier
            sep = "" if buf.endswith("-") else " "
            buf += sep + q
    close_qualifier()
    if cur:
        feats.append(cur)
    df = pd.DataFrame(feats)
    for col in ("note", "evidence", "ligand", "id"):
        if col not in df:
            df[col] = None
    return df


def read_uniprot_text(path: str) -> list:
    """All entries in a UniProtKB text file, as UniProtEntry objects."""
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8") as fh:
        text = fh.read()
    entries = []
    for block in text.split("\n//"):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not any(ln.startswith("ID   ") for ln in lines):
            continue
        get = lambda code: [ln[5:].strip() for ln in lines if ln.startswith(code)]  # noqa
        idl = get("ID   ")[0].split()[0]
        accs = [a.strip() for ln in get("AC   ") for a in ln.split(";") if a.strip()]
        gn = " ".join(get("GN   "))
        gm = re.search(r"Name=([^;{ ]+)", gn)
        os_ = " ".join(get("OS   ")).rstrip(".")
        sq_start = next((i for i, ln in enumerate(lines) if ln.startswith("SQ   ")), None)
        seq = "".join(ln.replace(" ", "") for ln in lines[sq_start + 1:]) \
            if sq_start is not None else ""
        feats = _parse_features([ln for ln in lines if ln.startswith("FT   ")])
        entries.append(UniProtEntry(accession=accs[0] if accs else idl, accessions=accs,
                                    entry_name=idl, gene=gm.group(1) if gm else idl.split("_")[0],
                                    organism=os_, sequence=seq.upper(), features=feats))
    if not entries:
        raise ValueError(f"no UniProt entries found in {path} (expected the text format, "
                         "starting with 'ID   ')")
    return entries


def is_uniprot_text(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.readline().startswith("ID   ")
    except (OSError, UnicodeDecodeError):
        return False
