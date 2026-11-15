"""
Run-level metadata files and contrasts defined by group names.

read_run_metadata reads a two-column file (run or column name, condition), e.g.

    230801kw_..._PD1_A1_S3_A1_1_15972 MaxLFQ Intensity<TAB>IP unstimulated

strips the quantity suffix so the key matches data.runs["run"], tidies the condition labels
(whitespace, "5min" -> "5 min") and, with parse_factors=True, splits them into factors:

    "IP TCS PDL1 20 min"  -> pulldown "IP",  treatment "PDL1",         time_min 20
    "IgG unstimulated"    -> pulldown "IgG", treatment "unstimulated", time_min 0

Contrasts can then be written with group names instead of column indices:

    {"IP vs IgG, unstimulated": ("IP unstimulated", "IgG unstimulated"),
     "PDL1 vs Ctrl (IP, 5 min)": ("IP TCS PDL1 5 min", "IP TCS Ctrl 5 min"),
     "PDL1 effect, IgG-corrected": {"IP TCS PDL1 5 min": 1, "IgG TCS PDL1 5 min": -1,
                                    "IP TCS Ctrl 5 min": -1, "IgG TCS Ctrl 5 min": 1}}
"""
import re

import numpy as np
import pandas as pd

QUANTITY_SUFFIXES = (" MaxLFQ Intensity", " Unique Spectral Count", " Total Spectral Count",
                     " Spectral Count", " Intensity")


def clean_condition(label: str) -> str:
    """Collapses whitespace and writes durations consistently: '5min' -> '5 min', '4h' -> '4 h'."""
    s = re.sub(r"\s+", " ", str(label)).strip()
    s = re.sub(r"(\d+)\s*min\b", r"\1 min", s, flags=re.I)
    s = re.sub(r"(\d+)\s*h\b", r"\1 h", s, flags=re.I)
    return s


def parse_condition(label: str) -> dict:
    """
    Splits '<pulldown> [TCS] <treatment> [<time>]' into factors. Pulldown is the first word
    (IP / IgG); 'unstimulated' has time 0; durations are converted to minutes.
    """
    words = label.split()
    pulldown = words[0] if words else ""
    rest = " ".join(words[1:])
    m = re.search(r"(\d+)\s*(min|h)\b", rest)
    time_min = 0 if m is None else int(m.group(1)) * (60 if m.group(2).lower() == "h" else 1)
    body = rest[: m.start()].strip() if m else rest
    body = re.sub(r"^TCS\s+", "", body)
    return {"pulldown": pulldown, "treatment": body or "unstimulated", "time_min": time_min}


def read_run_metadata(path: str, run_col: int = 0, condition_col: int = 1,
                      condition_name: str = "condition", parse_factors: bool = True,
                      sep: str | None = None, sample_col: int | None = None) -> pd.DataFrame:
    """
    Reads a run -> condition table (tab-, comma- or semicolon-separated, header optional).
    Returns columns: run (matches data.runs["run"]), condition (tidied), and, if
    parse_factors, pulldown / treatment / time_min.
    sample_col: 0-based column holding a biological-sample ID (returned as "sample_id"); use it
        to link runs of different files (e.g. phospho and whole proteome) or to define
        technical replicates when run names cannot be parsed.
    """
    if sep is None:
        with open(path, encoding="utf-8-sig") as fh:
            first = fh.readline()
        sep = max(["\t", ",", ";"], key=first.count)
    raw = pd.read_csv(path, sep=sep, header=None, dtype=str, encoding="utf-8-sig",
                      skip_blank_lines=True)
    first_key = str(raw.iloc[0, run_col])
    looks_like_data = first_key.endswith(QUANTITY_SUFFIXES) or re.search(r"\d", first_key)
    if not looks_like_data:                     # header row (names without digits, e.g. "run")
        raw = raw.iloc[1:]
    run = raw.iloc[:, run_col].astype(str).str.strip()
    for suf in QUANTITY_SUFFIXES:
        run = run.str.replace(re.escape(suf) + "$", "", regex=True)
    meta = pd.DataFrame({"run": run.to_numpy(),
                         condition_name: raw.iloc[:, condition_col].map(clean_condition)
                         .to_numpy()})
    if sample_col is not None:
        meta["sample_id"] = raw.iloc[:, sample_col].astype(str).str.strip().to_numpy()
    meta = meta[meta["run"].str.len() > 0].reset_index(drop=True)
    if meta["run"].duplicated().any():
        dup = meta.loc[meta["run"].duplicated(), "run"].iloc[0]
        raise ValueError(f"run {dup!r} appears more than once in the metadata")
    if parse_factors:
        meta = pd.concat([meta, pd.DataFrame([parse_condition(c) for c in meta[condition_name]])],
                         axis=1)
    return meta


def condition_order(meta: pd.DataFrame, condition_col: str = "condition",
                    by=("pulldown", "treatment", "time_min"), pulldown_order=("IP", "IgG"),
                    treatment_order=("unstimulated", "Ctrl", "PDL1")) -> list:
    """Condition labels sorted by pulldown, then treatment, then time (unknown levels last)."""
    t = meta.drop_duplicates(condition_col).copy()

    def rank(values, order):
        order = list(order)
        return values.map(lambda v: order.index(v) if v in order else len(order))

    keys = []
    if "pulldown" in by and "pulldown" in t:
        t["_p"] = rank(t["pulldown"], pulldown_order)
        keys.append("_p")
    if "treatment" in by and "treatment" in t:
        t["_t"] = rank(t["treatment"], treatment_order)
        keys.append("_t")
    if "time_min" in by and "time_min" in t:
        keys.append("time_min")
    return t.sort_values(keys + [condition_col])[condition_col].tolist()


def contrast_vector(spec, group_names) -> np.ndarray:
    """
    (g1, g0) by name or index -> +1/-1; {name: weight} -> weights; a vector passes through.
    """
    names = list(group_names)
    v = np.zeros(len(names))

    def idx(g):
        if isinstance(g, (int, np.integer)):
            return int(g)
        if g not in names:
            close = [n for n in names if n.lower().replace(" ", "") == str(g).lower().replace(" ", "")]
            if close:
                return names.index(close[0])
            raise KeyError(f"group {g!r} not found; groups are {names}")
        return names.index(g)

    if isinstance(spec, dict):
        for g, wgt in spec.items():
            v[idx(g)] += wgt
    elif isinstance(spec, tuple) and len(spec) == 2:
        v[idx(spec[0])] += 1.0
        v[idx(spec[1])] -= 1.0
    else:
        v = np.asarray(spec, dtype=float)
        if len(v) != len(names):
            raise ValueError(f"contrast has length {len(v)}, expected {len(names)}")
    return v


def ap_ms_contrasts(meta: pd.DataFrame, condition_col: str = "condition",
                    ip: str = "IP", control: str = "IgG", ctrl_treatment: str = "Ctrl",
                    treatment: str = "PDL1") -> dict:
    """
    Standard contrasts for an IP / IgG x treatment x time design, by condition name:
      enrichment       IP vs IgG for every condition (interactors / specific binders)
      treatment        treatment vs control treatment within IP, at each time
      corrected        the same, minus the IgG difference (removes background changes)
      time             each IP control-treatment time point vs IP unstimulated
    Only contrasts whose conditions all exist are returned.
    """
    t = meta.drop_duplicates(condition_col)
    cond = {(r.pulldown, r.treatment, r.time_min): r[condition_col] for _, r in t.iterrows()}
    out = {}

    def has(*keys):
        return all(k in cond for k in keys)

    for (pd_, tr, tm), name in sorted(cond.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        if pd_ == ip and has((control, tr, tm)):
            when = "unstimulated" if tr == "unstimulated" else f"{tr} {_fmt(tm)}"
            out[f"IP vs IgG: {when}"] = (name, cond[(control, tr, tm)])
    times = sorted({tm for (_, tr, tm) in cond if tr == treatment})
    for tm in times:
        a, b = (ip, treatment, tm), (ip, ctrl_treatment, tm)
        if has(a, b):
            out[f"{treatment} vs {ctrl_treatment} (IP, {_fmt(tm)})"] = (cond[a], cond[b])
        ca, cb = (control, treatment, tm), (control, ctrl_treatment, tm)
        if has(a, b, ca, cb):
            out[f"{treatment} vs {ctrl_treatment} (IP − IgG, {_fmt(tm)})"] = {
                cond[a]: 1, cond[b]: -1, cond[ca]: -1, cond[cb]: 1}
    for tm in sorted({tm for (_, tr, tm) in cond if tr == ctrl_treatment}):
        a, b = (ip, ctrl_treatment, tm), (ip, "unstimulated", 0)
        if has(a, b):
            out[f"{ctrl_treatment} {_fmt(tm)} vs unstimulated (IP)"] = (cond[a], cond[b])
    return out


def _fmt(minutes: int) -> str:
    return f"{minutes // 60} h" if minutes >= 60 and minutes % 60 == 0 else f"{minutes} min"
