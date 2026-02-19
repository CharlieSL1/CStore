"""
Cab-ParToCsv

Extract Cabbage/Csound parameters from .csd files (like .snaps stores them)
and write them to CSV. Supports:
- Cabbage widget channels and defaults (hslider range, combobox value, etc.)
- chnset overrides from Export-style CSDs (instr 998 Scheduler).
"""

from __future__ import annotations

import csv
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple


def find_tag_block(
    lines: Sequence[str], start_tag: str, end_tag: str
) -> Tuple[Optional[int], Optional[int]]:
    start = end = None
    for i, ln in enumerate(lines):
        if start is None and start_tag in ln:
            start = i
            continue
        if start is not None and end_tag in ln:
            end = i
            break
    return start, end


# Cabbage: channel("name") and range(min, max, default, ...) or value("x")
CHAN_RE = re.compile(r'channel\s*\(\s*"([^"]+)"\s*\)')
RANGE_RE = re.compile(r'range\s*\(\s*[^,]+,\s*[^,]+,\s*([^,)]+)\s*(?:,[^)]*)?\)')
VALUE_RE = re.compile(r'value\s*\(\s*"([^"]*)"\s*\)')

# Column order matching .snaps format (CAB-Trapped09.snaps)
SNAPS_PARAM_ORDER = [
    "form",
    "trigger",
    "dur",
    "note",
    "rndNote",
    "amp",
    "rndRate",
    "rndAmp",
    "delaySend",
    "delayTime",
    "rvbSend",
    "rvbPan",
    "masterLvl",
    "filebutton32",
    "filebutton33",
]


def _parse_number(s: str) -> float:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def extract_cabbage_params(cabbage_text: str) -> Dict[str, float]:
    """Extract channel -> default value from Cabbage block text."""
    params: Dict[str, float] = {}
    # Split by lines but also handle inline widgets
    for line in cabbage_text.splitlines():
        m = CHAN_RE.search(line)
        if not m:
            continue
        ch = m.group(1)
        default: Optional[float] = None
        rm = RANGE_RE.search(line)
        if rm:
            default = _parse_number(rm.group(1))
        else:
            vm = VALUE_RE.search(line)
            if vm:
                default = _parse_number(vm.group(1))
            elif "button" in line.lower() and "file" not in line.lower():
                default = 0.0
            elif "filebutton" in line.lower():
                default = 0.0
            else:
                default = 0.0
        if default is not None:
            params[ch] = default
    return params


def extract_chnset_overrides(lines: Sequence[str]) -> Dict[str, float]:
    """
    Extract chnset value, "channel" from Scheduler-like instr (e.g. 998).
    Returns channel -> value overrides.
    """
    overrides: Dict[str, float] = {}
    in_scheduler = False
    chnset_re = re.compile(r'chnset\s+([^,]+)\s*,\s*"([^"]+)"')
    for ln in lines:
        if re.search(r'\binstr\s+998\b', ln):
            in_scheduler = True
            continue
        if in_scheduler:
            if 'endin' in ln:
                break
            mm = chnset_re.search(ln)
            if mm:
                try:
                    v = float(mm.group(1).strip())
                    overrides[mm.group(2)] = v
                except ValueError:
                    pass
    return overrides


def get_cabbage_block(lines: Sequence[str]) -> str:
    start, end = find_tag_block(lines, "<Cabbage>", "</Cabbage>")
    if start is None or end is None or end <= start:
        return ""
    return "\n".join(lines[start + 1 : end])


def csd_to_params(csd_path: str) -> Dict[str, float]:
    """
    Extract all parameters (snaps-like) from a CSD file.
    Uses Cabbage defaults; Export-style CSDs override with chnset from instr 998.
    """
    with open(csd_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    cabbage = get_cabbage_block(lines)
    params = extract_cabbage_params(cabbage)
    overrides = extract_chnset_overrides(lines)
    for k, v in overrides.items():
        params[k] = v
    return params


def params_to_csv_rows(
    presets: Dict[str, Dict[str, float]],
    param_order: Optional[List[str]] = None,
) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    presets: preset_name -> { param -> value }.
    param_order: use snaps-like order when provided (e.g. SNAPS_PARAM_ORDER).
    Returns (header_list, list of dicts for csv.DictWriter).
    """
    all_params: set = set()
    for p in presets.values():
        all_params.update(p.keys())
    if param_order is not None:
        order = [x for x in param_order if x in all_params]
        rest = sorted(all_params - set(order))
        headers = ["preset"] + order + rest
    else:
        headers = ["preset"] + sorted(all_params)
    rows: List[Dict[str, str]] = []
    for name, pv in presets.items():
        row: Dict[str, str] = {"preset": name}
        for h in headers[1:]:
            v = pv.get(h)
            row[h] = str(v) if v is not None and v != "" else ""
        rows.append(row)
    return headers, rows


def write_csv(
    path: str,
    presets: Dict[str, Dict[str, float]],
    param_order: Optional[List[str]] = None,
) -> None:
    """Write presets to CSV. Uses snaps-like column order when param_order is SNAPS_PARAM_ORDER."""
    if param_order is None:
        param_order = SNAPS_PARAM_ORDER
    headers, rows = params_to_csv_rows(presets, param_order)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def csd_to_csv(csd_path: str, csv_path: Optional[str] = None) -> str:
    """
    Extract parameters from CSD (single "Default" preset) and write CSV.
    Output column order matches .snaps format. Adds form=0.0 when missing.
    Returns path to written CSV.
    """
    params = csd_to_params(csd_path)
    if "form" not in params:
        params["form"] = 0.0
    presets = {"Default": params}
    if csv_path is None:
        base = os.path.splitext(os.path.basename(csd_path))[0]
        csv_path = os.path.join(os.path.dirname(csd_path), base + "_params.csv")
    d = os.path.dirname(os.path.abspath(csv_path))
    if d:
        os.makedirs(d, exist_ok=True)
    write_csv(csv_path, presets)
    return csv_path


def snaps_to_csv(snaps_path: str, csv_path: Optional[str] = None) -> str:
    """
    Read a .snaps JSON file and write presets to CSV (same structure as csd_to_csv).
    Returns path to written CSV.
    """
    import json

    with open(snaps_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    presets: Dict[str, Dict[str, float]] = {}
    for name, pv in data.items():
        presets[name] = {k: float(v) for k, v in pv.items()}
    if csv_path is None:
        base = os.path.splitext(os.path.basename(snaps_path))[0]
        csv_path = os.path.join(os.path.dirname(snaps_path), base + "_params.csv")
    d = os.path.dirname(os.path.abspath(csv_path))
    if d:
        os.makedirs(d, exist_ok=True)
    write_csv(csv_path, presets)
    return csv_path


if __name__ == "__main__":
    base = "/Users/lishi/Desktop/Research/CStore"
    # 1) CAB-Trapped09.csd (csd/)
    csd1 = os.path.join(base, "csd", "CAB-Trapped09.csd")
    out1 = os.path.join(base, "out", "CAB-Trapped09_params.csv")
    # 2) Export-CAB-Trapped09.csd (out/)
    csd2 = os.path.join(base, "out", "Export-CAB-Trapped09.csd")
    out2 = os.path.join(base, "out", "Export-CAB-Trapped09_params.csv")

    for csd, csv_out in [(csd1, out1), (csd2, out2)]:
        if not os.path.exists(csd):
            print(f"Skip (not found): {csd}")
            continue
        written = csd_to_csv(csd, csv_out)
        print(f"Wrote {written}")
        p = csd_to_params(csd)
        print(f"  Params: {list(p.keys())}")

    # Optional: convert example snaps to CSV for comparison
    snaps = "/Users/lishi/Desktop/Research/CStore-Editable-Persistent-and-Interpretable-Csound-Specifications/csdData/snap/CAB-Trapped09.snaps"
    if os.path.exists(snaps):
        out_snaps = os.path.join(base, "out", "CAB-Trapped09_snaps_params.csv")
        written = snaps_to_csv(snaps, out_snaps)
        print(f"Wrote (from snaps): {written}")
