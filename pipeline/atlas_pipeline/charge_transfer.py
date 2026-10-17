"""Snapshot the charge-transfer benchmark table of King, Kim, Zhong & Cheng into Atlas JSON.

Parses Table 1 (test RMSEs on the four 4G-HDNNP datasets of Ko et al. 2021: C10H2/C10H3+, Ag3+/-,
Na8/9Cl8+, Au2 on MgO(001)) from the arXiv HTML rendering of arXiv:2412.15455v1. Numbers are copied as
printed, never edited or combined; each value keeps the cutoff printed above it. The 3G- and 4G-HDNNP
values are the paper's quotations of Ko et al.; the ACE and chi+eta(ACE) values its quotations of a
charge-constrained ACE study (its ref. 41). The table caption records this and the JSON keeps it.

Usage:
  curl -sL https://arxiv.org/html/2412.15455v1 -o charges.html
  python -m atlas_pipeline.charge_transfer charges.html ../atlas/src/data/benchmarks/charge-transfer-king2025.json
"""

from __future__ import annotations

import html
import json
import re
import sys
from datetime import date
from pathlib import Path

SOURCE = {
    "paper": "king2025charges",
    "arxiv": "2412.15455v1",
    "url": "https://arxiv.org/html/2412.15455v1",
    "table": "Table 1",
    "quoted_from": {"3G-HDNNP": "ko2021fourth", "4G-HDNNP": "ko2021fourth", "ACE": "ref. 41 of the paper", "χ + η (ACE)": "ref. 41 of the paper"},
}
UNITS = {"E": "meV/atom", "F": "meV/Å"}
CAPTION = "Test root mean squared errors (RMSE)"

# Row labels as rendered (sub/superscripts flattened to spaces) → Atlas keys and display names.
SYSTEMS = {
    "C 10 H 2 /C 10 H + 3": ("c10h2", "C₁₀H₂ / C₁₀H₃⁺"),
    "Ag + / − 3": ("ag3", "Ag₃⁺ / Ag₃⁻"),
    "Na 8/9 Cl + 8": ("nacl", "Na₈/₉Cl₈⁺"),
    "Au 2 -MgO(001)": ("au2mgo", "Au₂ on MgO(001)"),
}


def _cells(row: str) -> list[str]:
    out = []
    for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S):
        c = re.sub(r"<annotation[^>]*>.*?</annotation>", " ", c, flags=re.S)
        out.append(" ".join(html.unescape(re.sub(r"<[^>]+>", " ", c)).split()))
    return out


def _num(x: str) -> float | None:
    x = x.replace("−", "-").strip()
    try:
        return float(x)
    except ValueError:
        return None


def _cut(x: str) -> float | None:
    m = re.fullmatch(r"([\d.]+)\s*Å", x.strip())
    return float(m.group(1)) if m else None


def parse(page: str) -> dict:
    i = page.find(CAPTION)
    if i < 0:
        raise ValueError(f"table caption not found: {CAPTION!r}")
    s = page.rfind("<figure", 0, i)
    e = page.find("</figure>", i)
    rows = [_cells(r) for r in re.findall(r"<tr[^>]*>(.*?)</tr>", page[s:e], re.S)]
    header = rows[0][2:]
    if not header or any(not h for h in header):
        raise ValueError(f"unexpected header: {rows[0]!r}")
    systems: list[dict] = []
    cur: dict | None = None
    cutoffs: list[float | None] = []
    cut_printed: list[str] = []
    for row in rows[1:]:
        if len(row) != len(header) + 2:
            raise ValueError(f"row has {len(row)} cells, expected {len(header) + 2}: {row!r}")
        label, kind, vals = row[0], row[1], row[2:]
        if kind == "r cut":
            cutoffs = [_cut(v) for v in vals]
            cut_printed = vals
            continue
        if label:
            if label not in SYSTEMS:
                raise ValueError(f"unknown system {label!r}")
            key, name = SYSTEMS[label]
            cur = {"key": key, "name": name, "label_as_printed": label, "models": {h: {"rcut": c, "rcut_printed": p} for h, c, p in zip(header, cutoffs, cut_printed)}}
            systems.append(cur)
        if cur is None or kind not in UNITS:
            raise ValueError(f"unexpected row {row!r}")
        for h, v in zip(header, vals):
            cur["models"][h][kind] = _num(v)
            cur["models"][h][f"{kind}_printed"] = v
    if [s["key"] for s in systems] != [k for k, _ in SYSTEMS.values()]:
        raise ValueError(f"systems found: {[s['key'] for s in systems]}")
    return {
        "source": {**SOURCE, "retrieved": date.today().isoformat()},
        "units": UNITS,
        "models": header,
        "systems": systems,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    data = parse(Path(argv[1]).read_text(encoding="utf-8"))
    out = Path(argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(data['systems'])} systems × {len(data['models'])} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
