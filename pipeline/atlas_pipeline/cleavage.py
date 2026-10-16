"""Snapshot the cleavage-energy benchmark tables of Mehdizadeh & Schindler into Atlas JSON.

Parses Table 1 (models and training data) and Table 2 (error metrics and surface-stability
agreement) from the arXiv HTML rendering of arXiv:2508.21663v1. Numbers are copied as printed,
never edited or combined; the source version and table locators are stored with them.

Usage:
  curl -sL https://arxiv.org/html/2508.21663v1 -o cleavage.html
  python -m atlas_pipeline.cleavage cleavage.html ../atlas/src/data/benchmarks/cleavage-mehdizadeh2025.json
"""

from __future__ import annotations

import html
import json
import re
import sys
from datetime import date
from pathlib import Path

SOURCE = {
    "paper": "mehdizadeh2025cleavage",
    "arxiv": "2508.21663v1",
    "url": "https://arxiv.org/html/2508.21663v1",
    "tables": {"models": "Table 1", "metrics": "Table 2"},
}

METRIC_COLUMNS = [
    ("ape_mean", "%"),
    ("ape_median", "%"),
    ("mae", "meV/Å²"),
    ("rmse", "meV/Å²"),
    ("r2", ""),
    ("outliers", "structures with APE > 100%"),
    ("term", "%"),
    ("miller", "%"),
    ("none", "%"),
    ("params", ""),
    ("time_ms", "ms"),
]

# Table 1 and Table 2 name two models differently; map Table 1 names onto Table 2 names.
ALIASES = {
    "Eqf.V2-153M-OMAT24": "Eqf.V2-OMAT24",
    "Eqf.V2-153M-OAM": "Eqf.V2-OMAT24-MP-sAlex",
    "Eqf.V2-153M-OC20": "Eqf.V2-OC20",
}


def _cells(row: str) -> list[str]:
    out = []
    for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S):
        c = re.sub(r"<annotation[^>]*>.*?</annotation>", " ", c, flags=re.S)
        out.append(" ".join(html.unescape(re.sub(r"<[^>]+>", " ", c)).split()))
    return out


def _table(page: str, caption_start: str) -> list[list[str]]:
    i = page.find(caption_start)
    if i < 0:
        raise ValueError(f"table caption not found: {caption_start!r}")
    s = page.rfind("<figure", 0, i)
    e = page.find("</figure>", i)
    return [_cells(r) for r in re.findall(r"<tr[^>]*>(.*?)</tr>", page[s:e], re.S)]


def _num(x: str) -> float | None:
    x = x.replace("−", "-").strip()
    try:
        return float(x)
    except ValueError:
        return None


def data_class(description: str) -> str:
    """Coarse training-data class used only for colouring; the verbatim description is kept too."""
    d = description.lower()
    if "oc20" in d:
        return "oc20"
    if "combined" in d or ("omat24" in d and "+" in d):
        return "omat-combined"
    if "omat24" in d:
        return "omat"
    if "alexandria" in d:
        return "mp-alexandria"
    if "mptrj" in d or "materials project" in d:
        return "mp"
    return "other"


def parse(page: str) -> dict:
    t1 = _table(page, "The 19 state-of-the-art uMLIPs considered")
    models: dict[str, dict] = {}
    family = ""
    for row in t1[1:]:
        if len(row) != 4:
            continue
        family = row[0] or family
        name = ALIASES.get(row[1], row[1])
        models[name] = {"name": name, "family": family, "training": row[3], "data_class": data_class(row[3])}

    t2 = _table(page, "Comprehensive performance evaluation of 19")
    baseline = None
    seen = []
    for row in t2:
        if len(row) != 12 or _num(row[1]) is None:
            continue
        values = {}
        for (key, _unit), cell in zip(METRIC_COLUMNS, row[1:]):
            values[key] = cell if key == "params" else _num(cell)
        if row[0] == "Mean Baseline":
            baseline = values
            continue
        if row[0] not in models:
            raise ValueError(f"Table 2 model {row[0]!r} has no Table 1 entry")
        models[row[0]].update(values)
        seen.append(row[0])
    missing = sorted(set(models) - set(seen))
    if missing:
        raise ValueError(f"Table 1 models missing from Table 2: {missing}")
    return {
        "source": {**SOURCE, "retrieved": date.today().isoformat()},
        "units": dict(METRIC_COLUMNS),
        "baseline": baseline,
        "models": [models[n] for n in seen],
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    data = parse(Path(argv[1]).read_text(encoding="utf-8"))
    out = Path(argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(data['models'])} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
