"""Snapshot the per-model catalysis error tables of Kempen, Cheula & Andersen into Atlas JSON.

Parses Supplementary Tables S2–S15 (RMSE and maximum absolute error of every evaluated MLIP
for each catalysis target property) and Table S1's model list from the arXiv HTML rendering of
arXiv:2512.16702v1. Numbers are copied as printed; each task keeps its table locator. The Atlas
shows ranks within one task at a time and never combines tasks into an overall ranking.

Usage:
  curl -sL https://arxiv.org/html/2512.16702v1 -o catalysis.html
  python -m atlas_pipeline.catalysis catalysis.html ../atlas/src/data/benchmarks/catalysis-kempen2025.json
"""

from __future__ import annotations

import html
import json
import re
import sys
from datetime import date
from pathlib import Path

SOURCE = {"paper": "kempen2025catalysis", "arxiv": "2512.16702v1", "url": "https://arxiv.org/html/2512.16702v1"}


def _text(cell: str) -> str:
    cell = re.sub(r"<annotation[^>]*>.*?</annotation>", " ", cell, flags=re.S)
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", cell)).split())


def grid(table_html: str) -> list[list[str]]:
    """Expand an HTML table into a rectangular grid, honouring rowspan and colspan."""
    out: list[list[str]] = []
    pending: dict[tuple[int, int], str] = {}
    for r, row in enumerate(re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.S)):
        line: list[str] = []
        c = 0
        cells = re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>", row, re.S)
        k = 0
        while k < len(cells) or (r, c) in pending:
            if (r, c) in pending:
                line.append(pending.pop((r, c)))
                c += 1
                continue
            attrs, body = cells[k]
            k += 1
            rs = int((re.search(r'rowspan="(\d+)"', attrs) or [0, 1])[1])
            cs = int((re.search(r'colspan="(\d+)"', attrs) or [0, 1])[1])
            text = _text(body)
            for j in range(cs):
                line.append(text)
                for i in range(1, rs):
                    pending[(r + i, c + j)] = text
            c += cs
        out.append(line)
    return out


def _num(x: str) -> float | None:
    tok = x.replace("−", "-").split()
    try:
        return float(tok[0]) if tok else None
    except ValueError:
        return None


def _caption(fig: str) -> str:
    m = re.search(r"<figcaption.*?</figcaption>", fig, re.S)
    return _text(m.group(0)) if m else ""


TASK_RE = re.compile(r"Table (S\d+)\s*:\s*Prediction performance of the MLIPs for the (.+?)\.?$")


def parse(page: str) -> dict:
    figs = re.findall(r'<figure[^>]*class="ltx_table"[^>]*>(.*?)</figure>', page, re.S)
    tasks = []
    models: dict[str, dict] = {}
    for fig in figs:
        cap = _caption(fig)
        m = TASK_RE.match(cap)
        if not m:
            continue
        table, what = m.group(1), m.group(2)
        # Drop duplicated TeX renderings of formulas in the caption (alttext and text both appear).
        what = re.sub(r"\s*\\text\{.*$", "", what).strip().replace("​", "")
        what = re.sub(r"ZrO 2\s*\( 101\)", "ZrO₂(101)", what)
        what = what.replace("CO 2 + H → ← HCOO", "CO₂ + H ⇌ HCOO")
        g = grid(fig)
        header = g[0]
        if header[:3] != ["Model family", "", "Variant"] or "RMSE" not in header[5]:
            raise ValueError(f"{table}: unexpected header {header}")
        task = {"id": table.lower(), "table": f"Table {table}", "label": what, "rmse": {}, "maxae": {}}
        for row in g[1:]:
            if len(row) < 7:
                raise ValueError(f"{table}: short row {row}")
            family, variant, training = row[0], row[2], row[4]
            rmse, maxae = _num(row[5]), _num(row[6])
            if not family or rmse is None:
                continue
            # Task-specific baselines span the whole row and carry a reference, e.g. "SISSO [ 178 ]".
            specific = family == variant == training
            label = family
            if specific:
                family, variant, training = re.sub(r"\s*\[[^\]]*\]$", "", family), "", "task-specific"
            key = " | ".join([family, variant, training]) + (f" | {label}" if specific else "")
            models.setdefault(
                key,
                {"key": key, "family": family, "variant": variant, "training": training, "kind": "task-specific" if specific else "mlip"},
            )
            task["rmse"][key] = rmse
            task["maxae"][key] = maxae
        if not task["rmse"]:
            raise ValueError(f"{table}: no rows parsed")
        tasks.append(task)
    if len(tasks) != 14:
        raise ValueError(f"expected 14 task tables (S2–S15), found {len(tasks)}")
    return {
        "source": {**SOURCE, "retrieved": date.today().isoformat(), "tables": "Tables S2–S15"},
        "models": sorted(models.values(), key=lambda m: m["key"].lower()),
        "tasks": tasks,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    data = parse(Path(argv[1]).read_text(encoding="utf-8"))
    out = Path(argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(data['models'])} models, {len(data['tasks'])} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
