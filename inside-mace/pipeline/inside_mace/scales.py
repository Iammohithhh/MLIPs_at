"""Fixed colour scales for the channel heatmaps, derived from the stored frame chunks (no model run).

For each layer and block, over every evaluated frame and every atom of the local set:
  value  = max |h|               (baseline / perturbed heatmaps, symmetric scale)
  change = max |h(frame) − h(baseline)|   (difference heatmap, symmetric scale)
so a heatmap's colours mean the same thing in every frame (brief §11: stable scales).

Usage: python -m inside_mace.scales ../src/data/experiments/<id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str]) -> int:
    root = Path(argv[0])
    m = json.loads((root / "manifest.json").read_text())
    chunks = {f["k"]: json.loads((root / f["chunk"]).read_text()) for f in m["frames"] if f["status"] == "ok"}
    base = chunks[m["displacement"]["baselineIndex"]]
    out = []
    for t, B in enumerate(base["layers"]):
        entry = {}
        for block in ("h0", "h1", "Anorm"):
            if B[block] is None:
                entry[block] = None
                continue
            b = np.array(B[block])
            vals = [np.abs(np.array(c["layers"][t][block])).max() for c in chunks.values()]
            dif = [np.abs(np.array(c["layers"][t][block]) - b).max() for c in chunks.values()]
            entry[block] = {"value": float(max(vals)), "change": float(max(dif))}
        out.append(entry)
    doc = {"format": "inside-mace-scales", "schema": m["schema"], "experiment": m["id"], "layers": out,
           "definition": "max over evaluated frames and local-set atoms of |h| (value) and |h − h_baseline| (change)",
           "script": "inside-mace/pipeline/inside_mace/scales.py"}
    (root / "scales.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
