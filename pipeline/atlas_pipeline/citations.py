"""Build the citation universe around the Atlas papers from OpenAlex (CC0) and lay it out in 3D.

Nodes: every Atlas paper, every work it cites, and the most-cited works that cite it. Edges: all
citations among those nodes (from each work's `referenced_works`). The layout is a seeded 3D
Fruchterman–Reingold (dense numpy), so the same input gives the same picture.

Usage:
  python -m atlas_pipeline.citations ../atlas/src/data/bibliography.json ../atlas/public/data/citation-universe.json [works-cache.json]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np

API = "https://api.openalex.org/works"
SELECT = "id,display_name,publication_year,cited_by_count,referenced_works,primary_topic,doi"
UA = "MLIP-Atlas-pipeline (citation universe)"


def _get(url: str) -> dict:
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 — retry transient network errors
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def short(wid: str) -> str:
    return wid.rsplit("/", 1)[-1]


def fetch_works(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    ids = sorted(set(ids))
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        q = urllib.parse.urlencode({"filter": "openalex_id:" + "|".join(chunk), "per-page": 50, "select": SELECT})
        for w in _get(f"{API}?{q}")["results"]:
            out[short(w["id"])] = w
        time.sleep(0.15)
    return out


def citing(wid: str, n: int) -> list[dict]:
    q = urllib.parse.urlencode({"filter": f"cites:{wid}", "sort": "cited_by_count:desc", "per-page": n, "select": SELECT})
    return _get(f"{API}?{q}")["results"]


def layout(n: int, edges: list[tuple[int, int]], seed: int = 20261016, iters: int = 220) -> np.ndarray:
    """Seeded 3D Fruchterman–Reingold on a dense distance matrix (fine for a few thousand nodes)."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3))
    adj = np.zeros((n, n), dtype=np.float32)
    for a, b in edges:
        adj[a, b] = adj[b, a] = 1.0
    k = 1.0 / np.sqrt(n) * 2.0
    t = 0.3
    for it in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta**2).sum(-1)) + 1e-6
        inv2 = 1.0 / dist**2
        np.fill_diagonal(inv2, 0.0)  # no self-repulsion (and no inf × 0 on the diagonal)
        rep = (k * k * inv2)[:, :, None] * delta
        att = (adj * dist / k)[:, :, None] * delta
        disp = (rep - att).sum(1)
        length = np.linalg.norm(disp, axis=1)[:, None] + 1e-9
        pos += disp / length * np.minimum(length, t)
        pos -= pos.mean(0)
        t *= 0.985
    return equalise(pos)


def equalise(pos: np.ndarray) -> np.ndarray:
    """Keep each node's direction from the centre but give radii a uniform-volume distribution by rank:
    a citation graph has a dense core and a sparse halo, and without this a few outliers fix the scale."""
    r = np.linalg.norm(pos, axis=1)
    order = np.argsort(np.argsort(r))
    target = ((order + 0.5) / len(r)) ** (1 / 3)
    return pos / np.maximum(r, 1e-12)[:, None] * target[:, None]


def gather(bib: dict, per_paper_citing: int = 12) -> dict[str, dict]:
    papers = bib.get("papers", bib)
    atlas = {k: v["openalex"] for k, v in papers.items() if v.get("openalex")}
    seeds = {short(v["id"]) for v in atlas.values()}
    refs = {short(r) for v in atlas.values() for r in v.get("referenced_works", [])}
    works = fetch_works(list(seeds) + list(refs))
    for wid in sorted(seeds):
        for w in citing(wid, per_paper_citing):
            works.setdefault(short(w["id"]), w)
        time.sleep(0.15)
    return works


def build(bib: dict, works: dict[str, dict], per_paper_citing: int = 12) -> dict:
    papers = bib.get("papers", bib)
    seeds = {short(v["openalex"]["id"]): k for k, v in papers.items() if v.get("openalex")}
    ids = sorted(works)
    index = {w: i for i, w in enumerate(ids)}
    edges = []
    for w in ids:
        for r in works[w].get("referenced_works") or []:
            j = index.get(short(r))
            if j is not None and j != index[w]:
                edges.append((index[w], j))
    pos = layout(len(ids), edges)
    nodes = []
    for w in ids:
        x = works[w]
        topic = (x.get("primary_topic") or {}).get("field", {}) or {}
        nodes.append(
            {
                "id": w,
                "title": x.get("display_name") or "",
                "year": x.get("publication_year"),
                "cites": x.get("cited_by_count") or 0,
                "field": topic.get("display_name"),
                "atlas": seeds.get(w),
                "doi": x.get("doi"),
                "p": [round(float(v), 4) for v in pos[index[w]]],
            }
        )
    return {
        "source": {"openalex": "https://openalex.org (CC0)", "retrieved": date.today().isoformat(), "per_paper_citing": per_paper_citing},
        "nodes": nodes,
        "edges": edges,
    }


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(__doc__)
        return 2
    bib = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    cache = Path(argv[3]) if len(argv) == 4 else None
    if cache and cache.is_file():
        works = json.loads(cache.read_text(encoding="utf-8"))
    else:
        works = gather(bib)
        if cache:
            cache.write_text(json.dumps(works), encoding="utf-8")
    data = build(bib, works)
    if any(not np.isfinite(c) for n in data["nodes"] for c in n["p"]):
        raise RuntimeError("layout produced non-finite coordinates")
    out = Path(argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(data['nodes'])} works, {len(data['edges'])} citations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
