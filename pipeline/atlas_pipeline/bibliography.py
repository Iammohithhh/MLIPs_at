"""Fetch and check bibliographic metadata for every paper record.

For each atlas/content/papers/<id>.yaml with a DOI and/or arXiv id, query
Crossref, the arXiv API and OpenAlex, and write atlas/src/data/bibliography.json:

  { "<id>": { "crossref": {...}|null, "arxiv": {...}|null, "openalex": {...}|null,
              "cites": [<atlas paper ids this paper references>],
              "checks": [{"level": "ok"|"warn", "field": ..., "message": ...}] } }

The YAML stays the human-reviewed source of truth: this script never edits it.
Mismatches (title, year, first author) are reported as warnings for review.
The output is committed, so the site builds offline and deterministically.

Usage: python -m atlas_pipeline.bibliography ../atlas/content/papers ../atlas/src/data/bibliography.json [--only id,...]
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import yaml

UA = "MLIP-Atlas-bibliography/0.1 (+https://github.com/Iammohithhh/MLIPs_at)"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


def get(url: str, retries: int = 3) -> bytes | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1.5 * (attempt + 1))
    return None


def crossref(doi: str) -> dict | None:
    raw = get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    if not raw:
        return None
    m = json.loads(raw)["message"]
    issued = (m.get("issued") or {}).get("date-parts", [[None]])[0]
    return {
        "doi": m.get("DOI"),
        "title": " ".join(m.get("title") or []),
        "authors": [{"family": a.get("family", a.get("name", "")), "given": a.get("given", "")} for a in m.get("author", [])],
        "container": " ".join(m.get("container-title") or []),
        "volume": m.get("volume"),
        "issue": m.get("issue"),
        "page": m.get("page"),
        "article_number": m.get("article-number"),
        "year": issued[0] if issued else None,
        "type": m.get("type"),
    }


def arxiv(aid: str) -> dict | None:
    raw = get(f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(aid)}")
    if not raw:
        return None
    entry = ET.fromstring(raw).find(f"{ATOM}entry")
    if entry is None or entry.find(f"{ATOM}title") is None:
        return None
    text = lambda tag: " ".join((entry.findtext(tag) or "").split())  # noqa: E731
    idurl = text(f"{ATOM}id")
    prim = entry.find(f"{ARXIV}primary_category")
    return {
        "id": idurl.rsplit("/abs/", 1)[-1],
        "title": text(f"{ATOM}title"),
        "authors": [" ".join((a.findtext(f"{ATOM}name") or "").split()) for a in entry.findall(f"{ATOM}author")],
        "published": text(f"{ATOM}published")[:10],
        "updated": text(f"{ATOM}updated")[:10],
        "primary_category": prim.get("term") if prim is not None else None,
    }


def openalex(doi: str | None, aid: str | None) -> dict | None:
    key = doi or (f"10.48550/arXiv.{aid}" if aid else None)
    if not key:
        return None
    raw = get(f"https://api.openalex.org/works/doi:{urllib.parse.quote(key)}")
    if not raw:
        return None
    w = json.loads(raw)
    return {
        "id": w["id"].rsplit("/", 1)[-1],
        "publication_year": w.get("publication_year"),
        "cited_by_count": w.get("cited_by_count"),
        "referenced_works": [r.rsplit("/", 1)[-1] for r in w.get("referenced_works", [])],
    }


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split() and " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def similarity(a: str, b: str) -> float:
    ta, tb = set(norm(a).split()), set(norm(b).split())
    return len(ta & tb) / max(1, len(ta | tb))


def family_of(author: str) -> str:
    return norm(author.split(",")[0])


def check(rec: dict, cr: dict | None, ax: dict | None) -> list[dict]:
    out: list[dict] = []
    titles = [t for t in ((cr or {}).get("title"), (ax or {}).get("title")) if t]
    if titles:
        best = max(similarity(rec["title"], t) for t in titles)
        out.append(
            {"level": "ok" if best >= 0.6 else "warn", "field": "title",
             "message": f"title similarity {best:.2f} to the fetched record(s)"}
        )
    years = {y for y in ((cr or {}).get("year"), int((ax or {}).get("published", "0")[:4] or 0) or None) if y}
    if years:
        ok = rec["year"] in years
        out.append({"level": "ok" if ok else "warn", "field": "year",
                    "message": f"record year {rec['year']}; fetched {sorted(years)}"})
    firsts = []
    if cr and cr["authors"]:
        firsts.append(norm(cr["authors"][0]["family"]))
    if ax and ax["authors"]:
        firsts.append(norm(ax["authors"][0].split()[-1]))
    if firsts and rec.get("authors"):
        mine = family_of(rec["authors"][0])
        ok = any(mine == f or mine in f or f in mine for f in firsts)
        out.append({"level": "ok" if ok else "warn", "field": "authors",
                    "message": f"first author '{rec['authors'][0]}' vs fetched {firsts}"})
    if not cr and not ax:
        out.append({"level": "warn", "field": "identifiers", "message": "no metadata found for the identifiers"})
    return out


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    only = None
    if "--only" in argv:
        only = set(argv[argv.index("--only") + 1].split(","))
        args = [a for a in args if a != argv[argv.index("--only") + 1]]
    if len(args) != 2:
        print(__doc__)
        return 2
    papers_dir, out_path = Path(args[0]), Path(args[1])
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    result: dict = {}
    warnings = 0
    for f in sorted(papers_dir.glob("*.yaml")):
        pid = f.stem
        rec = yaml.safe_load(f.read_text())
        if only and pid not in only:
            if pid in existing:
                result[pid] = existing[pid]
            continue
        ids = rec.get("identifiers", {})
        doi, aid = ids.get("doi"), ids.get("arxiv")
        cr = crossref(doi) if doi else None
        ax = arxiv(str(aid)) if aid else None
        oa = openalex(doi, str(aid) if aid else None)
        checks = check(rec, cr, ax)
        warnings += sum(c["level"] == "warn" for c in checks)
        result[pid] = {"fetched": date.today().isoformat(), "crossref": cr, "arxiv": ax, "openalex": oa,
                       "cites": [], "checks": checks}
        status = "warn" if any(c["level"] == "warn" for c in checks) else "ok"
        print(f"{status:4} {pid}")
        time.sleep(0.25)
    # Citation links between Atlas papers, from OpenAlex reference lists.
    by_oa = {v["openalex"]["id"]: k for k, v in result.items() if v.get("openalex")}
    for pid, v in result.items():
        refs = set((v.get("openalex") or {}).get("referenced_works", []))
        v["cites"] = sorted(by_oa[r] for r in refs if r in by_oa and by_oa[r] != pid)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path}: {len(result)} papers, {warnings} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
