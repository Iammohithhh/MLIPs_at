"""The released PACE copper potential of Lysogorskiy et al. (npj Comput. Mater. 7, 97, 2021), read and
evaluated on CPU, for the ACE page (atlas/src/pages/models/ace.astro).

The file ``Cu_npj_CompMat2021.ace`` (Zenodo 10.5281/zenodo.4734036, CC-BY-4.0; the same basis as LAMMPS'
``potentials/Cu-PBE-core-rep.ace`` apart from a header line) is a nonlinear ACE with two densities per atom,

    E_i = F_1(rho_1) + F_2(rho_2) + core repulsion,
    F_1(rho) = rho,  F_2 = FinnisSinclairShiftedScaled with m = 1/2  (sign(rho) (sqrt(x_off + |rho|) - y_off)),

where each rho_p = sum_v c~_{v,p} prod_t A_{n_t l_t m_t} is linear in its coefficients (the "ctilde" basis:
every basis function stores one coefficient per density). Two outputs:

``summary``: parsed header settings and the number of basis functions per rank (body order = rank + 1),
with the largest n and l per rank, copied from the file by ``parse_ace`` (pure Python, tested).

``curves`` (needs python-ace, the authors' evaluator, CPU wheel): the dimer energy and E(V) of fcc, bcc and
ideal hcp.
Every point is also evaluated with the potential's density-2 coefficients set to zero, with its density-1
coefficients set to zero, and with both zero, which splits the energy into rho_1, F_2(rho_2) and the core term;
the parts are checked to add up to the total.

Usage:
  curl -L -o Cu_npj_CompMat2021.ace https://zenodo.org/api/records/4734036/files/Cu_npj_CompMat2021.ace/content
  python -m atlas_pipeline.ace_pace <Cu_npj_CompMat2021.ace> ../atlas/src/data/models/ace-pace-cu.json
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

ZENODO_DOI = "10.5281/zenodo.4734036"
FILE_NAME = "Cu_npj_CompMat2021.ace"
FILE_MD5 = "584ab71fb84fd96583c310bbde8803a7"  # as listed by the Zenodo record

# ---------------------------------------------------------------------------
# Parsing (pure Python)


def _num(s: str) -> float:
    return float(s)


def parse_ace(text: str) -> dict:
    """Header settings and one record per basis function: rank, n, l, number of m-combinations and the
    coefficients of the first m-combination for each density."""
    lines = text.splitlines()
    head: dict = {}
    funcs: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ctilde_basis_func:"):
            rank = int(re.search(r"rank=(\d+)", line).group(1))
            ndens = int(re.search(r"ndens=(\d+)", line).group(1))
            n = [int(x) for x in re.search(r"n=\(([^)]*)\)", lines[i + 1]).group(1).split()]
            l = [int(x) for x in re.search(r"l=\(([^)]*)\)", lines[i + 2]).group(1).split()]
            num_ms = int(lines[i + 3].split("=")[1])
            coeffs = []
            for k in range(num_ms):
                body = lines[i + 4 + k].split(":")[1].split()
                coeffs.append([_num(x) for x in body[:ndens]])
            funcs.append({"rank": rank, "n": n, "l": l, "num_ms": num_ms, "coeffs": coeffs})
            i += 4 + num_ms
            continue
        if "=" in line and not line.startswith("crad") and not line.startswith("<"):
            k, _, v = line.partition("=")
            head[k.strip()] = v.strip()
        elif line.startswith("embedding-function:"):
            head["embedding-function"] = line.split(":", 1)[1].strip()
        elif line.startswith("4 FS parameters:"):
            head["FS parameters"] = [_num(x) for x in line.split(":", 1)[1].split()]
        elif line.startswith("core energy-cutoff parameters:"):
            head["core energy-cutoff parameters"] = [_num(x) for x in line.split(":", 1)[1].split()]
        elif line.startswith("core repulsion parameters:"):
            head["core repulsion parameters"] = [_num(x) for x in line.split(":", 1)[1].split()]
        elif line.startswith("elements:"):
            head["elements"] = line.split(":", 1)[1].split()
        elif line.startswith("E0:"):
            head["E0"] = _num(line.split(":", 1)[1])
        i += 1
    return {"header": head, "functions": funcs}


def summarise(parsed: dict) -> dict:
    h = parsed["header"]
    by_rank: dict[int, dict] = {}
    for f in parsed["functions"]:
        r = by_rank.setdefault(f["rank"], {"rank": f["rank"], "functions": 0, "n_max": 0, "l_max": 0, "ms_combinations": 0})
        r["functions"] += 1
        r["n_max"] = max(r["n_max"], max(f["n"]))
        r["l_max"] = max(r["l_max"], max(f["l"]))
        r["ms_combinations"] += f["num_ms"]
    fs = h["FS parameters"]
    return {
        "elements": h["elements"],
        "lmax": int(h["lmax"]),
        "embedding": h["embedding-function"],
        "fs_parameters": fs,
        "densities": [{"weight": fs[0], "m": fs[1]}, {"weight": fs[2], "m": fs[3]}],
        "radial_basis": h["radbasename"],
        "nradbase": int(h["nradbase"]),
        "nradmax": int(h["nradmax"]),
        "radial_parameter_lambda": float(h["radparameter"]),
        "cutoff": float(h["cutoff"]),
        "dcut": float(h["dcut"]),
        "core_repulsion": h["core repulsion parameters"],
        "core_energy_cutoff": h["core energy-cutoff parameters"],
        "rankmax": int(h["rankmax"]),
        "by_rank": [by_rank[k] for k in sorted(by_rank)],
        "total_functions": len(parsed["functions"]),
    }


def zero_density(text: str, which: tuple[int, ...]) -> str:
    """The same potential with the coefficients of the given densities (0-based) set to zero."""

    def fix(m: re.Match) -> str:
        vals = m.group(2).split()
        vals = ["0.0" if k in which else v for k, v in enumerate(vals)]
        return m.group(1) + "  " + "  ".join(vals)

    return re.sub(r"^(<[^>]*>:)\s+(.*)$", fix, text, flags=re.M)


# ---------------------------------------------------------------------------
# Closed forms (for the tests and the page): the embedding and the core repulsion


def fs_shifted_scaled(rho: float, m: float) -> float:
    """pacemaker's FinnisSinclairShiftedScaled embedding (ace_abstract_basis.cpp, FexpShiftedScaled)."""
    if abs(m - 1) < 1e-10:
        return rho
    a = abs(rho)
    e = math.exp(-a)
    nx = 1 / m
    xoff = nx ** (nx / (1 - nx)) * e
    yoff = nx ** (1 / (1 - nx)) * e
    return math.copysign(1, rho) * ((xoff + a) ** m - yoff)


def core_repulsion(r: float, pre: float, lam: float, rc: float) -> float:
    """pre exp(-lam r^2) / r with the cosine envelope (ace_radial.cpp, radcore; no inner cutoff)."""
    if r >= rc or lam * r * r >= 50:
        return 0.0
    return pre * math.exp(-lam * r * r) / r * 0.5 * (1 + math.cos(math.pi * r / rc))


# ---------------------------------------------------------------------------
# Evaluation with python-ace


def evaluate(path: Path) -> dict:
    from ase import Atoms
    from ase.build import bulk
    from pyace import PyACECalculator

    text = path.read_text()
    variants = {
        "full": text,
        "rho1": zero_density(text, (1,)),
        "rho2": zero_density(text, (0,)),
        "core": zero_density(text, (0, 1)),
    }
    calcs = {}
    tmp = Path(tempfile.mkdtemp())
    for k, t in variants.items():
        p = tmp / f"{k}.ace"
        p.write_text(t)
        calcs[k] = PyACECalculator(str(p))

    def parts(atoms) -> dict:
        out = {}
        for k, c in calcs.items():
            at = atoms.copy()
            at.calc = c
            out[k] = float(at.get_potential_energy())
        res = {"total": out["full"], "core": out["core"], "linear": out["rho1"] - out["core"], "embedding": out["rho2"] - out["core"]}
        res["closure"] = res["total"] - (res["linear"] + res["embedding"] + res["core"])
        return res

    box = 40.0
    rs = [round(0.6 + 0.05 * k, 3) for k in range(int((7.4 - 0.6) / 0.05) + 1)]
    dimer = []
    for r in rs:
        at = Atoms("Cu2", positions=[[0, 0, 0], [r, 0, 0]], cell=[box] * 3, pbc=False)
        dimer.append({"r": r, **parts(at)})

    eos = {}
    for kind in ("fcc", "bcc", "hcp"):
        pts = []
        for s in np.linspace(0.88, 1.12, 25):
            v0 = {"fcc": 3.62**3 / 4, "bcc": 2.87**3 / 2, "hcp": 3.62**3 / 4}[kind]
            v = v0 * s
            if kind == "fcc":
                at = bulk("Cu", "fcc", a=(4 * v) ** (1 / 3), cubic=True)
            elif kind == "bcc":
                at = bulk("Cu", "bcc", a=(2 * v) ** (1 / 3), cubic=True)
            else:
                a = (math.sqrt(2) * v) ** (1 / 3)  # ideal hcp: v = sqrt(3)/4 a^2 c with c = a sqrt(8/3)
                at = bulk("Cu", "hcp", a=a, c=a * math.sqrt(8 / 3))
            p = parts(at)
            n = len(at)
            pts.append({"volume": float(at.get_volume() / n), **{k: v / n for k, v in p.items()}})
        eos[kind] = pts

    closure = max(abs(p["closure"]) for p in dimer + [q for v in eos.values() for q in v])
    import pyace

    return {
        "dimer": dimer,
        "eos": eos,
        "closure_max_eV": closure,
        "python_ace_version": getattr(pyace, "__version__", "unknown"),
    }


def main(argv: list[str]) -> None:
    src, out = Path(argv[0]), Path(argv[1])
    raw = src.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != FILE_MD5:
        raise SystemExit(f"{src} has md5 {md5}, expected {FILE_MD5}")
    summary = summarise(parse_ace(raw.decode()))
    curves = evaluate(src)
    if curves["closure_max_eV"] > 1e-8:
        raise SystemExit(f"density split does not add up: {curves['closure_max_eV']}")
    data = {
        "source": {
            "file": FILE_NAME,
            "doi": ZENODO_DOI,
            "url": f"https://zenodo.org/records/4734036/files/{FILE_NAME}",
            "md5": md5,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "licence": "CC-BY-4.0",
            "evaluator": f"python-ace {curves.pop('python_ace_version')} (PyACECalculator), CPU",
            "generated": date.today().isoformat(),
        },
        "summary": summary,
        **curves,
    }
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out}: {summary['total_functions']} functions, closure {curves['closure_max_eV']:.1e} eV")


if __name__ == "__main__":
    main(sys.argv[1:])
