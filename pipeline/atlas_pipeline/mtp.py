"""Moment tensor potentials (MTP): basis enumeration, moments and contractions, on CPU with numpy.

Two outputs for the MTP page (atlas/src/pages/models/mtp.astro):

``templates``: the untrained MTP templates shipped with mlip-2 (gitlab.com/ashapeev/mlip-2, directory
``untrained_mtps``, levels 02 … 28) are parsed and their basis sizes copied verbatim, with the commit, into
``atlas/src/data/models/mtp-templates.json``. The site's own enumeration (src/lib/models/mtp.ts, mirrored
here) is tested against these numbers.

``si``: the Si training and test sets of Zuo et al. (J. Phys. Chem. A 2020; materialsvirtuallab/mlearn,
``data/Si``, BSD-3-Clause), PBE energies of 239 cells, are turned into configuration-level MTP basis values
b_α(cfg) = Σ_i B_α(n_i) for every basis function of level ≤ LEVEL_MAX, written to
``atlas/src/data/models/mtp-si.json``. The browser fits and selects from these (a linear MTP fitted to
energies only: the radial functions are fixed Chebyshev functions, as in Shapeev 2016, not learned).

Conventions (identical to mtp.ts):
  * M_{μ,ν} = Σ_j f_μ(r_j) r̂_j^⊗ν with f_μ = Q^μ, mlip-2's RBChebyshev Q^β(r) = T_β(ξ)(r − R_cut)².
  * A basis function is a loopless multigraph with ≤ 4 nodes (moments), node degree = ν, μ ≤ 5, ν ≤ 10.
    Its canonical key is the lexicographically smallest (μ list, upper-triangle multiplicities) over node
    relabellings, written "k|μ1,…|a01,a02,…".

Usage:
  curl -sSL https://gitlab.com/ashapeev/mlip-2/-/raw/<commit>/untrained_mtps/NN.mtp -o mtps/NN.mtp   (NN = 02 … 28)
  python -m atlas_pipeline.mtp templates mtps <commit> ../atlas/src/data/models/mtp-templates.json
  git clone --depth 1 https://github.com/materialsvirtuallab/mlearn
  python -m atlas_pipeline.mtp si mlearn/data/Si <commit> ../atlas/src/data/models/mtp-si.json
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

MAX_MOMENTS = 4
MU_MAX = 5
NU_MAX = 10

# ---------------------------------------------------------------------------
# Basis enumeration


def level(mu: int, nu: int) -> int:
    return 2 + 4 * mu + nu


def _pairs(k: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


def degrees(k: int, adj: tuple[int, ...]) -> list[int]:
    nu = [0] * k
    for (i, j), a in zip(_pairs(k), adj):
        nu[i] += a
        nu[j] += a
    return nu


def canonical(mu: tuple[int, ...], adj: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    k = len(mu)
    pairs = _pairs(k)
    A = {p: a for p, a in zip(pairs, adj)}

    def get(i: int, j: int) -> int:
        return 0 if i == j else A[(min(i, j), max(i, j))]

    best = None
    for p in itertools.permutations(range(k)):
        cand = (tuple(mu[q] for q in p), tuple(get(p[i], p[j]) for i, j in pairs))
        if best is None or cand < best:
            best = cand
    return best  # type: ignore[return-value]


def key(mu: tuple[int, ...], adj: tuple[int, ...]) -> str:
    return f"{len(mu)}|{','.join(map(str, mu))}|{','.join(map(str, adj))}"


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for c in range(total + 1):
        for rest in _compositions(total - c, parts - 1):
            yield (c,) + rest


def enumerate_basis(lev_max: int, max_moments: int = MAX_MOMENTS, mu_max: int = MU_MAX, nu_max: int = NU_MAX) -> list[dict]:
    seen: dict[str, dict] = {}
    for k in range(1, max_moments + 1):
        pairs = _pairs(k)
        E = 0
        while 2 * k + 2 * E <= lev_max:
            adjs = [()] if not pairs else list(_compositions(E, len(pairs)))
            if not pairs and E > 0:
                break
            budget = (lev_max - 2 * k - 2 * E) // 4
            for adj in adjs:
                if max(degrees(k, adj)) > nu_max:
                    continue
                for mu in itertools.product(range(min(budget, mu_max) + 1), repeat=k):
                    if sum(mu) > budget:
                        continue
                    cm, ca = canonical(mu, adj)
                    kk = key(cm, ca)
                    if kk in seen:
                        continue
                    nu = degrees(k, ca)
                    seen[kk] = {"mu": list(cm), "nu": nu, "adj": list(ca), "level": sum(level(m, n) for m, n in zip(cm, nu)), "key": kk}
            E += 1
    return sorted(seen.values(), key=lambda b: (b["level"], b["key"]))


# ---------------------------------------------------------------------------
# Moments and contractions


def chebyshev_radial(r: np.ndarray, rmin: float, rcut: float, size: int) -> np.ndarray:
    """Q^β(r), β < size, for an array of distances; zero at and beyond rcut. Shape (size, len(r))."""
    r = np.asarray(r, dtype=float)
    ksi = (2 * r - (rmin + rcut)) / (rcut - rmin)
    env = np.where(r < rcut, (r - rcut) ** 2, 0.0)
    q = np.zeros((size, r.size))
    q[0] = env
    if size > 1:
        q[1] = ksi * env
    for b in range(2, size):
        q[b] = 2 * ksi * q[b - 1] - q[b - 2]
    return q


def moment_tensor(vecs: np.ndarray, mu: int, nu: int, rmin: float, rcut: float) -> np.ndarray:
    """Σ_j f_μ(r_j) r̂_j^⊗ν as a (3,)*ν array (a scalar array for ν = 0)."""
    r = np.linalg.norm(vecs, axis=1)
    ok = (r < rcut) & (r > 0)
    r, v = r[ok], vecs[ok]
    f = chebyshev_radial(r, rmin, rcut, mu + 1)[mu]
    if nu == 0:
        return np.array(f.sum())
    u = v / r[:, None]
    acc = f[:, None] * u
    for _ in range(nu - 1):
        acc = acc[..., None] * u.reshape((u.shape[0],) + (1,) * (acc.ndim - 1) + (3,))
    return acc.sum(axis=0)


_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def einsum_spec(b: dict) -> str:
    """Subscripts contracting the graph's tensors: one letter per edge copy."""
    k = len(b["mu"])
    slots: list[list[str]] = [[] for _ in range(k)]
    e = 0
    for (i, j), a in zip(_pairs(k), b["adj"]):
        for _ in range(a):
            slots[i].append(_LETTERS[e])
            slots[j].append(_LETTERS[e])
            e += 1
    return ",".join("".join(s) for s in slots) + "->"


def basis_values(vecs: np.ndarray, basis: list[dict], rmin: float, rcut: float) -> np.ndarray:
    moments = {}
    for b in basis:
        for m, n in zip(b["mu"], b["nu"]):
            if (m, n) not in moments:
                moments[(m, n)] = moment_tensor(vecs, m, n, rmin, rcut)
    out = np.empty(len(basis))
    for t, b in enumerate(basis):
        ops = [moments[(m, n)] for m, n in zip(b["mu"], b["nu"])]
        out[t] = float(np.einsum(einsum_spec(b), *ops)) if len(ops) > 1 else float(ops[0])
    return out


# ---------------------------------------------------------------------------
# mlip-2 untrained templates


def parse_template(text: str) -> dict:
    def num(name: str) -> int:
        m = re.search(rf"^\s*{name}\s*=\s*(\d+)\s*$", text, re.M)
        if not m:
            raise ValueError(f"{name} not found")
        return int(m.group(1))

    m = re.search(r"alpha_index_basic\s*=\s*\{(.*?)\}\s*$", text, re.M)
    if not m:
        raise ValueError("alpha_index_basic not found")
    basic = [tuple(int(x) for x in g.split(",")) for g in re.findall(r"\{([^{}]*)\}", m.group(1))]
    if len(basic) != num("alpha_index_basic_count"):
        raise ValueError("alpha_index_basic has the wrong length")
    return {
        "species_count": num("species_count"),
        "radial_basis_size": num("radial_basis_size"),
        "radial_funcs_count": num("radial_funcs_count"),
        "alpha_moments_count": num("alpha_moments_count"),
        "alpha_index_basic_count": num("alpha_index_basic_count"),
        "alpha_scalar_moments": num("alpha_scalar_moments"),
        "mu_max": max(b[0] for b in basic),
        "nu_max": max(b[1] + b[2] + b[3] for b in basic),
    }


def templates(directory: Path, commit: str) -> dict:
    rows = []
    for p in sorted(directory.glob("*.mtp")):
        if not re.fullmatch(r"\d\d\.mtp", p.name):
            continue
        rows.append({"level": int(p.stem), "file": f"untrained_mtps/{p.name}", **parse_template(p.read_text())})
    if not rows:
        raise ValueError("no NN.mtp files found")
    return {
        "source": {
            "repository": "https://gitlab.com/ashapeev/mlip-2",
            "commit": commit,
            "directory": "untrained_mtps",
            "licence": "BSD-2-Clause",
            "retrieved": date.today().isoformat(),
        },
        "note": "Copied verbatim from the template headers. alpha_scalar_moments is the number of basis functions B_α (mlip-2 adds one constant function); alpha_index_basic_count the number of distinct moment components; radial_funcs_count the number of radial functions f_μ.",
        "templates": rows,
    }


# ---------------------------------------------------------------------------
# Si dataset (Zuo et al. 2020, mlearn)

LEVEL_MAX = 16
RMIN = 1.9
RCUT = 5.0


def _cell(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    s = entry["structure"]
    lat = np.array(s["lattice"]["matrix"], dtype=float)
    pos = np.array([site["xyz"] for site in s["sites"]], dtype=float)
    return lat, pos


def neighbour_vectors(lat: np.ndarray, pos: np.ndarray, rcut: float) -> list[np.ndarray]:
    """For each atom, the vectors to every periodic image within rcut (brute force over image shifts)."""
    inv = np.linalg.inv(lat)
    # number of images along each lattice vector needed to cover rcut
    heights = 1.0 / np.linalg.norm(inv, axis=0)
    n = np.ceil(rcut / heights).astype(int)
    shifts = np.array([[i, j, k] for i in range(-n[0], n[0] + 1) for j in range(-n[1], n[1] + 1) for k in range(-n[2], n[2] + 1)], dtype=float) @ lat
    out = []
    for i in range(len(pos)):
        d = (pos[None, :, :] - pos[i][None, None, :]) + shifts[:, None, :]
        d = d.reshape(-1, 3)
        r = np.linalg.norm(d, axis=1)
        out.append(d[(r > 1e-8) & (r < rcut)])
    return out


def si_dataset(data_dir: Path, commit: str, lev_max: int = LEVEL_MAX) -> dict:
    basis = enumerate_basis(lev_max)
    configs = []
    min_dist = np.inf
    for split, fname in (("train", "training.json"), ("test", "test.json")):
        for e in json.loads((data_dir / fname).read_text()):
            if e.get("element") != "Si":
                raise ValueError("expected Si only")
            lat, pos = _cell(e)
            envs = neighbour_vectors(lat, pos, RCUT)
            for v in envs:
                if len(v):
                    min_dist = min(min_dist, float(np.linalg.norm(v, axis=1).min()))
            b = sum(basis_values(v, basis, RMIN, RCUT) for v in envs)
            configs.append(
                {
                    "split": split,
                    "group": e["group"],
                    "description": e.get("description", ""),
                    "n": len(pos),
                    "energy": e["outputs"]["energy"],
                    "b": [float(f"{x:.10g}") for x in b],
                }
            )
    return {
        "source": {
            "paper": "zuo2020performance",
            "repository": "https://github.com/materialsvirtuallab/mlearn",
            "commit": commit,
            "files": ["data/Si/training.json", "data/Si/test.json"],
            "licence": "BSD-3-Clause",
            "retrieved": date.today().isoformat(),
        },
        "settings": {
            "level_max": lev_max,
            "rmin": RMIN,
            "rcut": RCUT,
            "radial": "f_mu = Q^mu, mlip-2 RBChebyshev (fixed, not learned)",
            "limits": {"max_moments": MAX_MOMENTS, "mu_max": MU_MAX, "nu_max": NU_MAX},
            "energy_unit": "eV (total energy of the cell, as stored)",
            "b": "configuration-level basis values: sum over atoms of B_alpha, in basis order",
            "min_neighbour_distance": round(min_dist, 4),
        },
        "basis": basis,
        "configs": configs,
    }


def main(argv: list[str]) -> None:
    if len(argv) != 5 or argv[1] not in ("templates", "si"):
        sys.exit(__doc__)
    cmd, src, commit, out = argv[1], Path(argv[2]), argv[3], Path(argv[4])
    data = templates(src, commit) if cmd == "templates" else si_dataset(src, commit)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv)
