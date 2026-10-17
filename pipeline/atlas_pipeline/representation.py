"""Representation Map: real MACE latent features of a reference set of atomic environments (CPU).

Every atom of a small, curated reference set (crystals ideal, rattled and compressed; surfaces; a vacancy;
molecules) is passed through three models that share one architecture:

  * MACE-MP-0a medium (2023-12-03-mace-128-L1_epoch-199.model, MIT),
  * MACE-MP-0a small  (2023-12-10-mace-128-L0_energy_epoch-249.model, MIT), same training data (MPtrj),
  * the medium architecture with random, untrained weights (torch.manual_seed(0)): a control, as the
    "Dummy" model of Li & Walsh (arXiv:2512.05349, Fig. 3).

The feature of an atom is the invariant (0e) part of its node features after each interaction, concatenated
(2 × 128 = 256 numbers for each model). Checkpoints are read with the restricted unpickler of
atlas_pipeline.xray: tensors only, nothing executed.

Analysis (pure numpy, tested in tests/test_representation.py):
  * PCA to 2D of each model's raw features (each model's own coordinates);
  * the relative ("Platonic") representation of Li & Walsh Eq. (1): cosine similarities to K anchor
    environments, the same anchors in every model, then PCA and an orthogonal Procrustes rotation onto the
    medium model's map so the maps can be compared by eye;
  * agreement between models: linear CKA of the raw features, and mutual k-nearest-neighbour overlap in the
    raw and the anchored spaces.

Exact duplicates (atoms whose environments within 6 Å are identical to rounding) are kept once, with their
multiplicity, so ties do not distort neighbour statistics.

Usage:
  python -m atlas_pipeline.representation <medium.model> <small.model> ../atlas/src/data/representation/map.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Analysis (numpy only)


def pca2(X: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Project onto the first two principal components; signs fixed so each component's largest-|loading|
    coordinate is positive. Returns (N × 2 coordinates, explained-variance fractions of PC1 and PC2)."""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    for k in range(2):
        if Vt[k, np.argmax(np.abs(Vt[k]))] < 0:
            Vt[k] *= -1
    var = S**2
    frac = (var[:2] / var.sum()).tolist() if var.sum() > 0 else [0.0, 0.0]
    return Xc @ Vt[:2].T, frac


def relative(X: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Li & Walsh Eq. (1)–(2): z_i = [cos(e_i, a_1), ..., cos(e_i, a_K)] with a_k = X[anchors[k]]."""
    A = X[anchors]
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    return Xn @ An.T


def procrustes(Y: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate/reflect (orthogonal) and scale the centred Y onto the centred ref. Returns the aligned Y and
    the Procrustes similarity 1 − m², m² the residual of the two unit-norm centred configurations."""
    Yc = Y - Y.mean(axis=0)
    Rc = ref - ref.mean(axis=0)
    ny = np.linalg.norm(Yc)
    nr = np.linalg.norm(Rc)
    U, S, Vt = np.linalg.svd((Yc / ny).T @ (Rc / nr))
    R = U @ Vt
    aligned = (Yc / ny) @ R * nr
    sim = float(S.sum() ** 2)  # = 1 − m² for unit-norm centred configurations
    return aligned, sim


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear centred kernel alignment ‖YᵀX‖²_F / (‖XᵀX‖_F ‖YᵀY‖_F) of centred features (Kornblith et al.
    2019): invariant to rotations and isotropic scaling of either space, 1 for identical geometry."""
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    num = np.linalg.norm(Yc.T @ Xc) ** 2
    return float(num / (np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)))


def knn(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest neighbours (Euclidean, excluding the point itself), ties by index."""
    sq = (X * X).sum(axis=1)
    D = sq[:, None] + sq[None, :] - 2 * X @ X.T
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1, kind="stable")[:, :k]


def mutual_knn(X: np.ndarray, Y: np.ndarray, k: int) -> float:
    """Mean fraction of shared k-nearest neighbours of each point in the two spaces."""
    a = knn(X, k)
    b = knn(Y, k)
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(X))]))


def stratified_anchors(groups: list[str], K: int, seed: int = 0) -> np.ndarray:
    """K anchor indices chosen without looking at any model: round-robin over groups (e.g. element), a
    seeded random member of each in turn. The same anchors serve every model."""
    rng = np.random.default_rng(seed)
    by: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    pools = {g: list(rng.permutation(v)) for g, v in sorted(by.items())}
    out: list[int] = []
    while len(out) < K and any(pools.values()):
        for g in sorted(pools):
            if pools[g] and len(out) < K:
                out.append(int(pools[g].pop()))
    return np.array(sorted(out))


# ---------------------------------------------------------------------------
# Reference set (ASE)

LATTICE_NOTE = "Lattice constants are inputs (rounded experimental values), not results."


def _crystals():
    from ase import Atoms
    from ase.build import bulk

    from .structures import _alpha_quartz

    def perovskite(a=3.905):
        return Atoms(
            "SrTiO3",
            scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
            cell=[a, a, a],
            pbc=True,
        )

    return [
        ("Si", "Si (diamond)", lambda: bulk("Si", "diamond", a=5.43, cubic=True)),
        ("C", "C (diamond)", lambda: bulk("C", "diamond", a=3.567, cubic=True)),
        ("Ge", "Ge (diamond)", lambda: bulk("Ge", "diamond", a=5.658, cubic=True)),
        ("Cu", "Cu (fcc)", lambda: bulk("Cu", "fcc", a=3.615, cubic=True) * (2, 1, 1)),
        ("Al", "Al (fcc)", lambda: bulk("Al", "fcc", a=4.05, cubic=True) * (2, 1, 1)),
        ("Au", "Au (fcc)", lambda: bulk("Au", "fcc", a=4.078, cubic=True) * (2, 1, 1)),
        ("Fe", "Fe (bcc)", lambda: bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)),
        ("W", "W (bcc)", lambda: bulk("W", "bcc", a=3.165, cubic=True) * (2, 2, 2)),
        ("Mg", "Mg (hcp)", lambda: bulk("Mg", "hcp", a=3.21, c=5.21) * (2, 2, 1)),
        ("Ti", "Ti (hcp)", lambda: bulk("Ti", "hcp", a=2.95, c=4.68) * (2, 2, 1)),
        ("NaCl", "NaCl (rock salt)", lambda: bulk("NaCl", "rocksalt", a=5.64, cubic=True)),
        ("KCl", "KCl (rock salt)", lambda: bulk("KCl", "rocksalt", a=6.29, cubic=True)),
        ("LiF", "LiF (rock salt)", lambda: bulk("LiF", "rocksalt", a=4.03, cubic=True)),
        ("MgO", "MgO (rock salt)", lambda: bulk("MgO", "rocksalt", a=4.21, cubic=True)),
        ("CsCl", "CsCl (B2)", lambda: bulk("CsCl", "cesiumchloride", a=4.12) * (2, 2, 2)),
        ("ZnO", "ZnO (wurtzite)", lambda: bulk("ZnO", "wurtzite", a=3.25, c=5.21) * (2, 2, 1)),
        ("GaN", "GaN (wurtzite)", lambda: bulk("GaN", "wurtzite", a=3.19, c=5.19) * (2, 2, 1)),
        ("ZnS", "ZnS (zinc blende)", lambda: bulk("ZnS", "zincblende", a=5.41, cubic=True)),
        ("GaAs", "GaAs (zinc blende)", lambda: bulk("GaAs", "zincblende", a=5.65, cubic=True)),
        ("CaF2", "CaF₂ (fluorite)", lambda: bulk("CaF2", "fluorite", a=5.46, cubic=True)),
        ("SrTiO3", "SrTiO₃ (cubic perovskite)", lambda: perovskite() * (2, 1, 1)),
        ("SiO2", "SiO₂ (α-quartz)", _alpha_quartz),
    ]


def reference_set() -> list[dict]:
    """Every structure of the reference set: id, title, kind, ASE Atoms and how it was made."""
    from ase.build import fcc111, molecule
    from ase.build import surface

    out = []
    for key, title, build in _crystals():
        ideal = build()
        out.append({"id": f"{key}-ideal", "title": title, "kind": "crystal", "variant": "ideal", "atoms": ideal})
        for sigma, seed in ((0.05, 1), (0.15, 2)):
            a = ideal.copy()
            a.rattle(stdev=sigma, seed=seed)
            out.append({"id": f"{key}-rattle{int(sigma * 100):02d}", "title": f"{title}, rattled σ = {sigma} Å", "kind": "crystal", "variant": f"rattled {sigma} Å", "atoms": a})
        c = ideal.copy()
        c.set_cell(c.cell * 0.95, scale_atoms=True)
        out.append({"id": f"{key}-compressed", "title": f"{title}, 5% compressed", "kind": "crystal", "variant": "compressed 5%", "atoms": c})

    cu = fcc111("Cu", size=(2, 2, 4), a=3.615, vacuum=8.0)
    out.append({"id": "Cu111-slab", "title": "Cu(111), 4-layer slab", "kind": "surface", "variant": "surface", "atoms": cu})
    from ase.build import bulk

    mgo = surface(bulk("MgO", "rocksalt", a=4.21, cubic=True), (1, 0, 0), layers=4, vacuum=8.0)
    out.append({"id": "MgO100-slab", "title": "MgO(100), 4-layer slab", "kind": "surface", "variant": "surface", "atoms": mgo})
    si = surface(bulk("Si", "diamond", a=5.43, cubic=True), (1, 1, 1), layers=4, vacuum=8.0)
    out.append({"id": "Si111-slab", "title": "Si(111), unreconstructed slab", "kind": "surface", "variant": "surface", "atoms": si})
    vac = bulk("Cu", "fcc", a=3.615, cubic=True) * (2, 2, 2)
    del vac[0]
    out.append({"id": "Cu-vacancy", "title": "Cu (fcc) with one vacancy", "kind": "defect", "variant": "vacancy", "atoms": vac})
    for name, title in (("H2O", "Water"), ("CH4", "Methane"), ("NH3", "Ammonia"), ("CO2", "Carbon dioxide"), ("CH3CH2OH", "Ethanol"), ("C6H6", "Benzene")):
        out.append({"id": f"mol-{name}", "title": title, "kind": "molecule", "variant": "G2 geometry", "atoms": molecule(name)})
    return out


def env_signature(atoms, i: int, rc: float = 6.0) -> tuple:
    """Element of atom i plus its sorted (element, distance) list within rc, distances to 1e−3 Å."""
    from ase.neighborlist import neighbor_list

    I, J, D = neighbor_list("ijd", atoms, rc)
    sel = I == i
    pairs = sorted((atoms.numbers[j], round(float(d), 3)) for j, d in zip(J[sel], D[sel]))
    return (int(atoms.numbers[i]), tuple(pairs))


def coordination(atoms, i: int, scale: float = 1.25) -> int:
    from ase.data import covalent_radii
    from ase.neighborlist import neighbor_list

    I, J, D = neighbor_list("ijd", atoms, 4.5)
    sel = I == i
    ri = covalent_radii[atoms.numbers[i]]
    return int(sum(d < scale * (ri + covalent_radii[atoms.numbers[j]]) for j, d in zip(J[sel], D[sel])))


# ---------------------------------------------------------------------------
# Features (torch, MACE)


def invariant_features(model, atoms) -> np.ndarray:
    """Concatenated 0e channels of the node features after every interaction (N × 256)."""
    from .xray import forward_with_hooks

    run = forward_with_hooks(model, atoms)
    seen = run["seen"]
    parts = []
    for t in range(len(model.products)):
        x = seen[f"prod{t}"].detach().numpy()
        parts.append(x[:, :128])  # e3nn orders 128x0e first
    return np.concatenate(parts, axis=1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


MODELS = [
    {"id": "medium", "name": "MACE-MP-0a medium", "checkpoint": "2023-12-03-mace-128-L1_epoch-199.model", "trained": True},
    {"id": "small", "name": "MACE-MP-0a small", "checkpoint": "2023-12-10-mace-128-L0_energy_epoch-249.model", "trained": True},
    {"id": "random", "name": "Untrained MACE (medium architecture, random weights)", "checkpoint": None, "trained": False},
]
K_VALUES = [3, 10, 30, 100]
K_NN = 10


def _r(a: np.ndarray, d: int = 3) -> list:
    return np.round(a, d).tolist()


def build(medium: Path, small: Path) -> dict:
    import torch

    from .xray import build_model

    refs = reference_set()
    built = {"medium": build_model(medium)[0], "small": build_model(small)[0]}
    torch.manual_seed(0)
    built["random"] = build_model(medium, load_weights=False)[0]

    structures, envs, seen = [], [], {}
    feats: dict[str, list[np.ndarray]] = {m["id"]: [] for m in MODELS}
    for s_idx, ref in enumerate(refs):
        atoms = ref["atoms"]
        F = {k: invariant_features(m, atoms) for k, m in built.items()}
        structures.append(
            {
                "id": ref["id"],
                "title": ref["title"],
                "kind": ref["kind"],
                "variant": ref["variant"],
                "symbols": atoms.get_chemical_symbols(),
                "positions": _r(atoms.positions),
                "cell": _r(np.array(atoms.cell)) if atoms.pbc.any() else None,
            }
        )
        for i in range(len(atoms)):
            sig = env_signature(atoms, i)
            if sig in seen:
                envs[seen[sig]]["mult"] += 1
                continue
            seen[sig] = len(envs)
            envs.append({"s": s_idx, "i": i, "el": atoms.get_chemical_symbols()[i], "cn": coordination(atoms, i), "mult": 1})
            for k in feats:
                feats[k].append(F[k][i])

    X = {k: np.array(v) for k, v in feats.items()}
    groups = [e["el"] for e in envs]
    anchors = {K: stratified_anchors(groups, K) for K in K_VALUES}
    models_out = []
    ref_maps: dict[int, np.ndarray] = {}
    for m in MODELS:
        raw, frac = pca2(X[m["id"]])
        entry = {**m, "raw": {"xy": _r(raw), "var": _r(np.array(frac), 4)}, "nn": knn(X[m["id"]], K_NN).tolist(), "anchored": {}}
        for K in K_VALUES:
            Z = relative(X[m["id"]], anchors[K])
            xy, var = pca2(Z)
            if m["id"] == "medium":
                ref_maps[K] = xy
                sim = 1.0
            else:
                xy, sim = procrustes(xy, ref_maps[K])
            entry["anchored"][str(K)] = {"xy": _r(xy), "var": _r(np.array(var), 4), "procrustes": round(sim, 4)}
        models_out.append(entry)

    pairs = []
    ids = [m["id"] for m in MODELS]
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            A, B = X[ids[a]], X[ids[b]]
            pairs.append(
                {
                    "a": ids[a],
                    "b": ids[b],
                    "cka": round(linear_cka(A, B), 4),
                    "mknn_raw": round(mutual_knn(A, B, K_NN), 4),
                    "mknn_anchored": {str(K): round(mutual_knn(relative(A, anchors[K]), relative(B, anchors[K]), K_NN), 4) for K in K_VALUES},
                }
            )

    return {
        "source": {
            "pipeline": "pipeline/atlas_pipeline/representation.py",
            "feature": "Invariant (0e) node features after each of the two interactions, concatenated (256 per atom).",
            "checkpoints": {"medium": {"name": MODELS[0]["checkpoint"], "sha256": sha256(medium)}, "small": {"name": MODELS[1]["checkpoint"], "sha256": sha256(small)}},
            "random_seed": 0,
            "method": "arXiv:2512.05349 Eqs. (1)–(2) (relative representation); PCA; orthogonal Procrustes onto the medium map",
            "anchors": "Stratified by element, seed 0, identical for every model",
            "k_nn": K_NN,
            "lattice_note": LATTICE_NOTE,
        },
        "structures": structures,
        "environments": envs,
        "anchors": {str(K): anchors[K].tolist() for K in K_VALUES},
        "models": models_out,
        "pairs": pairs,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    data = build(Path(argv[1]), Path(argv[2]))
    out = Path(argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(data['environments'])} environments from {len(data['structures'])} structures")
    for p in data["pairs"]:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
