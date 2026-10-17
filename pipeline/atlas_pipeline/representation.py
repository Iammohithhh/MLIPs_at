"""Representation Map: real latent features of six models for a reference set of atomic environments (CPU).

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

Exact duplicates (atoms whose environments within 25 Å are identical to rounding: symmetry-equivalent atoms)
are kept once, with their multiplicity, so ties do not distort neighbour statistics.

Three more architectures trained on the same data (MPtrj) are added from their own packages, each in its
own virtual environment: CHGNet 0.3.0, SevenNet-0 (11Jul2024) and Orb v2 (MPtrj only). Their features are
the per-atom vectors entering the energy readout, captured with a hook on the package's own ASE calculator.

Invariance checks: for every model, how far symmetry-equivalent atoms of one structure, and the same atom
after a rigid rotation or a mirror reflection of the structure, land from each other in feature space (Li & Walsh report that
Orb-v3 embeddings change under rotation, Section 2.4).

Usage: see USAGE below (extract per family, then merge).
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


SIG_RC = 25.0


def env_signatures(atoms, rc: float = SIG_RC) -> list[tuple]:
    """For every atom: its element plus its sorted (element, distance) list within rc, distances to 1e−3 Å.
    rc = 25 Å is beyond the receptive field of every model here except Orb v2 (15 steps × 6 Å): MACE 2 × 6 Å,
    CHGNet 4 × 6 Å, SevenNet-0 5 × 5 Å. A 6 Å signature is not enough: it merges an ideal-crystal atom with a
    bulk-like atom in the middle of a slab, which a message-passing model can tell apart."""
    from ase.neighborlist import neighbor_list

    I, J, D = neighbor_list("ijd", atoms, rc)
    out = []
    for i in range(len(atoms)):
        sel = I == i
        pairs = sorted((int(atoms.numbers[j]), round(float(d), 3)) for j, d in zip(J[sel], D[sel]))
        out.append((int(atoms.numbers[i]), tuple(pairs)))
    return out


def coordination(atoms, i: int, scale: float = 1.25) -> int:
    from ase.data import covalent_radii
    from ase.neighborlist import neighbor_list

    I, J, D = neighbor_list("ijd", atoms, 4.5)
    sel = I == i
    ri = covalent_radii[atoms.numbers[i]]
    return int(sum(d < scale * (ri + covalent_radii[atoms.numbers[j]]) for j, d in zip(J[sel], D[sel])))


# ---------------------------------------------------------------------------
# Invariance checks (numpy only)

ROT_AXIS = (1.0, 2.0, 3.0)
ROT_ANGLE_DEG = 37.0
BOX = 20.0


def rotation_matrix(axis=ROT_AXIS, angle_deg: float = ROT_ANGLE_DEG) -> np.ndarray:
    """Rodrigues rotation about `axis` by `angle_deg` (proper, det = +1)."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    t = np.radians(angle_deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def rotated(atoms):
    """The same structure rigidly rotated: positions and, for a crystal, the cell (so every interatomic
    vector is rotated and every distance is unchanged)."""
    R = rotation_matrix()
    a = atoms.copy()
    a.positions = atoms.positions @ R.T
    if atoms.pbc.any():
        a.set_cell(np.array(atoms.cell) @ R.T, scale_atoms=False)
    return a


def reflected(atoms):
    """The mirror image through the xy plane (z → −z). For a crystal the reflected cell is left-handed, so two
    of its vectors are swapped: the same lattice, described by a right-handed cell (some packages assume a
    positive cell volume). Every distance is unchanged; handedness is reversed."""
    a = atoms.copy()
    M = np.diag([1.0, 1.0, -1.0])
    a.positions = atoms.positions @ M
    if atoms.pbc.any():
        c = np.array(atoms.cell) @ M
        a.set_cell(c[[1, 0, 2]], scale_atoms=False)
        a.pbc = atoms.pbc[[1, 0, 2]]
    return a


def boxed(atoms):
    """A molecule centred in a 20 Å periodic box, for models that need a cell (as in the Observatory); a slab
    (periodic in-plane, 16 Å of vacuum between images) made periodic along z too, since Orb accepts only
    all-or-nothing periodicity. With cutoffs ≤ 6 Å, no edge crosses a vacuum gap in either case."""
    if atoms.pbc.all():
        return atoms
    if atoms.pbc.any():
        a = atoms.copy()
        a.pbc = True
        return a
    a = atoms.copy()
    a.set_cell([BOX, BOX, BOX])
    a.center()
    a.pbc = True
    return a


def nearest_distinct(X: np.ndarray) -> np.ndarray:
    """Distance from each row to its nearest other row (rows are distinct environments)."""
    sq = (X * X).sum(axis=1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1)


def invariance(rep: np.ndarray, copies: list[np.ndarray], rot: np.ndarray, ref: np.ndarray | None = None) -> dict:
    """How far a model moves an environment that should not move, in units of the median distance between
    an environment and its nearest distinct neighbour (d_ref) in the same model's raw space.

    rep:    E × d, the feature of each distinct environment (its first atom);
    copies: for each environment, the features of all its symmetry-equivalent atoms (m_e × d);
    rot:    E × d, the feature of the same first atom after a rigid rotation of the whole structure;
    ref:    E × d, the same after a mirror reflection (optional).
    """
    d_ref = float(np.median(nearest_distinct(rep)))
    spread = np.array([float(np.linalg.norm(c - r, axis=1).max()) for c, r in zip(copies, rep)]) / d_ref
    rotd = np.linalg.norm(rot - rep, axis=1) / d_ref
    multi = np.array([len(c) > 1 for c in copies])

    def summary(v: np.ndarray) -> dict:
        if len(v) == 0:
            return {"n": 0, "median": 0.0, "p95": 0.0, "max": 0.0}
        return {"n": int(len(v)), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    out = {"d_ref": d_ref, "equiv": spread, "rot": rotd, "equiv_summary": summary(spread[multi]), "rot_summary": summary(rotd)}
    if ref is not None:
        refd = np.linalg.norm(ref - rep, axis=1) / d_ref
        out.update({"ref": refd, "ref_summary": summary(refd)})
    return out


# ---------------------------------------------------------------------------
# Features (torch; one virtual environment per package, as for the Observatory)


def invariant_features(model, atoms) -> np.ndarray:
    """MACE: concatenated 0e channels of the node features after every interaction (N × 256)."""
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
    {"id": "medium", "name": "MACE-MP-0a medium", "family": "mace", "checkpoint": "2023-12-03-mace-128-L1_epoch-199.model", "trained": True, "data": "MPtrj",
     "feature": "0e node features after each of the 2 interactions, concatenated (2 × 128)"},
    {"id": "small", "name": "MACE-MP-0a small", "family": "mace", "checkpoint": "2023-12-10-mace-128-L0_energy_epoch-249.model", "trained": True, "data": "MPtrj",
     "feature": "0e node features after each of the 2 interactions, concatenated (2 × 128)"},
    {"id": "random", "name": "Untrained MACE (medium architecture, random weights)", "family": "mace", "checkpoint": None, "trained": False, "data": None,
     "feature": "0e node features after each of the 2 interactions, concatenated (2 × 128)"},
    {"id": "chgnet", "name": "CHGNet 0.3.0", "family": "chgnet", "checkpoint": "chgnet_0.3.0_e29f68s314m37.pth.tar", "trained": True, "data": "MPtrj",
     "feature": "atom features after the 4th (last) atom convolution and the readout layer norm, the input of the site-energy MLP (64)"},
    {"id": "sevennet", "name": "SevenNet-0 (11Jul2024)", "family": "sevennet", "checkpoint": "checkpoint_sevennet_0.pth", "trained": True, "data": "MPtrj",
     "feature": "0e node features after the 5th (last) interaction, the input of the energy readout (128)"},
    {"id": "orb", "name": "Orb v2 (MPtrj only)", "family": "orb", "checkpoint": "orb-mptraj-only-v2-20241014.ckpt", "trained": True, "data": "MPtrj",
     "feature": "node latents after the 15th (last) message-passing step, the input of every head (256)"},
]
OBS_KEY = {"chgnet": "chgnet-0.3.0", "sevennet": "sevennet-0", "orb": "orb-v2-mptrj"}
K_VALUES = [3, 10, 30, 100]
K_NN = 10


def _hooked_calculator(mid: str, ckpt: Path | None):
    """The package's own ASE calculator (checkpoint gated by observatory.check_checkpoint) plus a hook that
    captures the per-atom features entering the energy readout on every call."""
    from .observatory import load_calculator

    calc, path = load_calculator(OBS_KEY[mid], ckpt)
    box: dict[str, np.ndarray] = {}
    if mid == "chgnet":
        mlp = calc.model.mlp
        assert calc.model.mlp_first, "CHGNet 0.3.0 reads site energies from atom features (mlp_first)"
        mlp.register_forward_pre_hook(lambda _m, inp: box.__setitem__("x", inp[0].detach().double().numpy()))
    elif mid == "sevennet":
        layer = calc.model._modules["reduce_input_to_hidden"]
        layer.register_forward_pre_hook(lambda _m, inp: box.__setitem__("x", inp[0]["x"].detach().double().numpy()))
    elif mid == "orb":
        backbone = calc.model.model
        backbone.register_forward_hook(lambda _m, _i, out: box.__setitem__("x", out["node_features"].detach().double().numpy()))
    else:
        raise KeyError(mid)
    return calc, path, box


def _features_via(calc, box, atoms) -> np.ndarray:
    a = boxed(atoms)
    a.calc = calc
    box.clear()
    a.get_potential_energy()
    x = box["x"]
    assert x.shape[0] == len(atoms), (x.shape, len(atoms))
    return x


def _version(pkg: str) -> str:
    from importlib.metadata import version

    return version(pkg)


def extract(family: str, ckpts: list[Path]) -> dict[str, np.ndarray]:
    """Features of every atom of the reference set (and of its rotated copy) for one model family.
    Returns arrays F_<id>, Rot_<id> (all atoms of all structures, in reference-set order) and a JSON meta."""
    import torch

    refs = reference_set()
    out: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    if family == "mace":
        from .xray import build_model

        medium, small = ckpts
        built = {"medium": build_model(medium)[0], "small": build_model(small)[0]}
        torch.manual_seed(0)
        built["random"] = build_model(medium, load_weights=False)[0]
        fns = {k: (lambda a, m=m: invariant_features(m, a)) for k, m in built.items()}
        meta = {"medium": {"sha256": sha256(medium), "package": f"mace-torch {_version('mace-torch')}"},
                "small": {"sha256": sha256(small), "package": f"mace-torch {_version('mace-torch')}"},
                "random": {"sha256": None, "seed": 0, "package": f"mace-torch {_version('mace-torch')}"}}
    else:
        torch.manual_seed(0)
        calc, path, box = _hooked_calculator(family, ckpts[0] if ckpts else None)
        fns = {family: lambda a: _features_via(calc, box, a)}
        pkg = {"chgnet": "chgnet", "sevennet": "sevenn", "orb": "orb-models"}[family]
        meta = {family: {"sha256": sha256(Path(path)), "package": f"{pkg} {_version(pkg)}"}}
    for k, fn in fns.items():
        out[f"F_{k}"] = np.concatenate([fn(r["atoms"]) for r in refs])
        out[f"Rot_{k}"] = np.concatenate([fn(rotated(r["atoms"])) for r in refs])
        out[f"Ref_{k}"] = np.concatenate([fn(reflected(r["atoms"])) for r in refs])
    out["meta"] = np.array(json.dumps(meta))
    return out


# ---------------------------------------------------------------------------
# Merge and analysis


def _r(a: np.ndarray, d: int = 3) -> list:
    return np.round(a, d).tolist()


def _sig(x: float) -> float:
    """Three significant figures (the invariance numbers span many orders of magnitude)."""
    return float(f"{x:.3g}")


def merge(partials: list[Path]) -> dict:
    feats_all: dict[str, np.ndarray] = {}
    rot_all: dict[str, np.ndarray] = {}
    ref_all: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for p in partials:
        z = np.load(p)
        meta.update(json.loads(str(z["meta"])))
        for k in z.files:
            if k.startswith("F_"):
                feats_all[k[2:]] = z[k]
            elif k.startswith("Rot_"):
                rot_all[k[4:]] = z[k]
            elif k.startswith("Ref_"):
                ref_all[k[4:]] = z[k]
    models = [m for m in MODELS if m["id"] in feats_all]
    assert models and models[0]["id"] == "medium", "the medium MACE model is the reference map"

    refs = reference_set()
    structures, envs, seen = [], [], {}
    copies: list[list[int]] = []  # global atom indices of each environment's atoms
    offset = 0
    for s_idx, ref in enumerate(refs):
        atoms = ref["atoms"]
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
        sigs = env_signatures(atoms)
        for i in range(len(atoms)):
            sig = sigs[i]
            if sig in seen:
                envs[seen[sig]]["mult"] += 1
                copies[seen[sig]].append(offset + i)
                continue
            seen[sig] = len(envs)
            envs.append({"s": s_idx, "i": i, "el": atoms.get_chemical_symbols()[i], "cn": coordination(atoms, i), "mult": 1})
            copies.append([offset + i])
        offset += len(atoms)
    for k, F in feats_all.items():
        assert F.shape[0] == offset, (k, F.shape, offset)

    first = [c[0] for c in copies]
    X = {m["id"]: feats_all[m["id"]][first] for m in models}
    groups = [e["el"] for e in envs]
    anchors = {K: stratified_anchors(groups, K) for K in K_VALUES}
    models_out = []
    ref_maps: dict[int, np.ndarray] = {}
    for m in models:
        mid = m["id"]
        raw, frac = pca2(X[mid])
        inv = invariance(X[mid], [feats_all[mid][c] for c in copies], rot_all[mid][first], ref_all[mid][first])
        entry = {
            **m,
            "dim": int(X[mid].shape[1]),
            "raw": {"xy": _r(raw), "var": _r(np.array(frac), 4)},
            "nn": knn(X[mid], K_NN).tolist(),
            "anchored": {},
            "invariance": {
                "d_ref": _sig(inv["d_ref"]),
                "equiv": {k: _sig(v) if isinstance(v, float) else v for k, v in inv["equiv_summary"].items()},
                "rot": {k: _sig(v) if isinstance(v, float) else v for k, v in inv["rot_summary"].items()},
                "ref": {k: _sig(v) if isinstance(v, float) else v for k, v in inv["ref_summary"].items()},
                "equiv_env": [_sig(v) for v in inv["equiv"]],
                "rot_env": [_sig(v) for v in inv["rot"]],
                "ref_env": [_sig(v) for v in inv["ref"]],
            },
        }
        for K in K_VALUES:
            Z = relative(X[mid], anchors[K])
            xy, var = pca2(Z)
            if mid == "medium":
                ref_maps[K] = xy
                sim = 1.0
            else:
                xy, sim = procrustes(xy, ref_maps[K])
            entry["anchored"][str(K)] = {"xy": _r(xy), "var": _r(np.array(var), 4), "procrustes": round(sim, 4)}
        models_out.append(entry)

    pairs = []
    ids = [m["id"] for m in models]
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
            "feature": "Per-atom features entering each model's energy readout (see each model's 'feature').",
            "checkpoints": {m["id"]: {"name": m["checkpoint"], **meta[m["id"]]} for m in models},
            "random_seed": 0,
            "method": "arXiv:2512.05349 Eqs. (1)–(2) (relative representation); PCA; orthogonal Procrustes onto the medium map",
            "anchors": "Stratified by element, seed 0, identical for every model",
            "equivalence": f"Atoms with identical (element, distance) lists within {SIG_RC:g} Å are one environment",
            "k_nn": K_NN,
            "rotation": {"axis": list(ROT_AXIS), "angle_deg": ROT_ANGLE_DEG},
            "reflection": "z → −z (mirror through the xy plane); for crystals two cell vectors swapped to keep the cell right-handed",
            "box": f"Molecules in a {BOX:g} Å periodic box for CHGNet, SevenNet and Orb (not for MACE, which needs no cell)",
            "lattice_note": LATTICE_NOTE,
        },
        "structures": structures,
        "environments": envs,
        "anchors": {str(K): anchors[K].tolist() for K in K_VALUES},
        "models": models_out,
        "pairs": pairs,
    }


USAGE = """Usage (from pipeline/, each family in its own venv; see requirements-observatory.txt):
  python -m atlas_pipeline.representation extract mace <medium.model> <small.model> <partial.npz>
  python -m atlas_pipeline.representation extract chgnet <partial.npz>
  python -m atlas_pipeline.representation extract sevennet <partial.npz>
  python -m atlas_pipeline.representation extract orb <orb-mptraj-only-v2-20241014.ckpt> <partial.npz>
  python -m atlas_pipeline.representation merge ../atlas/src/data/representation/map.json <partial.npz>...
"""


def main(argv: list[str]) -> int:
    if len(argv) >= 4 and argv[1] == "extract":
        family, rest = argv[2], argv[3:]
        data = extract(family, [Path(p) for p in rest[:-1]])
        np.savez_compressed(rest[-1], **data)
        print(f"wrote {rest[-1]}: {sorted(k for k in data if k != 'meta')}")
        return 0
    if len(argv) >= 4 and argv[1] == "merge":
        data = merge([Path(p) for p in argv[3:]])
        out = Path(argv[2])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"wrote {out}: {len(data['environments'])} environments from {len(data['structures'])} structures")
        for m in data["models"]:
            inv = m["invariance"]
            print(m["id"], m["dim"], "equiv", inv["equiv"], "rot", inv["rot"], "ref", inv["ref"])
        for p in data["pairs"]:
            print(p["a"], p["b"], p["cka"], p["mknn_raw"], p["mknn_anchored"])
        return 0
    print(USAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
