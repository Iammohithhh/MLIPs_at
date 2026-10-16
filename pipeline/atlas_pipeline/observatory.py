"""Model Observatory: several released checkpoints run on the same inputs (CPU, precomputed).

Every model sees exactly the same atoms. The inputs are built here, deterministically,
from the Atlas structure gallery; each model runs in its own virtual environment
(their packages pin conflicting dependencies) and writes a partial file, and ``merge``
joins the partials into ``atlas/src/data/observatory/observatory.json``.

Inputs (``build_tasks``):
  gallery      every gallery structure, rattled (Gaussian, σ = 0.05 Å, seed 7) so that
               forces are not zero by symmetry; molecules sit in a 20 Å periodic box
  ev-cu/ev-si  isotropic strain of the conventional cell, lattice scale 0.94 … 1.06
  dimer-si/cu  an isolated dimer (20 Å box), 1.0 … 5.0 Å: far outside the training data
  na-au        a neutral Na atom above the fcc hollow of the Au(111) slab, 1.6 … 8 Å
  rotation     rattled ZnO 2×2×2, before and after a rigid rotation of atoms and cell
  loop         rattled Si 2×2×2 carried around a closed loop in configuration space;
               ∮ F · dx = 0 for any force that is the gradient of an energy

Checkpoint safety: before a package loads a checkpoint, ``scan_pickle_globals`` reads the
pickle opcodes (without executing them) and refuses any global outside an allow-list of
data constructors. The MACE file stores ``torch.fx`` source and fails that gate by design,
so MACE is rebuilt from tensors only with ``xray.build_model``.

Usage (from pipeline/, each with the model's own venv):
  python -m atlas_pipeline.observatory run <model> ../atlas/src/data/structures <partial.json> [--ckpt FILE]
  python -m atlas_pipeline.observatory merge <out.json> ../atlas/src/data/structures <partial.json>...
"""

from __future__ import annotations

import hashlib
import io
import json
import pickletools
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

RATTLE_SIGMA = 0.05
RATTLE_SEED = 7
BOX = 20.0

MODELS: dict[str, dict] = {
    "mace-mp-0a-medium": {
        "name": "MACE-MP-0a medium",
        "family": "mace",
        "checkpoint": "2023-12-03-mace-128-L1_epoch-199.model",
        "url": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
        "licence": "MIT",
        "package": "mace-torch",
        "dtype": "float64",
        "forces": "conservative",
    },
    "chgnet-0.3.0": {
        "name": "CHGNet 0.3.0",
        "family": "chgnet",
        "checkpoint": "chgnet_0.3.0_e29f68s314m37.pth.tar",
        "url": "https://github.com/CederGroupHub/chgnet/tree/main/chgnet/pretrained/0.3.0",
        "licence": "BSD-3-Clause",
        "package": "chgnet",
        "dtype": "float32",
        "forces": "conservative",
    },
    "sevennet-0": {
        "name": "SevenNet-0 (11Jul2024)",
        "family": "sevennet",
        "checkpoint": "checkpoint_sevennet_0.pth",
        "url": "https://github.com/MDIL-SNU/SevenNet/tree/main/sevenn/pretrained_potentials/SevenNet_0__11Jul2024",
        "licence": "GPL-3.0",
        "package": "sevenn",
        "dtype": "float32",
        "forces": "conservative",
    },
    "orb-v2-mptrj": {
        "name": "Orb v2 (MPtrj only)",
        "family": "orb",
        "checkpoint": "orb-mptraj-only-v2-20241014.ckpt",
        "url": "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-mptraj-only-v2-20241014.ckpt",
        "licence": "Apache-2.0",
        "package": "orb-models",
        "dtype": "float64",
        "forces": "direct",
    },
}

# ---------------------------------------------------------------------------
# Checkpoint gate

ALLOWED_GLOBALS = {
    "collections OrderedDict",
    "torch._utils _rebuild_tensor_v2",
    "torch._utils _rebuild_parameter",
    "torch device",
    "torch Size",
    "_codecs encode",
    "builtins set",
    "__builtin__ set",
    "numpy ndarray",
    "numpy dtype",
    "numpy.core.multiarray _reconstruct",
    "numpy.core.multiarray scalar",
    "numpy._core.multiarray _reconstruct",
    "numpy._core.multiarray scalar",
} | {f"torch {t}Storage" for t in ("Float", "Double", "Long", "Int", "Bool", "Half", "BFloat16", "Byte", "Short", "Char")}


def _pickles(path: Path) -> list[bytes]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            return [z.read(n) for n in z.namelist() if n.endswith(".pkl")]
    raw = path.read_bytes()
    out, f = [], io.BytesIO(raw)
    while f.tell() < len(raw):  # legacy format: several pickles back to back
        start = f.tell()
        for _ in pickletools.genops(f):
            pass
        out.append(raw[start : f.tell()])
        if out and len(out) > 4:
            break
    return out


def scan_pickle_globals(path: Path) -> set[str]:
    """Every global a checkpoint's pickle would import, read from its opcodes (nothing runs)."""
    found: set[str] = set()
    for data in _pickles(path):
        strings: list[str] = []
        for op, arg, _pos in pickletools.genops(data):
            if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"):
                strings.append(arg)
            elif op.name == "GLOBAL":
                found.add(str(arg))
            elif op.name == "STACK_GLOBAL":
                found.add(" ".join(strings[-2:]) if len(strings) >= 2 else "<unresolved STACK_GLOBAL>")
    return found


def check_checkpoint(path: Path) -> None:
    bad = sorted(scan_pickle_globals(path) - ALLOWED_GLOBALS)
    if bad:
        raise RuntimeError(f"{path.name}: pickle imports non-data globals {bad[:6]}; refusing to load")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Inputs


def _gallery_atoms(structure: dict):
    from ase import Atoms

    if any(structure["pbc"]):
        # Slabs are made periodic in all three directions (Orb rejects mixed periodicity);
        # the gallery slab's vacuum is wider than every model's cutoff, so nothing changes.
        return Atoms(structure["symbols"], positions=structure["positions"], cell=structure["cell"], pbc=True)
    a = Atoms(structure["symbols"], positions=structure["positions"], cell=[BOX] * 3, pbc=True)
    a.center()
    return a


def _rattled(atoms, seed: int = RATTLE_SEED):
    a = atoms.copy()
    rng = np.random.default_rng(seed)
    a.positions += rng.normal(0.0, RATTLE_SIGMA, size=a.positions.shape)
    return a


def _dimer(symbol: str, d: float):
    from ase import Atoms

    a = Atoms([symbol, symbol], positions=[[0, 0, 0], [d, 0, 0]], cell=[BOX] * 3, pbc=True)
    a.center()
    return a


def rotation_matrix(axis, angle_deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    t = np.radians(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


ROTATION = {"axis": [1.0, 2.0, 3.0], "angleDeg": 37.0}
LOOP = {"points": 48, "amplitude": 0.08, "seed": 11}
EV_SCALES = [round(0.94 + 0.01 * i, 2) for i in range(13)]
DIMER_D = [round(1.0 + 0.1 * i, 1) for i in range(41)]
NA_HEIGHTS = [round(1.6 + 0.2 * i, 1) for i in range(33)]


def loop_path(n_atoms: int):
    """A closed loop x(θ) = x0 + a[(cos θ − 1) u + sin θ v] with random unit directions u ⟂ v in R^{3N}."""
    rng = np.random.default_rng(LOOP["seed"])
    u = rng.normal(size=(n_atoms, 3))
    u /= np.linalg.norm(u)
    v = rng.normal(size=(n_atoms, 3))
    v -= (v * u).sum() * u
    v /= np.linalg.norm(v)
    th = 2 * np.pi * np.arange(LOOP["points"]) / LOOP["points"]
    a = LOOP["amplitude"]
    disp = [a * ((np.cos(t) - 1) * u + np.sin(t) * v) for t in th]
    tangent = [a * (-np.sin(t) * u + np.cos(t) * v) for t in th]  # dx/dθ
    return th, disp, tangent


def build_tasks(structures_dir: Path) -> dict[str, dict]:
    """Every input frame, keyed by task. Identical for every model."""
    gal = json.loads((structures_dir / "index.json").read_text())
    S = {g["id"]: json.loads((structures_dir / f"{g['id']}.json").read_text()) for g in gal}
    tasks: dict[str, dict] = {}

    tasks["gallery"] = {"frames": [], "ids": []}
    for g in gal:
        tasks["gallery"]["frames"].append(_rattled(_gallery_atoms(S[g["id"]])))
        tasks["gallery"]["ids"].append(g["id"])

    for sid, key in (("cu-fcc", "ev-cu"), ("si-diamond", "ev-si")):
        base = _gallery_atoms(S[sid])
        frames = []
        for s in EV_SCALES:
            a = base.copy()
            a.set_cell(base.cell * s, scale_atoms=True)
            frames.append(a)
        tasks[key] = {"frames": frames, "x": EV_SCALES, "structure": sid}

    for el, key in (("Si", "dimer-si"), ("Cu", "dimer-cu")):
        tasks[key] = {"frames": [_dimer(el, d) for d in DIMER_D], "x": DIMER_D}

    slab = _gallery_atoms(S["au111-slab"])
    z = slab.positions[:, 2]
    top = np.where(z > z.max() - 0.5)[0]
    sub = np.where((z < z.max() - 0.5) & (z > z.max() - 3.0))[0]
    third = np.where((z < z.max() - 3.0) & (z > z.max() - 5.5))[0]
    # fcc hollow: above a third-layer atom, inside a triangle of top-layer atoms.
    cx = slab.positions[third[0], :2]
    na_frames = []
    for h in NA_HEIGHTS:
        a = slab.copy()
        a.append("Na")
        a.positions[-1] = [cx[0], cx[1], z.max() + h]
        na_frames.append(a)
    tasks["na-au"] = {"frames": na_frames, "x": NA_HEIGHTS, "site": "fcc", "layers": [len(top), len(sub), len(third)]}

    zno = _rattled(_gallery_atoms(S["zno-wurtzite"]).repeat((2, 2, 2)))
    R = rotation_matrix(ROTATION["axis"], ROTATION["angleDeg"])
    rot = zno.copy()
    rot.set_cell(zno.cell.array @ R.T, scale_atoms=False)
    rot.positions = zno.positions @ R.T
    tasks["rotation"] = {"frames": [zno, rot], "R": R.tolist()}

    si = _rattled(_gallery_atoms(S["si-diamond"]).repeat((2, 2, 2)), seed=RATTLE_SEED + 1)
    th, disp, tangent = loop_path(len(si))
    frames = []
    for d in disp:
        a = si.copy()
        a.positions = si.positions + d
        frames.append(a)
    tasks["loop"] = {"frames": frames, "theta": th.tolist(), "tangent": tangent}
    return tasks


def frame_record(atoms) -> dict:
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": _r(atoms.positions, 6),
        "cell": _r(atoms.cell.array, 6),
        "pbc": [bool(p) for p in atoms.pbc],
    }


def _r(a, sig: int = 7):
    a = np.asarray(a, dtype=float)
    out = np.array([float(f"{x:.{sig}g}") for x in a.ravel()]).reshape(a.shape)
    return out.tolist()


# ---------------------------------------------------------------------------
# Models


def load_calculator(key: str, ckpt: Path | None):
    m = MODELS[key]
    if key == "mace-mp-0a-medium":
        from mace.calculators import MACECalculator

        from .xray import CHECKPOINT_SHA256, build_model

        assert ckpt is not None and sha256(ckpt) == CHECKPOINT_SHA256
        model, _ = build_model(ckpt)
        calc = MACECalculator(models=model, device="cpu", default_dtype="float64")
        return calc, ckpt
    if key == "chgnet-0.3.0":
        import chgnet
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator

        path = Path(chgnet.__file__).parent / "pretrained" / "0.3.0" / m["checkpoint"]
        check_checkpoint(path)
        return CHGNetCalculator(CHGNet.load(model_name="0.3.0", use_device="cpu", verbose=False), use_device="cpu"), path
    if key == "sevennet-0":
        import sevenn
        from sevenn.calculator import SevenNetCalculator

        path = Path(sevenn.__file__).parent / "pretrained_potentials" / "SevenNet_0__11Jul2024" / m["checkpoint"]
        check_checkpoint(path)
        return SevenNetCalculator(str(path), device="cpu"), path
    if key == "orb-v2-mptrj":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator

        assert ckpt is not None
        check_checkpoint(ckpt)
        model, adapter = pretrained.orb_mptraj_only_v2(weights_path=str(ckpt), device="cpu", precision="float64", compile=False)
        calc = ORBCalculator(model, adapter, device="cpu")
        if calc.conservative != (m["forces"] == "conservative"):
            raise RuntimeError("orb-models reports a different force head than recorded")
        return calc, ckpt
    raise KeyError(key)


def _evaluate(calc, atoms) -> dict:
    a = atoms.copy()
    a.calc = calc
    e = float(a.get_potential_energy())
    f = np.asarray(a.get_forces(), dtype=float)
    out = {"energy": e, "forces": f}
    if all(a.pbc):
        try:
            out["stress"] = np.asarray(a.get_stress(voigt=True), dtype=float)  # eV/Å³, ASE sign
        except Exception:  # noqa: BLE001 - a model without stress simply has none
            pass
    return out


def run(key: str, structures_dir: Path, out: Path, ckpt: Path | None) -> None:
    import ase
    import torch

    torch.manual_seed(0)
    calc, path = load_calculator(key, ckpt)
    tasks = build_tasks(structures_dir)
    t0 = time.time()
    res: dict[str, dict] = {}

    g = tasks["gallery"]
    res["gallery"] = {}
    for sid, a in zip(g["ids"], g["frames"]):
        r = _evaluate(calc, a)
        res["gallery"][sid] = {"energy": r["energy"], "forces": _r(r["forces"])}
        if "stress" in r:
            res["gallery"][sid]["stress"] = _r(r["stress"])

    for key2 in ("ev-cu", "ev-si", "dimer-si", "dimer-cu"):
        ev = [_evaluate(calc, a) for a in tasks[key2]["frames"]]
        res[key2] = {"energy": [r["energy"] for r in ev]}
        if key2.startswith("dimer"):
            res[key2]["force"] = [float(r["forces"][1, 0]) for r in ev]  # on atom 2, along the bond

    na = [_evaluate(calc, a) for a in tasks["na-au"]["frames"]]
    res["na-au"] = {"energy": [r["energy"] for r in na], "fz": [float(r["forces"][-1, 2]) for r in na]}

    R = np.asarray(tasks["rotation"]["R"])
    r0, r1 = (_evaluate(calc, a) for a in tasks["rotation"]["frames"])
    res["rotation"] = {"energy": [r0["energy"], r1["energy"]], "forces": [_r(r0["forces"]), _r(r1["forces"])]}

    lp = tasks["loop"]
    proj, energies = [], []
    for a, t in zip(lp["frames"], lp["tangent"]):
        r = _evaluate(calc, a)
        energies.append(r["energy"])
        proj.append(float((r["forces"] * t).sum()))  # F · dx/dθ
    res["loop"] = {"energy": energies, "power": proj}
    _ = R

    m = MODELS[key]
    pkg = m["package"]
    from importlib.metadata import version

    record = {
        "key": key,
        **{k: m[k] for k in ("name", "family", "checkpoint", "url", "licence", "dtype", "forces")},
        "sha256": sha256(path),
        "versions": {pkg: version(pkg), "torch": torch.__version__, "ase": ase.__version__, "numpy": np.__version__},
        "cutoff": _cutoff(calc, key),
        "neighbours": _max_neighbours(calc, key),
        "seconds": round(time.time() - t0, 1),
        "results": res,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, separators=(",", ":")))
    print(f"{key}: {record['seconds']} s, {path.name}")


def _cutoff(calc, key: str) -> float | None:
    try:
        if key == "mace-mp-0a-medium":
            return float(calc.models[0].r_max)
        if key == "chgnet-0.3.0":
            return float(calc.model.graph_converter.atom_graph_cutoff)
        if key == "sevennet-0":
            return float(calc.cutoff)
        if key == "orb-v2-mptrj":
            return float(calc.adapter.radius)
    except Exception:  # noqa: BLE001
        return None
    return None


def _max_neighbours(calc, key: str) -> int | None:
    """A cap on neighbours per atom, if the model's graph has one (Orb v2: nearest 20 within 6 Å)."""
    if key == "orb-v2-mptrj":
        return int(calc.adapter.max_num_neighbors)
    return None


def merge(out: Path, partials: list[Path], structures_dir: Path) -> None:
    tasks = build_tasks(structures_dir)
    models = [json.loads(p.read_text()) for p in partials]
    order = list(MODELS)
    models.sort(key=lambda m: order.index(m["key"]))
    lp = tasks["loop"]
    doc = {
        "format": "atlas-observatory",
        "version": 1,
        "settings": {
            "rattle": {"sigma": RATTLE_SIGMA, "seed": RATTLE_SEED},
            "box": BOX,
            "rotation": ROTATION,
            "loop": LOOP,
            "units": {"energy": "eV", "forces": "eV/Å", "stress": "eV/Å³ (ASE sign: positive = tensile)", "length": "Å"},
        },
        "inputs": {
            "gallery": [{"id": sid, **frame_record(a)} for sid, a in zip(tasks["gallery"]["ids"], tasks["gallery"]["frames"])],
            "ev-cu": {"x": tasks["ev-cu"]["x"], "atoms": len(tasks["ev-cu"]["frames"][0]), "volume0": tasks["ev-cu"]["frames"][6].get_volume()},
            "ev-si": {"x": tasks["ev-si"]["x"], "atoms": len(tasks["ev-si"]["frames"][0]), "volume0": tasks["ev-si"]["frames"][6].get_volume()},
            "dimer-si": {"x": tasks["dimer-si"]["x"]},
            "dimer-cu": {"x": tasks["dimer-cu"]["x"]},
            "na-au": {"x": tasks["na-au"]["x"], "site": "fcc", "frame": frame_record(tasks["na-au"]["frames"][7])},
            "rotation": {"R": tasks["rotation"]["R"], "frames": [frame_record(a) for a in tasks["rotation"]["frames"]]},
            "loop": {"theta": lp["theta"], "atoms": len(lp["frames"][0])},
        },
        "models": models,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB, {len(models)} models)")


def main(argv: list[str]) -> int:
    if len(argv) >= 4 and argv[0] == "run":
        ckpt = Path(argv[argv.index("--ckpt") + 1]) if "--ckpt" in argv else None
        run(argv[1], Path(argv[2]), Path(argv[3]), ckpt)
        return 0
    if len(argv) >= 4 and argv[0] == "merge":
        merge(Path(argv[1]), [Path(p) for p in argv[3:]], Path(argv[2]))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
