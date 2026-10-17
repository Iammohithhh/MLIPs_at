"""Atom Intervention experiments for "MACE: Inside the Prediction" (CPU, float64).

One atom of a periodic crystal is moved along a fixed, physically interpretable axis. Every
displacement frame is a separate, complete MACE-MP-0a medium forward pass (energy, forces by
autograd, and every captured intermediate). Nothing is interpolated: the site snaps to these frames.

Outputs (written into ``inside-mace/src/data/experiments/<id>/``):

  manifest.json      geometry, displacement definition, per-frame energy / forces / site energies,
                     per-atom feature-change metrics for every atom, finite-difference force checks,
                     rotation run, provenance (checkpoint hash, versions, dtype, device, command)
  frames/<k>.json    lazy-loaded per frame: full layer features h (l = 0, 1) and message norms for the
                     "local set" (moved atom and its first two neighbour shells), and the radial basis /
                     spherical harmonics of every edge that ends on the moved atom

Usage (from inside-mace/pipeline):
  python -m inside_mace.intervention <checkpoint.model> ../../atlas/src/data/structures ../src/data/experiments
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .capture import CHECKPOINT_NAME, CHECKPOINT_SHA256, CHECKPOINT_URL, _cart, _irreps_split, _blocks, build_model, forward_with_hooks, rotation_matrix

SCHEMA_VERSION = 1
C = 128


# ---------------------------------------------------------------------------
# Experiment definitions


def si_bond_stretch() -> dict:
    return {
        "id": "si-bond-stretch",
        "title": "Silicon: push one atom along a Si–Si bond",
        "structure": "si-diamond",
        "supercell": [3, 3, 3],
        # Index chosen below: the atom nearest the supercell centre, so the scene can show its
        # whole neighbourhood without wrapping.
        "axis_label": "[111] — towards bonded neighbour",
        "displacements": [round(-0.30 + 0.025 * k, 4) for k in range(25)],  # Å; 0 is the baseline
        "local_shells": 2,
    }


EXPERIMENTS = [si_bond_stretch]
ROTATION = {"axis": [1.0, 2.0, 3.0], "angleDeg": 50.0}
FD_STEPS = [1e-2, 3e-3, 1e-3, 3e-4]  # Å, central differences


# ---------------------------------------------------------------------------


def _atoms(structure: dict, supercell):
    from ase import Atoms

    a = Atoms(symbols=structure["symbols"], positions=structure["positions"], cell=structure["cell"], pbc=structure["pbc"])
    return a.repeat(tuple(supercell))


def _mic_vectors(atoms, i: int) -> np.ndarray:
    """Minimum-image vectors from atom i to every atom (Å)."""
    from ase.geometry import find_mic

    d = atoms.positions - atoms.positions[i]
    v, _ = find_mic(d, atoms.cell, pbc=True)
    return v


def evaluate(model, atoms) -> dict:
    """One complete forward pass: outputs plus captured intermediates (float64 numpy)."""
    run = forward_with_hooks(model, atoms)
    seen, out, batch = run["seen"], run["out"], run["batch"]
    n = len(atoms)
    layers = []
    for t in range(len(model.interactions)):
        msg, _sc = seen[f"inter{t}"]
        A = _blocks(msg, 3)
        prod = seen[f"prod{t}"]
        ls = [0, 1] if prod.shape[1] == 4 * C else [0]
        H = _irreps_split(prod, C, ls)
        layers.append(
            {
                "h0": H[0][:, :, 0].copy(),
                "h1": _cart(H[1]).copy() if len(H) > 1 else None,
                "Anorm": np.stack([np.linalg.norm(b, axis=-1) for b in A], axis=1),  # [n, 4, C]
                "site": seen[f"read{t}"].detach().numpy().reshape(n, -1)[:, 0].copy(),
            }
        )
    radial = seen["radial"][0] if isinstance(seen["radial"], tuple) else seen["radial"]
    return {
        "energy": float(out["energy"].detach().numpy()[0]),
        "forces": out["forces"].detach().numpy().copy(),
        "nodeEnergy": out["node_energy"].detach().numpy().copy(),
        "e0": model.atomic_energies_fn(batch["node_attrs"]).detach().numpy().reshape(n, -1)[:, 0].copy(),
        "layers": layers,
        "edge_src": batch["edge_index"][0].numpy().copy(),
        "edge_dst": batch["edge_index"][1].numpy().copy(),
        "shifts": batch["shifts"].detach().numpy().copy(),
        "radial": radial.detach().numpy().copy(),
        "sh": seen["sh"].detach().numpy().copy(),
        "embed": seen["embed"].detach().numpy().copy(),
    }


def energy_only(model, atoms) -> float:
    return evaluate(model, atoms)["energy"]


def sig(a, s: int):
    a = np.asarray(a, dtype=float)
    out = np.where(np.abs(a) < 1e-12, 0.0, a)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(out == 0, 0, np.floor(np.log10(np.abs(out))))
    scale = 10.0 ** (s - 1 - mag)
    return (np.round(out * scale) / scale).tolist()


def fixed(a, decimals: int):
    return np.round(np.asarray(a, dtype=float), decimals).tolist()


def block_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """‖b − a‖ over the channel (and Cartesian) axes, per atom: a feature-block difference norm,
    not the difference of the two norms."""
    d = (b - a).reshape(a.shape[0], -1)
    return np.linalg.norm(d, axis=1)


def block_norm(a: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a.reshape(a.shape[0], -1), axis=1)


def run_experiment(model, spec: dict, structures_dir: Path, out_dir: Path, versions: dict, digest: str) -> dict:
    structure = json.loads((structures_dir / f"{spec['structure']}.json").read_text())
    base = _atoms(structure, spec["supercell"])
    n = len(base)
    centre = base.cell.sum(axis=0) / 2
    moved = int(np.argmin(np.linalg.norm(base.positions - centre, axis=1)))
    v = _mic_vectors(base, moved)
    dist = np.linalg.norm(v, axis=1)
    # Bonded neighbour along +[111]-type direction: the nearest neighbour whose bond vector is
    # parallel to (1,1,1) or (-1,-1,-1) — pick the one with positive sum.
    nn = [j for j in range(n) if j != moved and dist[j] < dist[dist > 1e-6].min() + 1e-3]
    partner = max(nn, key=lambda j: v[j].sum())
    axis = v[partner] / np.linalg.norm(v[partner])
    shells = sorted({round(float(d), 3) for d in dist if d > 1e-6})
    local = [moved] + [j for j in np.argsort(dist) if 1e-6 < dist[j] <= shells[spec["local_shells"] - 1] + 1e-3]
    local = [int(j) for j in local]
    r_max = float(model.r_max)

    disps = spec["displacements"]
    base_index = disps.index(0.0)
    frames, evals = [], []
    for k, d in enumerate(disps):
        atoms = base.copy()
        atoms.positions[moved] = base.positions[moved] + d * axis
        t0 = time.perf_counter()
        try:
            ev = evaluate(model, atoms)
            status, err = "ok", None
        except Exception as exc:  # recorded, never replaced by numbers
            ev, status, err = None, "failed", "".join(traceback.format_exception_only(exc)).strip()
        evals.append(ev)
        frames.append({"k": k, "d": d, "status": status, "error": err, "seconds": round(time.perf_counter() - t0, 3)})
        print(f"  frame {k:2d} d={d:+.3f} Å  {status}  {frames[-1]['seconds']} s", flush=True)

    b = evals[base_index]
    if b is None:
        raise SystemExit("baseline frame failed; nothing to compare against")

    # Baseline restoration: evaluate the untouched baseline geometry a second time.
    again = evaluate(model, base.copy())
    restoration = {
        "energyAbsDiff": abs(again["energy"] - b["energy"]),
        "forceMaxAbsDiff": float(np.abs(again["forces"] - b["forces"]).max()),
    }

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    nl = len(b["layers"])
    for fr, ev in zip(frames, evals):
        if ev is None:
            continue
        atoms_pos = base.positions.copy()
        atoms_pos[moved] = base.positions[moved] + fr["d"] * axis
        fr["energy"] = ev["energy"]  # full float64
        fr["dE"] = ev["energy"] - b["energy"]
        fr["forces"] = fixed(ev["forces"], 7)
        fr["nodeEnergy"] = fixed(ev["nodeEnergy"], 7)
        fr["site"] = [fixed(L["site"], 7) for L in ev["layers"]]
        fr["movedPosition"] = fixed(atoms_pos[moved], 7)
        # Per-atom, per-layer feature-block difference norms against the baseline frame.
        fr["change"] = [
            {
                "h0": sig(block_change(B["h0"], L["h0"]), 4),
                "h1": sig(block_change(B["h1"], L["h1"]), 4) if L["h1"] is not None else None,
            }
            for B, L in zip(b["layers"], ev["layers"])
        ]
        # Lazy chunk: full features for the local set.
        into = ev["edge_dst"] == moved
        chunk = {
            "format": "inside-mace-frame",
            "schema": SCHEMA_VERSION,
            "experiment": spec["id"],
            "k": fr["k"],
            "d": fr["d"],
            "local": local,
            "layers": [
                {
                    "h0": sig(L["h0"][local], 5),
                    "h1": sig(L["h1"][local], 5) if L["h1"] is not None else None,
                    "Anorm": sig(L["Anorm"][local], 4),
                }
                for L in ev["layers"]
            ],
            "edgesIntoMoved": {
                "src": ev["edge_src"][into].tolist(),
                "shift": fixed(ev["shifts"][into], 6),
                # MACE convention (as in capture.py): vec = r[receiver] − r[sender] + shift; receiver = moved atom.
                "vec": fixed(atoms_pos[ev["edge_dst"][into]] - atoms_pos[ev["edge_src"][into]] + ev["shifts"][into], 6),
                "radial": sig(ev["radial"][into], 5),
                "sh": sig(ev["sh"][into], 5),
            },
        }
        (frames_dir / f"{fr['k']:02d}.json").write_text(json.dumps(chunk, separators=(",", ":")) + "\n")
        fr["chunk"] = f"frames/{fr['k']:02d}.json"

    # Finite-difference force checks at the baseline and one displaced frame: F = −∂E/∂x.
    fd = []
    for k in (base_index, min(base_index + 4, len(disps) - 1)):
        if evals[k] is None:
            continue
        pos0 = base.positions.copy()
        pos0[moved] = base.positions[moved] + disps[k] * axis
        for atom, direction, label in ((moved, axis, "moved atom · axis"), (partner, np.array([1.0, 0, 0]), "bonded partner · x"), (local[-1], np.array([0, 0, 1.0]), "second-shell atom · z")):
            analytic = float(evals[k]["forces"][atom] @ direction)
            rows = []
            for h in FD_STEPS:
                ep, em = base.copy(), base.copy()
                ep.positions, em.positions = pos0.copy(), pos0.copy()
                ep.positions[atom] += h * direction
                em.positions[atom] -= h * direction
                fdv = -(energy_only(model, ep) - energy_only(model, em)) / (2 * h)
                rows.append({"h": h, "fd": fdv, "absDiff": abs(fdv - analytic)})
            fd.append({"frame": k, "atom": int(atom), "direction": fixed(direction, 9), "label": label, "analytic": analytic, "steps": rows})

    # Rotation of the model input (positions and cell together) for the baseline and one displaced frame.
    R = rotation_matrix(ROTATION["axis"], ROTATION["angleDeg"])
    rot = []
    for k in (base_index, min(base_index + 4, len(disps) - 1)):
        a = base.copy()
        a.positions[moved] = base.positions[moved] + disps[k] * axis
        ar = a.copy()
        ar.set_cell(a.cell.array @ R.T, scale_atoms=False)
        ar.positions = a.positions @ R.T
        er = evaluate(model, ar)
        e = evals[k]
        rot.append(
            {
                "frame": k,
                "energy": er["energy"],
                "energyAbsDiff": abs(er["energy"] - e["energy"]),
                "forces": fixed(er["forces"], 7),
                "forceEquivarianceMaxAbs": float(np.abs(er["forces"] - e["forces"] @ R.T).max()),
                "h0InvarianceMaxAbs": [float(np.abs(Lr["h0"] - L["h0"]).max()) for L, Lr in zip(e["layers"], er["layers"])],
                "h1EquivarianceMaxAbs": [float(np.abs(Lr["h1"] - L["h1"] @ R.T).max()) if L["h1"] is not None else None for L, Lr in zip(e["layers"], er["layers"])],
                "local": {
                    "layers": [
                        {"h0": sig(Lr["h0"][local], 5), "h1": sig(Lr["h1"][local], 5) if Lr["h1"] is not None else None}
                        for Lr in er["layers"]
                    ]
                },
            }
        )

    manifest = {
        "format": "inside-mace-experiment",
        "schema": SCHEMA_VERSION,
        "status": "PRECOMPUTED",
        "id": spec["id"],
        "title": spec["title"],
        "structure": {
            "source": f"atlas/src/data/structures/{spec['structure']}.json",
            "sourceProvenance": structure.get("provenance"),
            "supercell": spec["supercell"],
            "n": n,
            "symbols": base.get_chemical_symbols(),
            "positions": fixed(base.positions, 7),
            "cell": fixed(base.cell.array, 7),
            "pbc": [True, True, True],
            "periodicImages": "positions are never wrapped; the moved atom keeps its index in every frame; edges use the minimum-image shifts computed by the MACE neighbour list",
        },
        "displacement": {
            "atom": moved,
            "partner": int(partner),
            "axis": fixed(axis, 9),
            "axisLabel": spec["axis_label"],
            "unit": "Å",
            "values": disps,
            "baselineIndex": base_index,
            "definition": "r_moved(d) = r_moved(0) + d·û, all other atoms fixed; d > 0 shortens the bond to the partner atom",
        },
        "local": local,
        "shells": shells[:6],
        "rMax": r_max,
        "frames": frames,
        "baseline": {
            "e0": fixed(b["e0"], 7),
            "norms": [
                {"h0": sig(block_norm(L["h0"]), 4), "h1": sig(block_norm(L["h1"]), 4) if L["h1"] is not None else None}
                for L in b["layers"]
            ],
            "embed": sig(b["embed"][moved], 5),
        },
        "checks": {"restoration": restoration, "finiteDifference": fd, "rotation": {"R": fixed(R, 12), **ROTATION, "runs": rot}},
        "provenance": {
            "checkpoint": "mace-mp-0a-medium",
            "file": CHECKPOINT_NAME,
            "url": CHECKPOINT_URL,
            "sha256": digest,
            "licence": "MIT",
            "trainingData": "MPtrj (Materials Project relaxation trajectories, PBE/PBE+U)",
            **versions,
            "dtype": "float64",
            "device": "cpu",
            "layers": nl,
            "rounding": "energies full float64; forces, site energies and positions to 1e-7; features 5 significant figures (frames) and change norms 4; |x| < 1e-12 written as 0",
            "command": "cd inside-mace/pipeline && python -m inside_mace.intervention <checkpoint> ../../atlas/src/data/structures ../src/data/experiments",
            "script": "inside-mace/pipeline/inside_mace/intervention.py",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    return manifest


def main(argv: list[str]) -> int:
    ckpt, structures_dir, out_root = Path(argv[0]), Path(argv[1]), Path(argv[2])
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint sha256 {digest} != expected {CHECKPOINT_SHA256}")
    model, hyper = build_model(ckpt)
    import ase
    import e3nn
    import mace

    versions = {"mace-torch": mace.__version__, "e3nn": e3nn.__version__, "torch": torch.__version__, "ase": ase.__version__,
                "numpy": np.__version__, "python": platform.python_version(), "hyperparameters": hyper}
    index = []
    for make in EXPERIMENTS:
        spec = make()
        print(f"experiment {spec['id']}", flush=True)
        m = run_experiment(model, spec, structures_dir, out_root / spec["id"], versions, digest)
        index.append({"id": m["id"], "title": m["title"], "n": m["structure"]["n"], "frames": len(m["frames"])})
    (out_root / "index.json").write_text(json.dumps({"format": "inside-mace-index", "schema": SCHEMA_VERSION, "experiments": index}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
