"""Architecture X-ray for a long-range model: MACELES-OFF (MACE + Latent Ewald Summation), on CPU.

MACELES-OFF (Kim et al., JCTC 2025, arXiv:2507.14302) is a two-layer MACE whose two readouts
each have a twin that predicts a latent charge instead of a site energy. The charges enter a
Gaussian-smeared Coulomb sum (LES; Cheng, npj Comput. Mater. 2025) that is added to the local
energy:

    E = sum_i (E0_i + s * (e1_i + e2_i) + mu)  +  E_lr[q],    q_i = q1_i + q2_i

Safety: the released ``MACELES-OFF_small_converted.model`` is a pickled ``torch.nn.Module``. It is
read with the restricted unpickler of ``xray.py`` (tensors, plain numbers and inert stubs; no
stored class or function is ever called), the MACE part is rebuilt with the installed
``mace-torch`` and loaded strictly, and the two charge readouts are rebuilt with mace's own
readout blocks.

The LES step itself is re-implemented here (``les_energy``) rather than imported: for a
non-periodic system it is the pairwise sum

    E_lr = (n_f / 2 pi) * 1/2 * sum_{i != j} q_i q_j erf(r_ij / (sqrt(2) sigma)) / r_ij

(Kim et al. arXiv v1 Eq. 3; LES ``Ewald.compute_potential_realspace``), with sigma and the
prefactor n_f read from the checkpoint's stored ``les.ewald`` attributes. It is a few lines,
tested against closed forms in ``tests/test_xray_les.py``.

Inputs besides the checkpoint: the Atlas ethanol structure, the S22 water dimer from ASE, and
the les_fit ``data-spice-test-bec`` test set (DFT energies, forces, dipoles and Born effective
charges; CC BY-NC 4.0), which validates the rebuild.

Usage:
  python -m atlas_pipeline.xray_les <MACELES-OFF_small_converted.model> <les_fit/data-spice-test-bec> \
      ../atlas/src/data/structures ../atlas/src/data/xray/maceles-off.json
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from atlas_pipeline.xray import _batch, _pickle_module, _r, _state, _flat_tensors, rotation_matrix

CHECKPOINT_NAME = "MACELES-OFF_small_converted.model"
CHECKPOINT_URL = "https://github.com/ChengUCB/les_fit/blob/main/MACELES-OFF/" + CHECKPOINT_NAME
CHECKPOINT_MD5 = "cc10e937b55e09f05b16dba756e2311b"  # given in the les_fit README
LES_FIT_COMMIT = "a886785caa4182a0effb95d95f0e402881adfc4d"
TEST_SUBSETS = ["pubchem", "des370k_monomers", "des370k_dimers", "dipeptides", "solvated_amino_acids", "water", "qmugs"]

# 1/(2 eps0) in eV Å / e^2 (CODATA 2018: e^2/(4 pi eps0) = 14.3996 eV Å, times 2 pi)
HALF_INV_EPS0 = 2 * math.pi * 14.399645


# ---------------------------------------------------------------------------
# Reading and rebuilding


def read_tree(path: Path) -> tuple[dict, dict]:
    """Tensors plus the plain hyperparameters stored on the module tree (never executed)."""
    tree = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_pickle_module)
    tensors = _flat_tensors(tree)
    mods = _state(tree)["_modules"]
    inter0 = _state(_state(mods["interactions"])["_modules"]["0"])
    les = _state(mods["les"])
    ewald = _state(les["_modules"]["ewald"])
    bec = _state(les["_modules"]["bec"])
    extra = {
        "avg_num_neighbors": float(inter0["avg_num_neighbors"]),
        "radial_MLP": list(inter0["radial_MLP"]),
        "les": {
            "sigma": float(ewald["sigma"]),
            "dl": float(ewald["dl"]),
            "norm_factor": float(ewald["norm_factor"]),
            "remove_self_interaction": bool(ewald["remove_self_interaction"]),
            "use_atomwise": bool(les["use_atomwise"]),
            "bec_remove_mean": bool(bec["remove_mean"]),
            "epsilon_factor": float(bec["epsilon_factor"]),
        },
    }
    return tensors, extra


def build_model(path: Path):
    from e3nn import o3
    from mace import modules
    from mace.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock

    torch.set_default_dtype(torch.float64)
    sd, extra = read_tree(path)
    C = int(sd["les_readouts.0.linear.weight"].numel())  # 192: one weight per scalar channel
    mlp = int(sd["readouts.1.linear_2.weight"].numel())  # 16
    hidden = f"{C}x0e+{C}x1o"
    block = modules.interaction_classes["RealAgnosticResidualInteractionBlock"]
    hyper = {
        "r_max": float(sd["r_max"]),
        "num_bessel": 8,
        "num_polynomial_cutoff": int(sd["radial_embedding.cutoff_fn.p"]),
        "max_ell": 3,
        "num_interactions": int(sd["num_interactions"]),
        "hidden_irreps": hidden,
        "MLP_irreps": f"{mlp}x0e",
        "correlation": 3,
        "avg_num_neighbors": extra["avg_num_neighbors"],
        "radial_MLP": extra["radial_MLP"],
        "atomic_inter_scale": float(sd["scale_shift.scale"].reshape(())),
        "atomic_inter_shift": float(sd["scale_shift.shift"]),
        "atomic_numbers": sd["atomic_numbers"].tolist(),
        "les": extra["les"],
    }
    if sd["radial_embedding.bessel_fn.bessel_weights"].numel() != hyper["num_bessel"]:
        raise RuntimeError("unexpected number of Bessel functions")
    model = modules.ScaleShiftMACE(
        r_max=hyper["r_max"],
        num_bessel=hyper["num_bessel"],
        num_polynomial_cutoff=hyper["num_polynomial_cutoff"],
        max_ell=hyper["max_ell"],
        interaction_cls=block,
        interaction_cls_first=block,
        num_interactions=hyper["num_interactions"],
        num_elements=len(hyper["atomic_numbers"]),
        hidden_irreps=o3.Irreps(hidden),
        MLP_irreps=o3.Irreps(hyper["MLP_irreps"]),
        atomic_energies=sd["atomic_energies_fn.atomic_energies"].numpy(),
        avg_num_neighbors=hyper["avg_num_neighbors"],
        atomic_numbers=hyper["atomic_numbers"],
        correlation=hyper["correlation"],
        gate=torch.nn.functional.silu,
        radial_MLP=hyper["radial_MLP"],
        atomic_inter_scale=hyper["atomic_inter_scale"],
        atomic_inter_shift=hyper["atomic_inter_shift"],
    )
    own = model.state_dict()
    base = {k: v for k, v in sd.items() if not k.startswith("les")}
    base["scale_shift.scale"] = base["scale_shift.scale"].reshape(own["scale_shift.scale"].shape)
    unexpected = sorted(set(base) - set(own))
    if unexpected:
        raise RuntimeError(f"checkpoint has tensors the rebuilt model lacks: {unexpected[:5]}")
    for k in sorted(set(own) - set(base)):
        if "._w3j_" in k:  # Wigner-3j constants the fresh model regenerates
            continue
        if not k.endswith("_zeroed") or bool(own[k]):
            raise RuntimeError(f"rebuilt model needs a tensor the checkpoint lacks: {k}")
    for k, v in base.items():
        if own[k].shape != v.shape:
            raise RuntimeError(f"shape mismatch at {k}: {tuple(own[k].shape)} vs {tuple(v.shape)}")
    model.load_state_dict(base, strict=False)
    model.eval()

    irreps = o3.Irreps(hidden)
    charge_readouts = torch.nn.ModuleList(
        [
            LinearReadoutBlock(irreps_in=irreps, irrep_out=o3.Irreps("0e")),
            NonLinearReadoutBlock(
                irreps_in=o3.Irreps(f"{C}x0e"), MLP_irreps=o3.Irreps(f"{mlp}x0e"), gate=torch.nn.functional.silu, irrep_out=o3.Irreps("0e")
            ),
        ]
    )
    lsd = {k[len("les_readouts.") :]: v for k, v in sd.items() if k.startswith("les_readouts.")}
    charge_readouts.load_state_dict(lsd, strict=True)
    charge_readouts.eval()
    return model, charge_readouts, hyper


# ---------------------------------------------------------------------------
# LES for a finite system


def les_kernel(r: torch.Tensor, sigma: float) -> torch.Tensor:
    """erf(r / (sqrt 2 sigma)) / r: the interaction of two unit Gaussian charges, finite at r = 0."""
    return torch.special.erf(r / (math.sqrt(2.0) * sigma)) / r


def les_potential(q: torch.Tensor, pos: torch.Tensor, sigma: float, norm_factor: float) -> torch.Tensor:
    """phi_i = (n_f / 2 pi) sum_{j != i} q_j erf(r_ij / sqrt2 sigma) / r_ij, the latent potential at atom i."""
    n = pos.shape[0]
    d = pos[:, None, :] - pos[None, :, :]
    eye = torch.eye(n, dtype=pos.dtype)
    r = torch.sqrt((d**2).sum(-1) + eye)  # the diagonal is masked below; +1 keeps it finite
    K = les_kernel(r, sigma) * (1 - eye)
    return (norm_factor / (2 * math.pi)) * (K @ q)


def les_energy(q: torch.Tensor, pos: torch.Tensor, sigma: float, norm_factor: float) -> torch.Tensor:
    """E_lr = 1/2 sum_i q_i phi_i (self-interaction removed, as in LES)."""
    return 0.5 * (q * les_potential(q, pos, sigma, norm_factor)).sum()


def polarization(q: torch.Tensor, pos: torch.Tensor, remove_mean: bool = True) -> torch.Tensor:
    """P = sum_i (q_i - mean q) r_i: LES's dipole of a finite system (Kim et al. Eq. 4)."""
    qc = q - q.mean() if remove_mean else q
    return (qc[:, None] * pos).sum(0)


# ---------------------------------------------------------------------------
# One forward pass, every step kept


def run(model, readouts, hyper: dict, atoms) -> dict:
    les = hyper["les"]
    batch = _batch(model, atoms)
    pos = batch["positions"].requires_grad_(True)
    seen: dict[str, object] = {}
    handles = []
    for t, (prod, read) in enumerate(zip(model.products, model.readouts)):
        handles.append(prod.register_forward_hook(lambda _m, _i, o, t=t: seen.__setitem__(f"h{t}", o)))
        handles.append(read.register_forward_hook(lambda _m, _i, o, t=t: seen.__setitem__(f"e{t}", o)))
    try:
        out = model(batch, training=False, compute_force=False)
    finally:
        for h in handles:
            h.remove()
    n = len(atoms)
    q_layers = [readouts[t](seen[f"h{t}"]).reshape(n) for t in range(2)]
    q = q_layers[0] + q_layers[1]
    e_sr = out["energy"][0]
    e_lr = les_energy(q, pos, les["sigma"], les["norm_factor"])
    phi = les_potential(q, pos, les["sigma"], les["norm_factor"])
    f_sr = -torch.autograd.grad(e_sr, pos, retain_graph=True)[0]
    f_lr = -torch.autograd.grad(e_lr, pos, retain_graph=True)[0]
    P = polarization(q, pos, les["bec_remove_mean"])
    Z = torch.stack([torch.autograd.grad(P[a], pos, retain_graph=True)[0] for a in range(3)], 1)  # [n, alpha, beta]
    s = hyper["atomic_inter_scale"]
    return {
        "batch": batch,
        "h": [seen["h0"].detach().numpy(), seen["h1"].detach().numpy()],
        "siteLayers": [(s * seen[f"e{t}"].reshape(n)).detach().numpy() for t in range(2)],
        "qLayers": [x.detach().numpy() for x in q_layers],
        "q": q.detach().numpy(),
        "phi": phi.detach().numpy(),
        "eSr": float(e_sr),
        "eLr": float(e_lr),
        "nodeEnergy": out["node_energy"].detach().numpy(),
        "fSr": f_sr.numpy(),
        "fLr": f_lr.numpy(),
        "P": P.detach().numpy(),
        "Z": Z.detach().numpy(),
    }


def _edges(batch) -> dict:
    ei = batch["edge_index"].numpy()
    pos = batch["positions"].detach().numpy()
    vec = pos[ei[1]] - pos[ei[0]] + batch["shifts"].detach().numpy()
    return {"count": int(ei.shape[1]), "src": ei[0].tolist(), "dst": ei[1].tolist(), "vec": _r(vec)}


def summarise(res: dict, atoms) -> dict:
    C = res["h"][1].shape[1]
    h1 = res["h"][0]  # [n, C + 3C]: C scalars then C vectors (e3nn mul-major)
    return {
        "n": len(atoms),
        "edges": _edges(res["batch"]),
        "features": {
            "h1Scalar": _r(np.linalg.norm(h1[:, :C], axis=1), 4),
            "h1Vector": _r(np.linalg.norm(h1[:, C:], axis=1), 4),
            "h2Scalar": _r(np.linalg.norm(res["h"][1], axis=1), 4),
            "shapes": [list(res["h"][0].shape), list(res["h"][1].shape)],
        },
        "siteLayers": [_r(x) for x in res["siteLayers"]],
        "qLayers": [_r(x) for x in res["qLayers"]],
        "q": _r(res["q"]),
        "phi": _r(res["phi"]),
        "eSr": res["eSr"],
        "eLr": res["eLr"],
        "nodeEnergy": _r(res["nodeEnergy"], 9),
        "fSr": _r(res["fSr"]),
        "fLr": _r(res["fLr"]),
        "P": _r(res["P"]),
        "Z": _r(res["Z"], 4),
    }


# ---------------------------------------------------------------------------
# Systems


def _atoms(symbols, positions):
    from ase import Atoms

    return Atoms(symbols=symbols, positions=positions, pbc=False)


def water_dimer_scan(model, readouts, hyper, distances) -> dict:
    """Rigidly pull the S22 water dimer apart along the O···O axis."""
    from ase.collections import s22

    base = s22["Water_dimer"]
    pos0 = base.positions.copy()
    axis = pos0[3] - pos0[0]
    d0 = float(np.linalg.norm(axis))
    axis /= d0
    syms = base.get_chemical_symbols()
    mono = [run(model, readouts, hyper, _atoms(syms[k : k + 3], pos0[k : k + 3])) for k in (0, 3)]
    rows = []
    for d in distances:
        p = pos0.copy()
        p[3:] += (d - d0) * axis
        res = run(model, readouts, hyper, _atoms(syms, p))
        gap = float(np.min(np.linalg.norm(p[:3, None, :] - p[None, 3:, :], axis=-1)))
        rows.append(
            {
                "dOO": d,
                "minGap": round(gap, 6),
                "eSr": res["eSr"],
                "eLr": res["eLr"],
                "q": _r(res["q"]),
                "fLrO": _r(res["fLr"][3]),
                "fSrO": _r(res["fSr"][3]),
            }
        )
    return {
        "symbols": syms,
        "positions": _r(pos0, 7),
        "axis": _r(axis, 9),
        "d0": d0,
        "monomers": [
            {"eSr": m["eSr"], "eLr": m["eLr"], "q": _r(m["q"]), "P": _r(m["P"])} for m in mono
        ],
        "rows": rows,
    }


def validation(model, readouts, hyper, test_dir: Path, keep_points: int = 2) -> dict:
    """Energies, forces, dipoles and BECs on the les_fit test configurations vs their DFT labels."""
    from ase.io import read

    subsets = []
    bec_points: list[list] = []
    dip_points: list[list] = []
    for name in TEST_SUBSETS:
        frames = read(test_dir / f"{name}.xyz", index=":")
        e_err, f_err, f_err_nolr, f_n = [], 0.0, 0.0, 0
        zz, zr, pp, pr = [], [], [], []
        for k, at in enumerate(frames):
            res = run(model, readouts, hyper, _atoms(at.get_chemical_symbols(), at.positions))
            n = len(at)
            f_dft = at.get_forces()
            e_err.append((res["eSr"] + res["eLr"] - at.get_potential_energy()) / n)
            f_err += float(((res["fSr"] + res["fLr"] - f_dft) ** 2).sum())
            f_err_nolr += float(((res["fSr"] - f_dft) ** 2).sum())
            f_n += 3 * n
            Zr = at.arrays["bec"].reshape(n, 3, 3)
            mu = np.asarray(at.calc.results["dipole"], dtype=float)
            zz.append(res["Z"].reshape(-1))
            zr.append(Zr.reshape(-1))
            pp.append(res["P"])
            pr.append(mu)
            if k < keep_points:
                for i in range(n):
                    for a in range(3):
                        bec_points.append([name, at.numbers[i].item(), round(float(Zr[i, a, a]), 4), round(float(res["Z"][i, a, a]), 4)])
                dip_points.append([name, _r(mu, 4), _r(res["P"], 4)])
        zz_, zr_ = np.concatenate(zz), np.concatenate(zr)
        pp_, pr_ = np.concatenate(pp), np.concatenate(pr)
        subsets.append(
            {
                "name": name,
                "configs": len(frames),
                "atoms": f_n // 3,
                "energyRmsePerAtom": float(np.sqrt(np.mean(np.square(e_err)))),
                "energyMeanPerAtom": float(np.mean(e_err)),
                "forceRmse": math.sqrt(f_err / f_n),
                "forceRmseWithoutLr": math.sqrt(f_err_nolr / f_n),
                "becSlope": float(zz_ @ zr_ / (zr_ @ zr_)),
                "becCorr": float(np.corrcoef(zz_, zr_)[0, 1]),
                "dipoleSlope": float(pp_ @ pr_ / (pr_ @ pr_)),
                "dipoleCorr": float(np.corrcoef(pp_, pr_)[0, 1]),
            }
        )
    return {"subsets": subsets, "becPoints": bec_points, "dipolePoints": dip_points}


ROTATION = {"axis": [1.0, 2.0, 3.0], "angleDeg": 50.0, "system": "ethanol"}
SCAN = [2.6, 2.75, 2.91, 3.1, 3.3, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 13.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0]


def main(argv: list[str]) -> int:
    ckpt, test_dir, structures_dir, out_path = (Path(a) for a in argv[:4])
    raw = ckpt.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != CHECKPOINT_MD5:
        raise SystemExit(f"checkpoint md5 {md5} != {CHECKPOINT_MD5} (les_fit README)")
    model, readouts, hyper = build_model(ckpt)
    import ase
    import e3nn
    import mace

    systems = []
    eth = json.loads((structures_dir / "ethanol.json").read_text())
    atoms = _atoms(eth["symbols"], eth["positions"])
    entry = {"id": "ethanol", "label": "Ethanol", "symbols": eth["symbols"], "positions": eth["positions"], "cell": None}
    entry.update(summarise(run(model, readouts, hyper, atoms), atoms))
    R = rotation_matrix(ROTATION["axis"], ROTATION["angleDeg"])
    rot = _atoms(eth["symbols"], atoms.positions @ R.T)
    rr = run(model, readouts, hyper, rot)
    entry["rotated"] = {
        "R": _r(R, 9),
        "positions": _r(rot.positions, 7),
        "eSr": rr["eSr"],
        "eLr": rr["eLr"],
        "q": _r(rr["q"]),
        "fSr": _r(rr["fSr"]),
        "fLr": _r(rr["fLr"]),
        "P": _r(rr["P"]),
    }
    systems.append(entry)

    from ase.io import read

    dip = read(test_dir / "dipeptides.xyz", index=0)
    datoms = _atoms(dip.get_chemical_symbols(), dip.positions)
    entry = {"id": "dipeptide", "label": "Dipeptide (SPICE test)", "symbols": dip.get_chemical_symbols(),
             "positions": _r(dip.positions, 7), "cell": None, "smiles": dip.info.get("smiles")}
    entry.update(summarise(run(model, readouts, hyper, datoms), datoms))
    entry["dft"] = {"dipole": _r(np.asarray(dip.calc.results["dipole"], dtype=float)), "Z": _r(dip.arrays["bec"].reshape(-1, 3, 3), 4),
                    "forces": _r(dip.get_forces()), "energy": float(dip.get_potential_energy())}
    systems.append(entry)

    doc = {
        "format": "atlas-xray-les",
        "version": 1,
        "status": "PRECOMPUTED",
        "model": {
            "family": "les",
            "checkpoint": "maceles-off",
            "file": CHECKPOINT_NAME,
            "url": CHECKPOINT_URL,
            "md5": md5,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "licence": "CC BY-NC 4.0",
            "hyperparameters": hyper,
            "halfInvEps0": HALF_INV_EPS0,
            "parameters": {
                "mace": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "chargeReadouts": sum(p.numel() for p in readouts.parameters()),
            },
        },
        "provenance": {
            "mace-torch": mace.__version__,
            "e3nn": e3nn.__version__,
            "torch": torch.__version__,
            "ase": ase.__version__,
            "les": "re-implemented in pipeline/atlas_pipeline/xray_les.py (finite systems only)",
            "lesFitCommit": LES_FIT_COMMIT,
            "dtype": "float64 (the checkpoint was trained in float32)",
            "device": "cpu",
            "rounding": "computed in float64; features to 4 significant figures, charges, energies and forces to 5; |x| < 1e-12 written as 0",
            "script": "pipeline/atlas_pipeline/xray_les.py",
            "rotation": ROTATION,
        },
        "systems": systems,
        "dimer": water_dimer_scan(model, readouts, hyper, SCAN),
        "validation": validation(model, readouts, hyper, test_dir),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
