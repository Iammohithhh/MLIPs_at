# Copied from pipeline/atlas_pipeline/xray.py (MLIP Atlas): restricted checkpoint reader, model rebuild and
# forward-pass hooks. inside_mace/intervention.py builds the Inside-MACE experiments on top of it.
"""Architecture X-ray: real intermediate tensors of a real MACE forward pass (CPU).

A small structure from the Atlas gallery is passed through the MACE-MP-0a medium
checkpoint (2023-12-03-mace-128-L1_epoch-199.model, MIT) and every intermediate
tensor of the forward pass is captured with forward hooks:

    node embedding → radial basis → spherical harmonics → interaction (A basis)
    → symmetric contraction (product basis, body order) → readout → site energies

Safety: the checkpoint is a pickled ``torch.nn.Module`` whose ``torch.fx`` parts
store Python source that ``torch.load`` would ``exec``. We never unpickle it as code.
A restricted unpickler resolves every global to an inert stub except the tensor
rebuilders and plain containers, so only tensors (and two plain numbers) are read.
A fresh ``ScaleShiftMACE`` is then built from the installed ``mace-torch`` and the
tensors are loaded into it; the only keys left to the fresh model are constants it
regenerates itself (Wigner-3j blocks, zero-weight flags), checked below.

Usage:
  python -m atlas_pipeline.xray <checkpoint.model> ../atlas/src/data/structures ../atlas/src/data/xray/mace-mp-0a-medium.json
"""

from __future__ import annotations

import collections
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch._utils

torch.serialization.add_safe_globals([slice])  # e3nn 0.4.4 stores slices in its own constants.pt

CHECKPOINT_NAME = "2023-12-03-mace-128-L1_epoch-199.model"
CHECKPOINT_URL = "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/" + CHECKPOINT_NAME
CHECKPOINT_SHA256 = "01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845"

# ---------------------------------------------------------------------------
# Restricted reading of the checkpoint


_SAFE = {
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("__builtin__", "set"): set,
    ("builtins", "set"): set,
}
for _st in ("DoubleStorage", "FloatStorage", "LongStorage", "IntStorage", "BoolStorage"):
    _SAFE[("torch", _st)] = getattr(torch, _st)


class _Stub:
    """Inert stand-in for any class or function named in the pickle."""

    _qual = "?"

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __setstate__(self, state):
        self.state = state

    def __repr__(self) -> str:
        return f"<stub {self._qual}>"


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _SAFE:
            return _SAFE[(module, name)]
        if (module, name) == ("_codecs", "encode"):
            return lambda s, enc="latin1": s.encode(enc)
        return type(name, (_Stub,), {"_qual": f"{module}.{name}"})


_pickle_module = types.SimpleNamespace(
    Unpickler=_Unpickler, load=lambda f, **kw: _Unpickler(f, **kw).load(), __name__="atlas_restricted_pickle"
)


def _state(obj) -> dict:
    s = getattr(obj, "state", None)
    if isinstance(s, tuple) and len(s) == 2 and isinstance(s[1], dict):
        s = {**(s[0] or {}), **s[1]}
    if s is None and getattr(obj, "args", None) and isinstance(obj.args[0], dict):
        s = obj.args[0]  # torch.fx GraphModule body (never rebuilt)
    return s if isinstance(s, dict) else {}


def _flat_tensors(obj, prefix: str = "", out: dict | None = None) -> dict:
    out = {} if out is None else out
    st = _state(obj)
    for key in ("_parameters", "_buffers"):
        for k, v in (st.get(key) or {}).items():
            if isinstance(v, torch.Tensor):
                out[prefix + k] = v.detach()
    for k, v in (st.get("_modules") or {}).items():
        if v is not None:
            _flat_tensors(v, prefix + k + ".", out)
    return out


def read_checkpoint(path: Path) -> tuple[dict, dict]:
    """Tensors and the two non-tensor hyperparameters, without executing the pickle."""
    # weights_only=False only selects our pickle_module, whose find_class returns nothing callable.
    tree = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_pickle_module)
    tensors = _flat_tensors(tree)
    inter0 = _state(_state(_state(tree)["_modules"]["interactions"])["_modules"]["0"])
    return tensors, {"avg_num_neighbors": float(inter0["avg_num_neighbors"]), "radial_MLP": list(inter0["radial_MLP"])}


def build_model(path: Path, load_weights: bool = True):
    """Rebuild the checkpoint's architecture; with load_weights=False the weights stay at their
    random initialisation (seed them with torch.manual_seed first): an untrained control."""
    from e3nn import o3
    from mace import modules

    torch.set_default_dtype(torch.float64)
    sd, extra = read_checkpoint(path)
    block = modules.interaction_classes["RealAgnosticResidualInteractionBlock"]
    hyper = {
        "r_max": float(sd["r_max"]),
        "num_bessel": int(sd["radial_embedding.bessel_fn.bessel_weights"].numel()),
        "num_polynomial_cutoff": int(sd["radial_embedding.cutoff_fn.p"]),
        "max_ell": 3,
        "num_interactions": int(sd["num_interactions"]),
        # L1 checkpoints (medium) carry an l = 1 symmetric contraction; L0 ones (small) do not.
        "hidden_irreps": "128x0e+128x1o" if "products.0.symmetric_contractions.contractions.1.weights_max" in sd else "128x0e",
        "MLP_irreps": "16x0e",
        "correlation": 3,
        "avg_num_neighbors": extra["avg_num_neighbors"],
        "radial_MLP": extra["radial_MLP"],
        "atomic_inter_scale": float(sd["scale_shift.scale"]),
        "atomic_inter_shift": float(sd["scale_shift.shift"]),
    }
    model = modules.ScaleShiftMACE(
        r_max=hyper["r_max"],
        num_bessel=hyper["num_bessel"],
        num_polynomial_cutoff=hyper["num_polynomial_cutoff"],
        max_ell=hyper["max_ell"],
        interaction_cls=block,
        interaction_cls_first=block,
        num_interactions=hyper["num_interactions"],
        num_elements=int(sd["atomic_numbers"].numel()),
        hidden_irreps=o3.Irreps(hyper["hidden_irreps"]),
        MLP_irreps=o3.Irreps(hyper["MLP_irreps"]),
        atomic_energies=sd["atomic_energies_fn.atomic_energies"].numpy(),
        avg_num_neighbors=hyper["avg_num_neighbors"],
        atomic_numbers=sd["atomic_numbers"].tolist(),
        correlation=hyper["correlation"],
        gate=torch.nn.functional.silu,
        radial_MLP=hyper["radial_MLP"],
        use_reduced_cg=False,  # the 2023 checkpoints predate reduced Clebsch-Gordan coupling
        atomic_inter_scale=hyper["atomic_inter_scale"],
        atomic_inter_shift=hyper["atomic_inter_shift"],
    )
    if not load_weights:
        model.eval()
        return model, hyper
    own = model.state_dict()
    unexpected = sorted(set(sd) - set(own))
    missing = sorted(set(own) - set(sd))
    if unexpected:
        raise RuntimeError(f"checkpoint has tensors the rebuilt model lacks: {unexpected[:5]}")
    for k in missing:  # regenerated constants only
        if "._w3j_" not in k and not k.endswith("_zeroed"):
            raise RuntimeError(f"rebuilt model needs a tensor the checkpoint lacks: {k}")
        if k.endswith("_zeroed") and bool(own[k]):
            raise RuntimeError(f"unexpected zeroed weights: {k}")
    for k, v in sd.items():
        if own[k].shape != v.shape:
            raise RuntimeError(f"shape mismatch at {k}: {tuple(own[k].shape)} vs {tuple(v.shape)}")
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, hyper


# ---------------------------------------------------------------------------
# Forward pass with hooks


def _atoms(structure: dict):
    from ase import Atoms

    return Atoms(
        symbols=structure["symbols"],
        positions=structure["positions"],
        cell=structure.get("cell") or None,
        pbc=structure.get("pbc", False),
    )


def _batch(model, atoms):
    from mace import data as mace_data
    from mace.tools import torch_geometric, utils

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    config = mace_data.config_from_atoms(atoms)
    graph = mace_data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max))
    loader = torch_geometric.dataloader.DataLoader([graph], batch_size=1, shuffle=False, drop_last=False)
    return next(iter(loader)).to_dict()


# In e3nn 0.4 the l = 1 components (spherical harmonics and 1o irreps) are ordered (x, y, z),
# so D¹(R) = R. Checked by the rotation test in pipeline/tests/test_xray.py.
L1_TO_XYZ = [0, 1, 2]


def _blocks(x: torch.Tensor, lmax: int) -> list[np.ndarray]:
    """[n, C, (lmax+1)^2] → per-l arrays [n, C, 2l+1]."""
    return [x[:, :, l * l : (l + 1) * (l + 1)].detach().numpy() for l in range(lmax + 1)]


def _irreps_split(x: torch.Tensor, channels: int, ls: list[int]) -> list[np.ndarray]:
    """[n, sum_l C(2l+1)] in e3nn mul-major layout → per-l arrays [n, C, 2l+1]."""
    out, k = [], 0
    for l in ls:
        d = 2 * l + 1
        out.append(x[:, k : k + channels * d].reshape(-1, channels, d).detach().numpy())
        k += channels * d
    return out


def forward_with_hooks(model, atoms) -> dict:
    batch = _batch(model, atoms)
    batch["positions"].requires_grad_(True)
    seen: dict[str, object] = {}
    handles = []

    def grab(name):
        def hook(_m, _inp, out):
            seen[name] = out

        return hook

    handles.append(model.node_embedding.register_forward_hook(grab("embed")))
    handles.append(model.spherical_harmonics.register_forward_hook(grab("sh")))
    handles.append(model.radial_embedding.register_forward_hook(grab("radial")))
    for t, (inter, prod, read) in enumerate(zip(model.interactions, model.products, model.readouts)):
        handles.append(inter.register_forward_hook(grab(f"inter{t}")))
        handles.append(prod.register_forward_hook(grab(f"prod{t}")))
        handles.append(read.register_forward_hook(grab(f"read{t}")))
    try:
        out = model(batch, training=False, compute_force=True)
    finally:
        for h in handles:
            h.remove()
    return {"batch": batch, "seen": seen, "out": out}


ZERO = 1e-12  # |x| below this (float64 round-off of exact symmetry zeros) is written as 0


def _r(a, sig: int = 5):
    """Round to `sig` significant figures for the JSON (values are float64 internally)."""
    a = np.asarray(a, dtype=float)
    a = np.where(np.abs(a) < ZERO, 0.0, a)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(a == 0, 0, np.floor(np.log10(np.abs(a))))
    scale = 10.0 ** (sig - 1 - mag)
    return (np.round(a * scale) / scale).tolist()


def _cart(v: np.ndarray) -> np.ndarray:
    """l = 1 block [..., 3] in e3nn order → Cartesian (x, y, z)."""
    return v[..., L1_TO_XYZ]


def summarise(model, atoms, run: dict) -> dict:
    seen, out, batch = run["seen"], run["out"], run["batch"]
    C = 128
    n = len(atoms)
    edge_index = batch["edge_index"].numpy()
    shifts = batch["shifts"].detach().numpy()
    pos = batch["positions"].detach().numpy()
    vec = pos[edge_index[1]] - pos[edge_index[0]] + shifts
    radial, _cutoff = seen["radial"] if isinstance(seen["radial"], tuple) else (seen["radial"], None)
    sh = seen["sh"].detach().numpy()

    layers = []
    for t in range(len(model.interactions)):
        msg, _sc = seen[f"inter{t}"]
        A = _blocks(msg, 3)  # message: [n, C, 16] → l = 0..3
        prod = seen[f"prod{t}"]
        ls = [0, 1] if prod.shape[1] == 4 * C else [0]
        H = _irreps_split(prod, C, ls)
        site = seen[f"read{t}"].detach().numpy().reshape(n, -1)[:, 0]
        # A block whose largest entry is below 1e-30 is reported as vanishing (null) with
        # that maximum, rather than as a field of round-off.
        peak = [float(np.abs(b).max()) for b in A]
        live = [p > 1e-30 for p in peak]
        layers.append(
            {
                "A": {
                    "shape": list(msg.shape),
                    "peak": peak,
                    "norms": [_r(np.linalg.norm(b, axis=-1), 4) if ok else None for b, ok in zip(A, live)],  # [l][atom][channel]
                    "l1": _r(_cart(A[1]), 4) if live[1] else None,  # [atom][channel][xyz]
                },
                "h": {
                    "shape": list(prod.shape),
                    "irreps": "+".join(f"{C}x{l}{'e' if l % 2 == 0 else 'o'}" for l in ls),
                    "l0": _r(H[0][:, :, 0], 4),
                    "l1": _r(_cart(H[1]), 4) if len(H) > 1 else None,
                },
                "siteEnergy": _r(site),
            }
        )
    forces = out["forces"].detach().numpy()
    return {
        "n": n,
        "edges": {
            "count": int(edge_index.shape[1]),
            "src": edge_index[0].tolist(),
            "dst": edge_index[1].tolist(),
            "vec": _r(vec),
            "radial": _r(radial.detach().numpy()),
            "sh": _r(sh),  # e3nn order, component normalisation
        },
        "embed": {"shape": list(seen["embed"].shape), "h0": _r(seen["embed"].detach().numpy())},
        "layers": layers,
        "energy": float(out["energy"].detach().numpy()[0]),
        "nodeEnergy": _r(out["node_energy"].detach().numpy()),
        "e0": _r(model.atomic_energies_fn(batch["node_attrs"]).detach().numpy().reshape(n, -1)[:, 0]),
        "forces": _r(forces),
    }


def rotation_matrix(axis, angle_deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    th = np.radians(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


SYSTEMS = [
    ("ethanol", "Ethanol"),
    ("zno-wurtzite", "ZnO (wurtzite)"),
    ("si-diamond", "Si (diamond)"),
]
ROTATION = {"axis": [1.0, 2.0, 3.0], "angleDeg": 50.0, "system": "ethanol"}


def main(argv: list[str]) -> int:
    ckpt, structures_dir, out_path = Path(argv[0]), Path(argv[1]), Path(argv[2])
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint sha256 {digest} != expected {CHECKPOINT_SHA256}")
    model, hyper = build_model(ckpt)
    import ase
    import e3nn
    import mace

    systems = []
    for sid, label in SYSTEMS:
        structure = json.loads((structures_dir / f"{sid}.json").read_text())
        atoms = _atoms(structure)
        entry = {"id": sid, "label": label, "symbols": structure["symbols"], "positions": structure["positions"],
                 "cell": structure.get("cell"), "pbc": structure.get("pbc", False)}
        entry.update(summarise(model, atoms, forward_with_hooks(model, atoms)))
        if sid == ROTATION["system"]:
            R = rotation_matrix(ROTATION["axis"], ROTATION["angleDeg"])
            rot = atoms.copy()
            rot.positions = atoms.positions @ R.T
            r = summarise(model, rot, forward_with_hooks(model, rot))
            entry["rotated"] = {
                "R": _r(R, 9),
                "positions": _r(rot.positions, 7),
                "energy": r["energy"],
                "forces": r["forces"],
                "nodeEnergy": r["nodeEnergy"],
                "layers": [
                    {"h": {"l0": L["h"]["l0"], "l1": L["h"]["l1"]}, "A": {"norms": L["A"]["norms"], "l1": L["A"]["l1"]}, "siteEnergy": L["siteEnergy"]}
                    for L in r["layers"]
                ],
            }
        systems.append(entry)

    doc = {
        "format": "atlas-xray",
        "version": 1,
        "status": "PRECOMPUTED",
        "model": {
            "family": "mace",
            "checkpoint": "mace-mp-0a-medium",
            "file": CHECKPOINT_NAME,
            "url": CHECKPOINT_URL,
            "sha256": digest,
            "licence": "MIT",
            "hyperparameters": hyper,
        },
        "provenance": {
            "mace-torch": mace.__version__,
            "e3nn": e3nn.__version__,
            "torch": torch.__version__,
            "ase": ase.__version__,
            "dtype": "float64",
            "device": "cpu",
            "rounding": "computed in float64; features written to 4 significant figures, energies, forces and edges to 5; |x| < 1e-12 written as 0",
            "script": "pipeline/atlas_pipeline/xray.py",
            "rotation": ROTATION,
        },
        "systems": systems,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
