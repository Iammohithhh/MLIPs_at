"""Architecture X-ray, NequIP: real intermediate tensors of a released NequIP checkpoint (CPU).

Gallery structures are passed through NequIP-OAM-S 0.1 (mir-group, Zenodo 10.5281/zenodo.18775904,
CC-BY-4.0) and every intermediate tensor of the forward pass is captured with forward hooks:

    type embedding → Bessel × cutoff → spherical harmonics → per layer: linear → edge MLP
    weights → tensor product + scatter (the convolution) → linear (+ self-connection)
    → gate → readout → per-type scale and shift → + ZBL repulsion → site energies

Safety: a ``.nequip.zip`` is a ``torch.package`` archive. Loading it the normal way imports the
Python source stored inside the archive and executes it. We never do that. The pickled module
tree is read with a restricted unpickler that resolves every global to an inert stub except the
tensor rebuilders and plain containers, and tensor storages are read as raw bytes from the
archive. A fresh model is then built by the installed ``nequip`` (pinned to the version that
wrote the package) from hyperparameters read off that inert tree, and the tensors are loaded
into it with ``strict=True``. The only pickled tensors the fresh model does not hold as state
are the non-persistent weight scalings ``alpha`` of the scalar MLPs, which it regenerates; they
are checked against the stored values.

Usage:
  python -m atlas_pipeline.xray_nequip <NequIP-OAM-S-0.1.nequip.zip> ../atlas/src/data/structures ../atlas/src/data/xray/nequip-oam-s.json
"""

from __future__ import annotations

import collections
import hashlib
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch._utils

from .xray import _r, rotation_matrix

CHECKPOINT_NAME = "NequIP-OAM-S-0.1.nequip.zip"
CHECKPOINT_URL = "https://zenodo.org/records/18775904/files/" + CHECKPOINT_NAME
CHECKPOINT_DOI = "10.5281/zenodo.18775904"
CHECKPOINT_SHA256 = "63d4bafd872850a014fd21dedeea416b61173a17750dee0b2b8dd2b126f407aa"
CHECKPOINT_MD5 = "399a98bf36fc4550bc85d48f35451c6c"  # as listed by Zenodo

# ---------------------------------------------------------------------------
# Restricted reading of the torch.package archive

_SAFE = {
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("builtins", "set"): set,
    ("torch", "float32"): torch.float32,
    ("torch", "float64"): torch.float64,
}
for _st in ("DoubleStorage", "FloatStorage", "LongStorage", "IntStorage", "BoolStorage"):
    _SAFE[("torch", _st)] = getattr(torch, _st)


class _Stub:
    """Inert stand-in for any class or function named in the pickle."""

    _qual = "?"

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.args = args
        return obj

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __setstate__(self, state):
        self.state = state

    def __repr__(self) -> str:
        return f"<stub {self._qual}>"


def _archive_root(zf: zipfile.ZipFile) -> str:
    roots = {n.split("/")[0] for n in zf.namelist()}
    if len(roots) != 1:
        raise RuntimeError(f"expected one top-level folder in the package, found {sorted(roots)}")
    return roots.pop()


def _unpickler(zf: zipfile.ZipFile, root: str):
    storages: dict[str, torch.Tensor] = {}

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in _SAFE:
                return _SAFE[(module, name)]
            return type(name, (_Stub,), {"_qual": f"{module}.{name}"})

        def persistent_load(self, pid):
            kind = pid[0].decode() if isinstance(pid[0], bytes) else pid[0]
            if kind != "storage":
                return _Stub(pid)  # 'reduce_package' objects (stored source) stay inert
            storage_type, key, _location, _size = pid[1:]
            dtype = getattr(storage_type, "dtype", torch.uint8)
            if key not in storages:
                raw = zf.read(f"{root}/.data/{key}.storage")
                storages[key] = torch.frombuffer(bytearray(raw), dtype=dtype).clone() if raw else torch.empty(0, dtype=dtype)
            return torch.storage.TypedStorage(wrap_storage=storages[key].untyped_storage(), dtype=dtype, _internal=True)

    return U


def _state(obj) -> dict:
    s = getattr(obj, "state", None)
    if isinstance(s, tuple) and len(s) == 2 and isinstance(s[1], dict):
        s = {**(s[0] or {}), **s[1]}
    return s if isinstance(s, dict) else {}


def _children(obj) -> dict:
    return {k: v for k, v in (_state(obj).get("_modules") or {}).items() if v is not None}


def _flat_tensors(obj, prefix: str = "", out: dict | None = None) -> dict:
    out = {} if out is None else out
    st = _state(obj)
    for key in ("_parameters", "_buffers"):
        for k, v in (st.get(key) or {}).items():
            if isinstance(v, torch.Tensor):
                out[prefix + k] = v.detach()
    for k, v in _children(obj).items():
        _flat_tensors(v, prefix + k + ".", out)
    return out


def _irreps(stub) -> str:
    """e3nn Irreps stub → '128x0e+64x1e'. Irreps is a tuple of _MulIr(mul, Irrep(l, p))."""
    parts = []
    for mulir in stub.args[0]:
        mul, ir = mulir.args[0]
        l, p = ir.args[0]
        parts.append(f"{mul}x{l}{'e' if p == 1 else 'o'}")
    return "+".join(parts)


def read_package(path: Path) -> tuple[dict, dict, dict]:
    """(tensors, hyperparameters, package metadata) without executing anything in the archive."""
    import yaml

    with zipfile.ZipFile(path) as zf:
        root = _archive_root(zf)
        meta = yaml.safe_load(zf.read(f"{root}/model/package_metadata.txt"))
        tree = _unpickler(zf, root)(io.BytesIO(zf.read(f"{root}/model/eager_model.pkl"))).load()
    sole = _children(tree)["sole_model"]
    tensors = _flat_tensors(sole)
    func = _children(_children(sole)["model"])["func"]
    mods = _children(func)
    layers = sorted(k for k in mods if k.startswith("layer") and k.endswith("_convnet"))
    edge_norm = _state(mods["edge_norm"])
    bessel = mods["bessel_encode"]
    conv0 = _children(mods[layers[0]])["conv"]
    edge_mlp0 = _state(_children(conv0)["edge_mlp"])
    hyper = {
        "r_max": float(edge_norm["r_max"]),
        "num_bessels": int(_state(bessel)["num_bessels"]),
        "bessel_trainable": bool(_state(bessel)["trainable"]),
        "polynomial_cutoff_p": float(_state(_children(bessel)["cutoff"])["p"]),
        "edge_factor": float(_state(mods["factor"])["factor"]),
        "type_embed_num_features": int(_state(mods["type_embed"])["num_features"]),
        "irreps_edge_sh": _irreps(_state(_children(mods["spharm"])["sh"])["irreps_out"]),
        "feature_irreps_hidden": [_irreps(_state(mods[k])["feature_irreps_hidden"]) for k in layers],
        "radial_mlp_dims": [list(_state(_children(_children(mods[k])["conv"])["edge_mlp"])["dims"]) for k in layers],
        "scatter_norm_factor": float(_state(conv0)["scatter_norm_factor"]),
        "pair_potential": "ZBL" if "pair_potential" in mods else None,
        "num_layers": len(layers),
        "type_names": list(_state(sole)["type_names"]),
        "model_dtype": str(_state(sole)["model_dtype"]).replace("torch.", ""),
    }
    hyper["avg_num_neighbors"] = 1.0 / hyper["scatter_norm_factor"] ** 2
    hyper["bessel_weights"] = [float(x) for x in tensors["model.func.bessel_encode.bessel_weights"].reshape(-1)]
    if edge_mlp0["bias"]:
        raise RuntimeError("radial MLP with bias: not the architecture this script rebuilds")
    return tensors, hyper, meta


def build_model(path: Path):
    """Rebuild the packaged architecture with the installed nequip and load the stored tensors."""
    from nequip.model import FullNequIPGNNModel
    from nequip.utils.global_state import set_global_state

    set_global_state()
    sd, hp, meta = read_package(path)
    tn = hp["type_names"]
    widths = [dims[1:-1] for dims in hp["radial_mlp_dims"]]
    if any(len(w) != 1 for w in widths):
        raise RuntimeError(f"unexpected radial MLP depths {hp['radial_mlp_dims']}")
    model = FullNequIPGNNModel(
        seed=0,
        model_dtype="float64",  # the package computes in float32; we run the same weights in float64
        r_max=hp["r_max"],
        type_names=tn,
        feature_irreps_hidden=hp["feature_irreps_hidden"],
        irreps_edge_sh=hp["irreps_edge_sh"],
        type_embed_num_features=hp["type_embed_num_features"],
        radial_mlp_depth=[1] * hp["num_layers"],
        radial_mlp_width=[w[0] for w in widths],
        num_bessels=hp["num_bessels"],
        bessel_trainable=hp["bessel_trainable"],
        polynomial_cutoff_p=hp["polynomial_cutoff_p"],
        avg_num_neighbors=hp["avg_num_neighbors"],
        per_type_energy_scales={t: 1.0 for t in tn},  # placeholders: overwritten by the stored buffers
        per_type_energy_shifts={t: 0.0 for t in tn},
        pair_potential={"_target_": "nequip.nn.pair_potential.ZBL", "units": "metal", "chemical_species": tn} if hp["pair_potential"] else None,
    )
    own = model.state_dict()
    alphas = {k: v for k, v in sd.items() if k.endswith(".alpha")}
    rest = {k: v for k, v in sd.items() if k not in alphas}
    if set(rest) != set(own):
        raise RuntimeError(f"state mismatch: extra {sorted(set(rest) - set(own))[:5]}, missing {sorted(set(own) - set(rest))[:5]}")
    for k, v in rest.items():
        if own[k].shape != v.shape:
            raise RuntimeError(f"shape mismatch at {k}: {tuple(own[k].shape)} vs {tuple(v.shape)}")
    model.load_state_dict({k: v.to(own[k].dtype) for k, v in rest.items()}, strict=True)
    mods = dict(model.named_modules())
    for k, v in alphas.items():
        mine = float(mods[k[: -len(".alpha")]].alpha)
        if abs(mine - float(v)) > 1e-7 * abs(mine):
            raise RuntimeError(f"alpha mismatch at {k}: {mine} vs {float(v)}")
    model.eval()
    return model, hp, meta


# ---------------------------------------------------------------------------
# Forward pass with hooks


def _atoms(structure: dict):
    from ase import Atoms

    return Atoms(symbols=structure["symbols"], positions=structure["positions"], cell=structure.get("cell") or None, pbc=structure.get("pbc", False))


def _data(atoms, type_names, r_max):
    from nequip.data.ase import from_ase
    from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper, NeighborListTransform

    d = from_ase(atoms)
    d = ChemicalSpeciesToAtomTypeMapper(type_names)(d)
    return NeighborListTransform(r_max=r_max)(d)


def forward_with_hooks(model, atoms, hp) -> dict:
    func = model.model.func
    seen: dict[str, object] = {}
    handles = []

    def grab(name, what="out"):
        def hook(_m, inp, out):
            seen[name] = inp if what == "in" else out

        return hook

    fields = {}

    def snap(name, key):
        def hook(_m, _inp, out):
            fields[name] = out[key].detach().clone()

        return hook

    handles.append(func.type_embed.register_forward_hook(snap("embed", "node_features")))
    handles.append(func.factor.register_forward_hook(snap("radial", "edge_embedding")))
    handles.append(func.bessel_encode.register_forward_hook(snap("cutoff", "edge_cutoff")))
    handles.append(func.spharm.register_forward_hook(snap("sh", "edge_attrs")))
    for t in range(hp["num_layers"]):
        L = getattr(func, f"layer{t}_convnet")
        c = L.conv
        handles.append(c.edge_mlp.register_forward_hook(grab(f"w{t}")))
        handles.append(c.tp_scatter.register_forward_hook(grab(f"m{t}")))
        handles.append(L.conv.register_forward_hook(snap(f"pre{t}", "node_features")))
        handles.append(L.register_forward_hook(snap(f"h{t}", "node_features")))
        handles.append(L.equivariant_nonlin.act_gates.register_forward_hook(grab(f"g{t}")))
    handles.append(func.per_atom_energy_readout.register_forward_hook(snap("readout", "atomic_energy")))
    handles.append(func.per_type_energy_scale_shift.register_forward_hook(snap("scaled", "atomic_energy")))
    if hp["pair_potential"]:
        handles.append(func.pair_potential.register_forward_hook(snap("withpair", "atomic_energy")))
    data = _data(atoms, hp["type_names"], hp["r_max"])
    try:
        out = model(data)
    finally:
        for h in handles:
            h.remove()
    return {"data": data, "seen": seen, "fields": fields, "out": out}


def _split(x: np.ndarray, irreps: str) -> list[np.ndarray]:
    """[n, sum_l C(2l+1)] in e3nn mul-major layout → per-irrep arrays [n, C, 2l+1]."""
    out, k = [], 0
    for part in irreps.split("+"):
        mul, ir = part.split("x")
        mul, l = int(mul), int(ir[:-1])
        d = 2 * l + 1
        out.append(x[:, k : k + mul * d].reshape(x.shape[0], mul, d))
        k += mul * d
    if k != x.shape[1]:
        raise RuntimeError(f"irreps {irreps} do not cover width {x.shape[1]}")
    return out


def summarise(model, atoms, run: dict, hp: dict) -> dict:
    seen, f, out, data = run["seen"], run["fields"], run["out"], run["data"]
    func = model.model.func
    n = len(atoms)
    ei = data["edge_index"].numpy()
    # NequIP: edge_index[0] is the atom that receives the message (edge_dst in the code),
    # edge_index[1] the neighbour that sends it; edge_vectors = r[ei[1]] − r[ei[0]] + shift.
    evec = out["edge_vectors"].detach().numpy()
    layers = []
    for t in range(hp["num_layers"]):
        L = getattr(func, f"layer{t}_convnet")
        tp = L.conv.tp_scatter.tp
        mid = str(tp.irreps_out).replace(" ", "")
        hid = hp["feature_irreps_hidden"][t]
        m = _split(seen[f"m{t}"].detach().numpy(), mid)
        h = _split(f[f"h{t}"].numpy(), hid)
        pre = _split(f[f"pre{t}"].numpy(), str(L.equivariant_nonlin.irreps_in).replace(" ", ""))
        w = seen[f"w{t}"].detach().numpy()  # [E, weight_numel]: one weight per path and channel
        gates = seen.get(f"g{t}")
        ls_m = [int(p.split("x")[1][:-1]) for p in mid.split("+")]
        layers.append(
            {
                "tp": {
                    "irreps_in1": str(tp.irreps_in1).replace(" ", ""),
                    "irreps_in2": str(tp.irreps_in2).replace(" ", ""),
                    "irreps_out": mid,
                    "paths": [[ins.i_in1, ins.i_in2, ins.i_out] for ins in tp.instructions],
                    "weightNumel": int(tp.weight_numel),
                },
                "message": {  # aggregated convolution output, before linear_2 (one block per path)
                    "ls": ls_m,
                    "norms": [_r(np.linalg.norm(b, axis=-1), 4) for b in m],  # [block][atom][channel]
                    "l1": [_r(b) if b.shape[-1] == 3 else None for b in m],
                },
                "edgeWeightNorm": _r(np.linalg.norm(w, axis=1), 4),  # [edge]
                "h": {
                    "irreps": hid,
                    "l0": _r(h[0][:, :, 0], 4),
                    "l1": _r(h[1], 4) if len(h) > 1 else None,
                },
                "preGate": {"l0": _r(pre[0][:, :, 0], 4)},
                "gates": _r(gates.detach().numpy(), 4) if gates is not None and gates.numel() else None,
            }
        )
    readout = f["readout"].numpy().reshape(n)
    scaled = f["scaled"].numpy().reshape(n)
    final = out["atomic_energy"].detach().numpy().reshape(n)
    zbl = final - scaled if hp["pair_potential"] else np.zeros(n)
    types = data["atom_types"].numpy().reshape(n)
    scales = func.per_type_energy_scale_shift.scales.detach().numpy().reshape(-1)
    shifts = func.per_type_energy_scale_shift.shifts.detach().numpy().reshape(-1)
    return {
        "n": n,
        "edges": {
            "count": int(ei.shape[1]),
            "src": ei[1].tolist(),
            "dst": ei[0].tolist(),
            "vec": _r(-evec),  # r_dst − r_src (+ shift), the convention of the MACE file
            "radial": _r(f["radial"].numpy()),
            "cutoff": _r(f["cutoff"].numpy().reshape(-1)),
            "sh": _r(f["sh"].numpy()),  # Y(r̂ from receiving atom to neighbour), e3nn order, component norm
        },
        "embed": {"shape": list(f["embed"].shape), "h0": _r(f["embed"].numpy(), 4)},
        "layers": layers,
        "readout": _r(readout),
        "scale": _r(scales[types]),
        "shift": _r(shifts[types]),
        "zbl": _r(zbl),
        "nodeEnergy": _r(final),
        "energy": float(out["total_energy"].detach().numpy().reshape(())),
        "forces": _r(out["forces"].detach().numpy()),
    }


# ---------------------------------------------------------------------------
# Dimer scans: the learned part and the ZBL prior separately


def dimer_scan(model, hp, a: str, b: str, rs: list[float]) -> dict:
    from ase import Atoms

    func = model.model.func
    rows = {"r": rs, "total": [], "learned": [], "zbl": [], "e0": []}
    for r in rs:
        atoms = Atoms(f"{a}{b}", positions=[[0, 0, 0], [0, 0, r]], cell=[40, 40, 40], pbc=False)
        run = forward_with_hooks(model, atoms, hp)
        scaled = run["fields"]["scaled"].numpy().reshape(-1)
        final = run["out"]["atomic_energy"].detach().numpy().reshape(-1)
        rows["total"].append(float(final.sum()))
        rows["zbl"].append(float((final - scaled).sum()))
        rows["learned"].append(float(scaled.sum()))
    # isolated atoms: no edges, so every site energy is the per-type shift (plus scale × readout of the bare embedding)
    for sym in (a, b):
        atoms = Atoms(sym, positions=[[0, 0, 0]], cell=[40, 40, 40], pbc=False)
        rows["e0"].append(float(forward_with_hooks(model, atoms, hp)["out"]["total_energy"].detach().numpy().reshape(())))
    return {"pair": f"{a}–{b}", **{k: (_r(v, 6) if k != "r" else v) for k, v in rows.items()}}


SYSTEMS = [
    ("ethanol", "Ethanol"),
    ("zno-wurtzite", "ZnO (wurtzite)"),
    ("si-diamond", "Si (diamond)"),
]
ROTATION = {"axis": [1.0, 2.0, 3.0], "angleDeg": 50.0, "system": "ethanol"}
MIRROR = {"normal": [0.0, 0.0, 1.0], "system": "ethanol"}  # reflection z → −z
DIMERS = [("Si", "Si"), ("Zn", "O"), ("C", "H")]
DIMER_R = [round(0.3 + 0.05 * k, 2) for k in range(int((4.6 - 0.3) / 0.05) + 1)]


def parameter_counts(model) -> dict:
    """Trainable parameters per module of the rebuilt model (fixed buffers are not counted)."""
    out: dict[str, int] = {}
    for name, p in model.named_parameters():
        key = name.removeprefix("model.func.").rsplit(".", 1)[0]
        for prefix in ("type_embed", "bessel_encode", "per_atom_energy_readout"):
            if key.startswith(prefix):
                key = prefix
        if ".edge_mlp." in key:
            key = key.split(".edge_mlp.")[0] + ".edge_mlp"
        out[key] = out.get(key, 0) + p.numel()
    return out


def _light(r: dict) -> dict:
    return {
        "positions": None,
        "energy": r["energy"],
        "forces": r["forces"],
        "nodeEnergy": r["nodeEnergy"],
        "layers": [{"h": {"l0": L["h"]["l0"], "l1": L["h"]["l1"]}, "message": {"norms": L["message"]["norms"], "l1": L["message"]["l1"]}} for L in r["layers"]],
    }


def main(argv: list[str]) -> int:
    ckpt, structures_dir, out_path = Path(argv[0]), Path(argv[1]), Path(argv[2])
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise SystemExit(f"package sha256 {digest} != expected {CHECKPOINT_SHA256}")
    torch.set_default_dtype(torch.float64)
    model, hp, meta = build_model(ckpt)
    import ase
    import e3nn
    import nequip

    systems = []
    for sid, label in SYSTEMS:
        structure = json.loads((structures_dir / f"{sid}.json").read_text())
        atoms = _atoms(structure)
        entry = {"id": sid, "label": label, "symbols": structure["symbols"], "positions": structure["positions"], "cell": structure.get("cell"), "pbc": structure.get("pbc", False)}
        entry.update(summarise(model, atoms, forward_with_hooks(model, atoms, hp), hp))
        if sid == ROTATION["system"]:
            R = rotation_matrix(ROTATION["axis"], ROTATION["angleDeg"])
            rot = atoms.copy()
            rot.positions = atoms.positions @ R.T
            entry["rotated"] = {**_light(summarise(model, rot, forward_with_hooks(model, rot, hp), hp)), "R": _r(R, 9), "positions": _r(rot.positions, 7)}
        if sid == MIRROR["system"]:
            nrm = np.asarray(MIRROR["normal"], dtype=float)
            M = np.eye(3) - 2 * np.outer(nrm, nrm)
            mir = atoms.copy()
            mir.positions = atoms.positions @ M.T
            entry["mirrored"] = {**_light(summarise(model, mir, forward_with_hooks(model, mir, hp), hp)), "M": _r(M, 9), "positions": _r(mir.positions, 7)}
        systems.append(entry)

    doc = {
        "format": "atlas-xray-nequip",
        "version": 1,
        "status": "PRECOMPUTED",
        "model": {
            "family": "nequip",
            "checkpoint": "nequip-oam-s-0-1",
            "file": CHECKPOINT_NAME,
            "url": CHECKPOINT_URL,
            "doi": CHECKPOINT_DOI,
            "sha256": digest,
            "licence": "CC-BY-4.0",
            "packagedWith": meta.get("versions", {}),
            "hyperparameters": {k: v for k, v in hp.items() if k != "type_names"},
            "numTypes": len(hp["type_names"]),
            "parameters": parameter_counts(model),
        },
        "provenance": {
            "nequip": nequip.__version__,
            "e3nn": e3nn.__version__,
            "torch": torch.__version__,
            "ase": ase.__version__,
            "dtype": "float64 (the package itself runs in float32)",
            "device": "cpu",
            "rounding": "computed in float64; features written to 4 significant figures, energies, forces and edges to 5, dimer energies to 6; |x| < 1e-12 written as 0",
            "script": "pipeline/atlas_pipeline/xray_nequip.py",
            "rotation": ROTATION,
            "mirror": MIRROR,
        },
        "systems": systems,
        "dimers": [dimer_scan(model, hp, a, b, DIMER_R) for a, b in DIMERS],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
