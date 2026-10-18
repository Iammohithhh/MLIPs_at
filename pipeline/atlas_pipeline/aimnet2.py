"""AIMNet2 on CPU: charges through the passes, fractional total charge, charged-dimer dissociation and
a methane distortion that the first pass cannot see.

AIMNet2 (Anstine, Zubatyuk & Isayev, Chem. Sci. 2025) writes the energy as U_local + U_disp + U_Coul
(Eq. 1). Three message-passing MLPs act on per-atom features built from Gaussian radial shells
(16 shifts, cosine cutoff 5 Å) times l = 0 and l = 1 direction factors. The first two also emit a
charge q and a non-negative weight f per atom, and after each update Neural Charge Equilibration
(NQE) restores the total charge Q:

    q_i = q~_i + f_i / (sum_j f_j + eps) * (Q - sum_j q~_j)

The released file (``aimnet2_wb97m_d3_0.pt``, ensemble member 0, MIT) is a plain state dict plus
a YAML model definition, read with ``torch.load(weights_only=True)``; nothing in it is executed.
The model is built and run with the authors' ``aimnet`` package (calculator ``AIMNet2Calculator``,
which adds the full Coulomb sum and DFT-D3(BJ) to the model's local energy), with
``AIMNet2._update_q`` wrapped to record q~, f and q after every pass and ``ConvSV.forward`` wrapped
to record each pass's environment features.

Usage:
  python -m atlas_pipeline.aimnet2 <aimnet2_wb97m_d3_0.pt> ../atlas/src/data/models/aimnet2.json
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

CHECKPOINT_NAME = "aimnet2_wb97m_d3_0.pt"
CHECKPOINT_URL = "https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/aimnet2_wb97m_d3_0.pt"
# From aimnet/calculators/model_registry.yaml at the commit below.
CHECKPOINT_SHA256 = "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
AIMNET_COMMIT = "d14692839f4e5544f4eb38c15cf01298c27960f8"

# Coulomb constant of the aimnet package: (Hartree / 2) * Bohr per ordered pair, i.e. Hartree * Bohr
# = 14.39964... eV Å per unordered pair (aimnet/constants.py).
HARTREE = 27.211386024367243
BOHR = 0.5291772105638411
K_E = HARTREE * BOHR

# Scans
CHARGE_GRID = [round(-1.0 + 0.05 * k, 2) for k in range(61)]  # Q = -1 ... +2
DIMER_D = [round(3.4 + 0.2 * k, 1) for k in range(84)]  # 3.4 ... 20.0 Å
THETA_DEG = list(range(30, 91))  # D2d path of CH4; 54.7356° is Td, 90° is D4h
CH_BOND = 1.087
TD_THETA = math.degrees(math.acos(1 / math.sqrt(3)))


# ---------------------------------------------------------------------------
# Geometry (pure numpy, tested)


def benzene() -> tuple[list[int], np.ndarray]:
    """D6h benzene in the xy plane, centroid at the origin (C–C 1.397 Å, C–H 1.084 Å)."""
    cc, ch = 1.397, 1.084
    nums, pos = [], []
    for k in range(6):
        a = math.pi / 3 * k
        nums.append(6)
        pos.append([cc * math.cos(a), cc * math.sin(a), 0.0])
    for k in range(6):
        a = math.pi / 3 * k
        nums.append(1)
        pos.append([(cc + ch) * math.cos(a), (cc + ch) * math.sin(a), 0.0])
    return nums, np.asarray(pos)


def water_h_down() -> tuple[list[int], np.ndarray]:
    """Water on the z axis with one O–H pointing down (−z) at the ring; that H at the origin.

    O–H 0.9572 Å, H–O–H 104.52° (gas-phase experimental geometry)."""
    oh, ang = 0.9572, math.radians(104.52)
    o = np.array([0.0, 0.0, oh])
    h1 = np.array([0.0, 0.0, 0.0])
    # second H in the xz plane at angle ang from the O→H1 direction (−z)
    h2 = o + oh * np.array([math.sin(ang), 0.0, -math.cos(ang)])
    return [8, 1, 1], np.array([o, h1, h2])


def methane_d2d(theta_deg: float, r: float = CH_BOND) -> tuple[list[int], np.ndarray]:
    """CH4 on the D2d path: H1,2 = r(±sin θ, 0, cos θ), H3,4 = r(0, ±sin θ, −cos θ).

    θ = arccos(1/√3) is tetrahedral, θ = 90° square planar. The four unit vectors sum to zero for
    every θ, and every C–H distance is r."""
    t = math.radians(theta_deg)
    s, c = math.sin(t), math.cos(t)
    pos = [[0.0, 0.0, 0.0], [r * s, 0, r * c], [-r * s, 0, r * c], [0, r * s, -r * c], [0, -r * s, -r * c]]
    return [6, 1, 1, 1, 1], np.asarray(pos, dtype=float)


def stack(a: tuple[list[int], np.ndarray], b: tuple[list[int], np.ndarray], d: float):
    """Fragment b shifted by d along z; returns numbers, positions and fragment index per atom."""
    nums = list(a[0]) + list(b[0])
    pos = np.vstack([a[1], b[1] + np.array([0.0, 0.0, d])])
    frag = [0] * len(a[0]) + [1] * len(b[0])
    return nums, pos, frag


def nqe(q_tilde: np.ndarray, f: np.ndarray, Q: float, eps: float = 1e-6) -> np.ndarray:
    """Neural charge equilibration (Zubatyuk et al. 2021; aimnet ops.nse)."""
    return q_tilde + f / (f.sum() + eps) * (Q - q_tilde.sum())


def exp_cutoff(d: np.ndarray, rc: float) -> np.ndarray:
    """aimnet ops.exp_cutoff: exp(-1 / (1 - (d/rc)^2)) / e^-1, 1 at d = 0 and 0 at d >= rc."""
    x = np.clip(d / rc, 0.0, 1.0 - 1e-6)
    return np.exp(-1.0 / (1.0 - x**2)) / math.exp(-1.0)


def coulomb_lr(q: np.ndarray, pos: np.ndarray, rc: float) -> float:
    """The Coulomb energy AIMNet2 adds: k Σ_{i<j} q_i q_j (1 − f_c(r_ij)) / r_ij (full sum minus the
    enveloped short-range part the model subtracts internally)."""
    e = 0.0
    n = len(q)
    for i in range(n):
        for j in range(i + 1, n):
            r = float(np.linalg.norm(pos[i] - pos[j]))
            e += q[i] * q[j] * (1.0 - float(exp_cutoff(np.array(r), rc))) / r
    return K_E * e


# ---------------------------------------------------------------------------
# Model


class Recorder:
    """Wraps AIMNet2._update_q and ConvSV.forward of one model instance."""

    def __init__(self, model):
        self.model = model
        self.passes: list[dict] = []
        self.conv: list[np.ndarray] = []
        orig_update = model._update_q
        orig_conv = model.conv_a.forward

        def update_q(data, x, delta_q=True):
            nq = model.num_charge_channels
            _q, _f = x[..., :nq], x[..., nq : 2 * nq]
            prev = data["charges"].detach().clone() if delta_q else None
            out = orig_update(data, x, delta_q=delta_q)
            q_tilde = (prev + _q) if delta_q else _q
            self.passes.append(
                {
                    "q_tilde": q_tilde.detach().reshape(-1).double().numpy().copy(),
                    "f": _f.detach().pow(2).reshape(-1).double().numpy().copy(),
                    "q": out["charges"].detach().reshape(-1).double().numpy().copy(),
                }
            )
            return out

        def conv(data, a):
            y = orig_conv(data, a)
            self.conv.append(y.detach().reshape(-1, y.shape[-1]).double().numpy().copy())
            return y

        model._update_q = update_q
        model.conv_a.forward = conv

    def reset(self):
        self.passes, self.conv = [], []


def load(ckpt: Path):
    import torch

    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint sha256 {digest} != expected {CHECKPOINT_SHA256}")
    from aimnet.calculators import AIMNet2Calculator

    calc = AIMNet2Calculator(model=str(ckpt.resolve()), device="cpu")
    calc_nod3 = AIMNet2Calculator(model=str(ckpt.resolve()), device="cpu", needs_dispersion=False)
    art = torch.load(ckpt, map_location="cpu", weights_only=True)
    meta = {k: v for k, v in art.items() if k not in ("state_dict", "model_yaml")}
    n_params = int(sum(v.numel() for k, v in art["state_dict"].items() if not k.startswith("aev.") and k != "outputs.srcoulomb.rc"))
    return calc, calc_nod3, meta, art["state_dict"], digest, n_params


def evaluate(calc, calc_nod3, rec: Recorder, nums, pos, Q: float, rc_coul: float, frag=None) -> dict:
    import torch

    data = {
        "coord": torch.tensor(pos, dtype=torch.float32),
        "numbers": torch.tensor(nums),
        "charge": torch.tensor([float(Q)]),
    }
    rec.reset()
    out = calc(dict(data))
    n = len(nums)
    passes = [{k: v[:n] for k, v in p.items()} for p in rec.passes]
    conv = [c[:n] for c in rec.conv]
    e = float(out["energy"].reshape(-1)[0])
    e_nod3 = float(calc_nod3(dict(data))["energy"].reshape(-1)[0])
    q = out["charges"].detach().reshape(-1).double().numpy()[:n]
    u_coul = coulomb_lr(q, pos, rc_coul)
    res = {
        "E": e,
        "U_disp": e - e_nod3,
        "U_coul": u_coul,
        "U_local": e_nod3 - u_coul,
        "q": q,
        "passes": passes,
        "conv": conv,
    }
    if frag is not None:
        fr = np.asarray(frag)
        res["frag"] = {
            "q": [float(q[fr == k].sum()) for k in (0, 1)],
            "passes": [
                {key: [float(p[key][fr == k].sum()) for k in (0, 1)] for key in ("q_tilde", "f", "q")} for p in passes
            ],
        }
    return res


def _r(x, nd=5):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_r(v, nd) for v in x]
    v = float(x)
    return 0.0 if abs(v) < 1e-12 else round(v, nd)


def main(argv: list[str]) -> int:
    import torch

    ckpt, out_path = Path(argv[0]), Path(argv[1])
    torch.manual_seed(0)
    calc, calc_nod3, meta, sd, digest, n_params = load(ckpt)
    model = calc.model
    rec = Recorder(model)
    rc_coul = float(meta["coulomb_sr_rc"])
    import aimnet  # noqa: F401
    import ase

    # 1. One benzene at fractional total charge
    bz = benzene()
    eq = []
    for Q in CHARGE_GRID:
        r = evaluate(calc, calc_nod3, rec, bz[0], bz[1], Q, rc_coul)
        eq.append({"Q": Q, "E": _r(r["E"], 5), "qC": _r(r["q"][0], 5), "qH": _r(r["q"][6], 5)})

    # 2. Charged dimers pulled apart along z
    dimers = []
    for did, label, a, b, Q in [
        ("bz2", "(C₆H₆)₂⁺, stacked", bz, (bz[0], bz[1]), 1.0),
        ("bzw", "C₆H₆ · H₂O, Q = +1", bz, water_h_down(), 1.0),
        ("bzw0", "C₆H₆ · H₂O, Q = 0", bz, water_h_down(), 0.0),
    ]:
        rows = []
        for d in DIMER_D:
            nums, pos, frag = stack(a, b, d)
            r = evaluate(calc, calc_nod3, rec, nums, pos, Q, rc_coul, frag)
            rows.append(
                {
                    "d": d,
                    "E": _r(r["E"], 5),
                    "U_local": _r(r["U_local"], 5),
                    "U_coul": _r(r["U_coul"], 5),
                    "U_disp": _r(r["U_disp"], 5),
                    "fragQ": _r(r["frag"]["q"], 5),
                    "passes": [{k: _r(v, 5) for k, v in p.items()} for p in r["frag"]["passes"]],
                    "q": _r(r["q"], 4),
                }
            )
        # integer-charge asymptotes from isolated fragments, as the model sees them
        iso = {}
        for name, frag_geom in (("A", a), ("B", b)):
            for qq in (0.0, 1.0, Q / 2):
                iso[f"{name}{qq:g}"] = _r(evaluate(calc, calc_nod3, rec, frag_geom[0], frag_geom[1], qq, rc_coul)["E"], 5)
        dimers.append(
            {
                "id": did,
                "label": label,
                "Q": Q,
                "numbers": list(a[0]) + list(b[0]),
                "fragment": [0] * len(a[0]) + [1] * len(b[0]),
                "A": _r(a[1], 5),
                "B": _r(b[1], 5),
                "isolated": iso,
                "rows": rows,
            }
        )

    # 3. Methane on the D2d path: environment features of each pass
    ref = evaluate(calc, calc_nod3, rec, *methane_d2d(TD_THETA), 0.0, rc_coul)
    path = []
    for th in THETA_DEG:
        r = evaluate(calc, calc_nod3, rec, *methane_d2d(th), 0.0, rc_coul)
        dist = []
        for k, (c, c0) in enumerate(zip(r["conv"], ref["conv"])):
            dist.append(
                {
                    "pass": k,
                    "C": float(np.linalg.norm(c[0] - c0[0]) / np.linalg.norm(c0[0])),
                    "H": float(np.linalg.norm(c[1] - c0[1]) / np.linalg.norm(c0[1])),
                }
            )
        path.append(
            {
                "theta": th,
                "E": _r(r["E"] - ref["E"], 5),
                "qC": _r(r["q"][0], 5),
                "conv": [{k: _r(v, 9) if k != "pass" else v for k, v in x.items()} for x in dist],
            }
        )

    hp = {
        "rc": float(sd["aev.rc_s"]),
        "eta": float(sd["aev.eta_s"]),
        "shifts": _r(sd["aev.shifts_s"].double().numpy(), 6),
        "nfeature": 16,
        "ncomb_v": int(sd["conv_a.agh"].shape[-1]),
        "passes": 3,
        "aim_size": int(sd["mlps.2.6.weight"].shape[0]),
        "coulomb_sr_rc": rc_coul,
        "coulomb_sr_envelope": meta["coulomb_sr_envelope"],
        "d3": meta["d3_params"],
        "species": meta["implemented_species"],
    }
    doc = {
        "format": "atlas-aimnet2",
        "version": 1,
        "status": "PRECOMPUTED",
        "model": {
            "family": "aimnet2",
            "checkpoint": "aimnet2-wb97m-d3",
            "file": CHECKPOINT_NAME,
            "member": 0,
            "url": CHECKPOINT_URL,
            "sha256": digest,
            "licence": "MIT",
            "parameters": n_params,
            "hyperparameters": hp,
        },
        "provenance": {
            "aimnet": AIMNET_COMMIT,
            "torch": torch.__version__,
            "ase": ase.__version__,
            "device": "cpu",
            "dtype": "float32 (as released); energies written to 1e-5 eV",
            "kE": K_E,
            "script": "pipeline/atlas_pipeline/aimnet2.py",
            "methaneRefTheta": TD_THETA,
        },
        "benzeneEQ": eq,
        "dimers": dimers,
        "methane": path,
    }
    out_path.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False) + "\n")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
