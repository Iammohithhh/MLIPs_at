"""Architecture X-ray for NequIP. Needs nequip==0.15.0 and e3nn==0.5.6 (see README); the forward-pass
tests also need the package file named by ATLAS_NEQUIP_PACKAGE (they are skipped without it)."""

import io
import json
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nequip")

from atlas_pipeline import xray_nequip as xn  # noqa: E402
from atlas_pipeline.xray import rotation_matrix  # noqa: E402

STRUCTURES = Path(__file__).resolve().parents[2] / "atlas" / "src" / "data" / "structures"
PKG = os.environ.get("ATLAS_NEQUIP_PACKAGE")
needs_pkg = pytest.mark.skipif(not PKG, reason="set ATLAS_NEQUIP_PACKAGE to NequIP-OAM-S-0.1.nequip.zip")


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def _zip_with(payload: bytes) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("pkg/model/eager_model.pkl", payload)
    return zipfile.ZipFile(io.BytesIO(buf.getvalue()))


def test_restricted_unpickler_never_resolves_executable_globals():
    zf = _zip_with(pickle.dumps(_Evil()))
    obj = xn._unpickler(zf, "pkg")(io.BytesIO(zf.read("pkg/model/eager_model.pkl"))).load()
    assert isinstance(obj, xn._Stub)
    assert obj.args == ("echo pwned",)  # recorded, never run


def test_irreps_stub_parsing():
    Irrep = type("Irrep", (xn._Stub,), {})
    MulIr = type("_MulIr", (xn._Stub,), {})
    Irreps = type("Irreps", (xn._Stub,), {})
    stub = Irreps((MulIr((128, Irrep((0, 1)))), MulIr((64, Irrep((1, 1))))))
    assert xn._irreps(stub) == "128x0e+64x1e"


def test_split_mul_major_layout():
    x = np.arange(2 * (3 + 2 * 3), dtype=float).reshape(2, 9)
    s, v = xn._split(x, "3x0e+2x1e")
    assert s.shape == (2, 3, 1) and v.shape == (2, 2, 3)
    assert v[0, 1].tolist() == [6.0, 7.0, 8.0]
    with pytest.raises(RuntimeError):
        xn._split(x, "3x0e+1x1e")


@pytest.fixture(scope="module")
def built():
    torch.set_default_dtype(torch.float64)
    return xn.build_model(Path(PKG))


def _run(built, atoms):
    model, hp, _ = built
    return xn.summarise(model, atoms, xn.forward_with_hooks(model, atoms, hp), hp)


@needs_pkg
def test_package_hash_and_recovered_architecture(built):
    import hashlib

    assert hashlib.sha256(Path(PKG).read_bytes()).hexdigest() == xn.CHECKPOINT_SHA256
    model, hp, meta = built
    assert hp["r_max"] == 4.5 and hp["num_layers"] == 2
    assert hp["feature_irreps_hidden"] == ["128x0e+64x1e", "128x0e"]
    assert hp["irreps_edge_sh"] == "1x0e+1x1e"
    assert hp["pair_potential"] == "ZBL"
    assert meta["versions"]["nequip"] == "0.15.0"
    # 2π / r_c², the factor that turns sinc(x·w)·w into Eq. (6) of Batzner et al.
    assert hp["edge_factor"] == pytest.approx(2 * np.pi / hp["r_max"] ** 2, rel=1e-12)
    assert sum(xn.parameter_counts(model).values()) == 616360


@needs_pkg
def test_rotation_and_mirror(built):
    atoms = xn._atoms(json.loads((STRUCTURES / "ethanol.json").read_text()))
    a = _run(built, atoms)
    R = rotation_matrix([1, 2, 3], 50)
    rot = atoms.copy()
    rot.positions = atoms.positions @ R.T
    b = _run(built, rot)
    assert abs(a["energy"] - b["energy"]) < 1e-9
    assert np.allclose(np.array(b["forces"]), np.array(a["forces"]) @ R.T, atol=1e-4)
    h, hr = np.array(a["layers"][0]["h"]["l1"]), np.array(b["layers"][0]["h"]["l1"])
    assert np.allclose(hr, h @ R.T, atol=2e-3)
    M = np.diag([1.0, 1.0, -1.0])
    mir = atoms.copy()
    mir.positions = atoms.positions @ M.T
    c = _run(built, mir)
    # No 1 ⊗ 1 → 1 path exists in this checkpoint, so no pseudoscalar can be formed: exact.
    assert abs(a["energy"] - c["energy"]) < 1e-10
    assert np.allclose(np.array(c["layers"][0]["h"]["l1"]), h @ M.T, atol=1e-9)


@needs_pkg
def test_isolated_atom_energy_is_the_shift(built):
    """No edges → zero messages, SiLU(0) = 0 and no biases anywhere, so E = shift(z) exactly."""
    from ase import Atoms

    model, hp, _ = built
    shifts = model.model.func.per_type_energy_scale_shift.shifts.detach().numpy().reshape(-1)
    for sym in ("Si", "O", "Zn"):
        out = xn.forward_with_hooks(model, Atoms(sym, positions=[[0, 0, 0]]), hp)["out"]
        assert float(out["total_energy"]) == pytest.approx(shifts[hp["type_names"].index(sym)], abs=1e-12)


@needs_pkg
def test_site_energies_reassemble(built):
    atoms = xn._atoms(json.loads((STRUCTURES / "zno-wurtzite.json").read_text()))
    s = _run(built, atoms)
    parts = np.array(s["readout"]) * np.array(s["scale"]) + np.array(s["shift"]) + np.array(s["zbl"])
    assert np.allclose(parts, s["nodeEnergy"], atol=1e-3)
    assert sum(s["nodeEnergy"]) == pytest.approx(s["energy"], abs=1e-3)
