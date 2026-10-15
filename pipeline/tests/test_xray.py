"""Architecture X-ray pipeline. Needs requirements-xray.txt; the forward-pass tests also need
the checkpoint file named by ATLAS_XRAY_CHECKPOINT (they are skipped without it)."""

import io
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mace")

from atlas_pipeline import xray  # noqa: E402

STRUCTURES = Path(__file__).resolve().parents[2] / "atlas" / "src" / "data" / "structures"
CKPT = os.environ.get("ATLAS_XRAY_CHECKPOINT")
needs_ckpt = pytest.mark.skipif(not CKPT, reason="set ATLAS_XRAY_CHECKPOINT to the MACE-MP-0a medium file")


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_restricted_unpickler_never_resolves_executable_globals():
    obj = xray._Unpickler(io.BytesIO(pickle.dumps(_Evil()))).load()
    assert isinstance(obj, xray._Stub)
    assert obj.args == ("echo pwned",)  # recorded, never run


def test_e3nn_l1_order_is_cartesian():
    from e3nn import o3

    sh = o3.SphericalHarmonics(o3.Irreps.spherical_harmonics(3), normalize=True, normalization="component")
    y1 = sh(torch.eye(3, dtype=torch.float64))[:, 1:4]
    assert torch.allclose(y1[:, xray.L1_TO_XYZ], np.sqrt(3) * torch.eye(3, dtype=torch.float64))


@pytest.fixture(scope="module")
def model():
    return xray.build_model(Path(CKPT))[0]


def _run(model, atoms):
    return xray.summarise(model, atoms, xray.forward_with_hooks(model, atoms))


@needs_ckpt
def test_rotation_invariance_and_equivariance_of_real_features(model):
    atoms = xray._atoms(json.loads((STRUCTURES / "ethanol.json").read_text()))
    R = xray.rotation_matrix([1, 2, 3], 50)
    rot = atoms.copy()
    rot.positions = atoms.positions @ R.T
    a, b = _run(model, atoms), _run(model, rot)
    assert abs(a["energy"] - b["energy"]) < 1e-9
    assert np.allclose(np.array(b["forces"]), np.array(a["forces"]) @ R.T, atol=1e-4)
    assert np.allclose(a["layers"][0]["h"]["l0"], b["layers"][0]["h"]["l0"], atol=1e-3)
    v, w = np.array(a["layers"][0]["h"]["l1"]), np.array(b["layers"][0]["h"]["l1"])
    assert np.allclose(w, v @ R.T, atol=2e-4)


@needs_ckpt
def test_site_energies_reassemble_the_node_energy(model):
    atoms = xray._atoms(json.loads((STRUCTURES / "zno-wurtzite.json").read_text()))
    s = _run(model, atoms)
    scale, shift = float(model.scale_shift.scale), float(model.scale_shift.shift)
    raw = np.sum([L["siteEnergy"] for L in s["layers"]], axis=0)
    assert np.allclose(s["nodeEnergy"], np.array(s["e0"]) + scale * raw + shift, atol=1e-3)
    assert abs(sum(s["nodeEnergy"]) - s["energy"]) < 1e-3


@needs_ckpt
def test_diamond_symmetry_zeros_survive_the_network(model):
    s = _run(model, xray._atoms(json.loads((STRUCTURES / "si-diamond.json").read_text())))
    first = s["layers"][0]["A"]
    assert first["peak"][1] < 1e-12 and first["peak"][2] < 1e-12 and first["peak"][3] > 0.1
    assert np.abs(np.array(s["forces"])).max() < 1e-10
