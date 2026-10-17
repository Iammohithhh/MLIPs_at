"""Architecture X-ray for MACELES-OFF. The closed-form tests need only torch; the forward-pass tests
also need mace-torch (requirements-xray.txt) and the checkpoint named by ATLAS_MACELES_CHECKPOINT."""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas_pipeline import xray_les as xl  # noqa: E402
from atlas_pipeline.xray import rotation_matrix  # noqa: E402

STRUCTURES = Path(__file__).resolve().parents[2] / "atlas" / "src" / "data" / "structures"
CKPT = os.environ.get("ATLAS_MACELES_CHECKPOINT")
needs_ckpt = pytest.mark.skipif(not CKPT, reason="set ATLAS_MACELES_CHECKPOINT to MACELES-OFF_small_converted.model")

NF = 90.0474


def test_two_far_charges_obey_coulomb():
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 30.0]], dtype=torch.float64)
    q = torch.tensor([0.7, -0.4], dtype=torch.float64)
    e = xl.les_energy(q, pos, 1.0, NF)
    assert float(e) == pytest.approx(NF / (2 * math.pi) * 0.7 * -0.4 / 30.0, rel=1e-12)


def test_kernel_is_erf_of_r_over_root2_sigma():
    r = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float64)
    expect = [math.erf(x / math.sqrt(2) / 0.8) / x for x in r.tolist()]
    assert xl.les_kernel(r, 0.8).tolist() == pytest.approx(expect, rel=1e-13)


def test_energy_is_even_in_the_charges_and_matches_half_q_phi():
    g = torch.Generator().manual_seed(1)
    pos = torch.randn(7, 3, generator=g, dtype=torch.float64) * 2
    q = torch.randn(7, generator=g, dtype=torch.float64)
    e = xl.les_energy(q, pos, 1.0, NF)
    assert float(xl.les_energy(-q, pos, 1.0, NF)) == pytest.approx(float(e), rel=1e-14)
    assert float(0.5 * (q * xl.les_potential(q, pos, 1.0, NF)).sum()) == pytest.approx(float(e), rel=1e-14)
    # brute-force pair sum
    brute = sum(
        q[i] * q[j] * math.erf(float((pos[i] - pos[j]).norm()) / math.sqrt(2)) / float((pos[i] - pos[j]).norm())
        for i in range(7)
        for j in range(i + 1, 7)
    )
    assert float(e) == pytest.approx(NF / (2 * math.pi) * float(brute), rel=1e-12)


def test_mean_removed_polarization_is_origin_independent():
    g = torch.Generator().manual_seed(2)
    pos = torch.randn(5, 3, generator=g, dtype=torch.float64)
    q = torch.randn(5, generator=g, dtype=torch.float64)  # net charge ≠ 0
    shift = torch.tensor([3.0, -2.0, 7.0], dtype=torch.float64)
    a, b = xl.polarization(q, pos), xl.polarization(q, pos + shift)
    assert torch.allclose(a, b, atol=1e-12)
    assert not torch.allclose(xl.polarization(q, pos, False), xl.polarization(q, pos + shift, False))


def test_stored_constant_is_lower_than_coulombs():
    assert NF / (2 * math.pi) == pytest.approx(14.3315, abs=1e-4)
    assert xl.HALF_INV_EPS0 == pytest.approx(90.4756, abs=1e-3)


@pytest.fixture(scope="module")
def built():
    pytest.importorskip("mace")
    return xl.build_model(Path(CKPT))


@needs_ckpt
def test_rebuild_reads_the_stored_les_settings(built):
    _model, _readouts, hp = built
    assert hp["les"]["sigma"] == 1.0
    assert hp["les"]["norm_factor"] == pytest.approx(NF)
    assert hp["les"]["use_atomwise"] is False and hp["les"]["bec_remove_mean"] is True
    assert hp["r_max"] == 4.5 and hp["num_interactions"] == 2 and hp["hidden_irreps"] == "192x0e+192x1o"


@needs_ckpt
def test_rotation_leaves_charges_and_energies_unchanged(built):
    model, readouts, hp = built
    s = json.loads((STRUCTURES / "ethanol.json").read_text())
    a = xl._atoms(s["symbols"], s["positions"])
    R = rotation_matrix([1, 2, 3], 50)
    b = xl._atoms(s["symbols"], a.positions @ R.T)
    ra, rb = xl.run(model, readouts, hp, a), xl.run(model, readouts, hp, b)
    assert np.abs(ra["q"] - rb["q"]).max() < 1e-9
    assert abs(ra["eSr"] - rb["eSr"]) < 1e-8 and abs(ra["eLr"] - rb["eLr"]) < 1e-8
    assert np.abs(ra["fLr"] @ R.T - rb["fLr"]).max() < 1e-7
    assert np.abs(ra["Z"].sum(0)).max() < 1e-10  # acoustic sum rule from the mean removal


@needs_ckpt
def test_waters_without_a_shared_edge_couple_only_through_les(built):
    model, readouts, hp = built
    scan = xl.water_dimer_scan(model, readouts, hp, [5.5, 12.0])
    A, B = scan["monomers"]
    for row in scan["rows"]:
        assert row["minGap"] > hp["r_max"]  # no edge between the waters: two separate graphs
        assert abs(row["eSr"] - A["eSr"] - B["eSr"]) < 1e-8
        assert np.abs(np.array(row["q"]) - np.array(A["q"] + B["q"])).max() < 1e-4  # rounded to 5 s.f.
        assert abs(row["eLr"] - A["eLr"] - B["eLr"]) > 1e-6


@needs_ckpt
def test_parameter_count(built):
    model, readouts, _hp = built
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 1888080
    assert sum(p.numel() for p in readouts.parameters()) == 192 + 192 * 16 + 16
