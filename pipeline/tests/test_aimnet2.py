"""AIMNet2 precompute. The geometry, NQE and Coulomb helpers are pure numpy; the written data file is
checked for internal consistency. Re-running the model needs torch and the aimnet package (see README)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from atlas_pipeline import aimnet2 as am

DATA = Path(__file__).resolve().parents[2] / "atlas" / "src" / "data" / "models" / "aimnet2.json"


def test_methane_path_keeps_bonds_and_cancels_directions():
    for th in (30, am.TD_THETA, 72.5, 90):
        nums, pos = am.methane_d2d(th)
        assert nums == [6, 1, 1, 1, 1]
        r = np.linalg.norm(pos[1:] - pos[0], axis=1)
        assert np.allclose(r, am.CH_BOND, atol=1e-12)
        assert np.linalg.norm((pos[1:] / r[:, None]).sum(0)) < 1e-12
    _, td = am.methane_d2d(am.TD_THETA)
    hh = [np.linalg.norm(td[i] - td[j]) for i in range(1, 5) for j in range(i + 1, 5)]
    assert np.allclose(hh, am.CH_BOND * math.sqrt(8 / 3))


def test_benzene_and_water_geometry():
    nums, pos = am.benzene()
    assert nums.count(6) == 6 and nums.count(1) == 6
    assert np.allclose(pos.mean(0), 0, atol=1e-12)
    assert np.isclose(np.linalg.norm(pos[0] - pos[1]), 1.397)
    wn, wp = am.water_h_down()
    assert wn == [8, 1, 1]
    assert np.isclose(np.linalg.norm(wp[1] - wp[0]), 0.9572) and np.isclose(np.linalg.norm(wp[2] - wp[0]), 0.9572)
    a = np.degrees(np.arccos(np.dot(wp[1] - wp[0], wp[2] - wp[0]) / 0.9572**2))
    assert np.isclose(a, 104.52)
    n, p, f = am.stack((nums, pos), (wn, wp), 4.0)
    assert len(n) == 15 and f.count(1) == 3 and np.isclose(p[13, 2], 4.0)


def test_nqe_conserves_charge_and_uses_weights():
    qt = np.array([0.2, -0.1, 0.3])
    f = np.array([1.0, 3.0, 0.0])
    q = am.nqe(qt, f, -1.0, eps=0.0)
    assert np.isclose(q.sum(), -1.0)
    assert q[2] == qt[2]
    assert np.isclose((q[1] - qt[1]) / (q[0] - qt[0]), 3.0)


def test_exp_cutoff_and_coulomb_tail():
    assert np.isclose(am.exp_cutoff(np.array(0.0), 4.6), 1.0)
    assert am.exp_cutoff(np.array(4.6), 4.6) < 1e-12
    # beyond the envelope the term is the bare Coulomb energy in eV
    pos = np.array([[0.0, 0, 0], [0, 0, 10.0]])
    assert np.isclose(am.coulomb_lr(np.array([1.0, -1.0]), pos, 4.6), -am.K_E / 10)
    assert np.isclose(am.K_E, 14.399645, atol=1e-5)


@pytest.mark.skipif(not DATA.exists(), reason="run the precompute first")
def test_data_file_is_consistent():
    doc = json.loads(DATA.read_text())
    assert doc["model"]["sha256"] == am.CHECKPOINT_SHA256
    assert doc["status"] == "PRECOMPUTED"
    assert [r["Q"] for r in doc["benzeneEQ"]] == am.CHARGE_GRID
    for dm in doc["dimers"]:
        assert [r["d"] for r in dm["rows"]] == am.DIMER_D
        for r in dm["rows"]:
            assert abs(r["U_local"] + r["U_coul"] + r["U_disp"] - r["E"]) < 2e-4
            assert abs(sum(r["fragQ"]) - dm["Q"]) < 1e-4
            assert abs(sum(r["q"]) - dm["Q"]) < 1e-2
    for r in doc["methane"]:
        assert r["conv"][0]["C"] < 1e-5
