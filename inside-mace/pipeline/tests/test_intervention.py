"""Checks on the committed Atom Intervention data (fast), and an optional re-evaluation against the real
checkpoint when INSIDE_MACE_CHECKPOINT points at the verified MACE-MP-0a medium file."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[2] / "src" / "data" / "experiments"
MANIFESTS = sorted(DATA.glob("*/manifest.json"))


@pytest.fixture(params=MANIFESTS, ids=lambda p: p.parent.name)
def manifest(request):
    return json.loads(request.param.read_text())


def test_there_is_at_least_one_experiment():
    assert MANIFESTS


def test_baseline_frame_is_the_zero_displacement_and_restores_exactly(manifest):
    d = manifest["displacement"]
    base = manifest["frames"][d["baselineIndex"]]
    assert base["d"] == 0.0 and base["dE"] == 0.0
    r = manifest["checks"]["restoration"]
    assert r["energyAbsDiff"] <= 1e-9 and r["forceMaxAbsDiff"] <= 1e-9


def test_energy_changes_are_differences_of_stored_energies(manifest):
    e0 = manifest["frames"][manifest["displacement"]["baselineIndex"]]["energy"]
    for f in manifest["frames"]:
        if f["status"] == "ok":
            assert abs(f["dE"] - (f["energy"] - e0)) < 1e-9


def test_failed_frames_carry_no_numbers(manifest):
    for f in manifest["frames"]:
        if f["status"] != "ok":
            assert f["error"] and "energy" not in f and "forces" not in f


def test_moved_atom_keeps_its_index_and_moves_along_the_axis(manifest):
    d = manifest["displacement"]
    p0 = np.array(manifest["structure"]["positions"][d["atom"]])
    u = np.array(d["axis"])
    assert abs(np.linalg.norm(u) - 1) < 1e-8
    for f in manifest["frames"]:
        if f["status"] == "ok":
            assert np.allclose(f["movedPosition"], p0 + f["d"] * u, atol=2e-7)


def test_forces_sum_to_zero_in_a_periodic_cell(manifest):
    for f in manifest["frames"]:
        if f["status"] == "ok":
            assert np.abs(np.array(f["forces"]).sum(axis=0)).max() < 1e-5


def test_force_along_path_matches_energy_slope(manifest):
    """F·û = −dE/dd, checked with central differences of the stored frames (spacing Δ, error O(Δ²))."""
    d = manifest["displacement"]
    fr = manifest["frames"]
    u = np.array(d["axis"])
    for k in range(1, len(fr) - 1):
        a, b, c = fr[k - 1], fr[k], fr[k + 1]
        if not all(x["status"] == "ok" for x in (a, b, c)):
            continue
        slope = (c["energy"] - a["energy"]) / (c["d"] - a["d"])
        f = np.array(b["forces"][d["atom"]]) @ u
        assert abs(f + slope) < 0.02 * max(1.0, abs(f)), (k, f, slope)


def test_finite_difference_forces_converge_quadratically(manifest):
    for row in manifest["checks"]["finiteDifference"]:
        errs = [s["absDiff"] for s in row["steps"]]
        assert errs[-1] < 2e-6, row
        if errs[0] > 1e-7:  # above round-off, halving h by ~3.3 should cut the error ~10×
            assert errs[1] < errs[0] / 5


def test_rotation_invariance_and_equivariance(manifest):
    for r in manifest["checks"]["rotation"]["runs"]:
        assert r["energyAbsDiff"] < 1e-8
        assert r["forceEquivarianceMaxAbs"] < 1e-9
        assert max(r["h0InvarianceMaxAbs"]) < 1e-9
        assert all(x is None or x < 1e-9 for x in r["h1EquivarianceMaxAbs"])


def test_frame_chunks_exist_and_match(manifest):
    root = DATA / manifest["id"]
    for f in manifest["frames"]:
        if f["status"] != "ok":
            continue
        chunk = json.loads((root / f["chunk"]).read_text())
        assert chunk["k"] == f["k"] and chunk["d"] == f["d"] and chunk["local"] == manifest["local"]
        assert len(chunk["layers"][0]["h0"]) == len(manifest["local"])


def test_unpickler_never_resolves_executable_globals():
    from inside_mace.capture import _Stub, _Unpickler
    import io
    import pickle

    payload = pickle.dumps(os.system)
    obj = _Unpickler(io.BytesIO(payload)).load()
    assert isinstance(obj, type) and issubclass(obj, _Stub)


@pytest.mark.skipif(not os.environ.get("INSIDE_MACE_CHECKPOINT"), reason="set INSIDE_MACE_CHECKPOINT to re-evaluate")
def test_stored_baseline_matches_a_fresh_evaluation(manifest):
    from ase import Atoms
    from inside_mace.capture import build_model
    from inside_mace.intervention import evaluate

    model, _ = build_model(Path(os.environ["INSIDE_MACE_CHECKPOINT"]))
    s = manifest["structure"]
    k = manifest["displacement"]["baselineIndex"] + 4
    f = manifest["frames"][k]
    pos = np.array(s["positions"])
    pos[manifest["displacement"]["atom"]] = f["movedPosition"]
    ev = evaluate(model, Atoms(symbols=s["symbols"], positions=pos, cell=s["cell"], pbc=True))
    assert abs(ev["energy"] - f["energy"]) < 1e-5
    assert np.abs(ev["forces"] - np.array(f["forces"])).max() < 1e-5
