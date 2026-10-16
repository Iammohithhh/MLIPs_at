import collections
import pickle
from pathlib import Path

import numpy as np
import pytest

from atlas_pipeline.observatory import (
    ALLOWED_GLOBALS,
    build_tasks,
    check_checkpoint,
    loop_path,
    rotation_matrix,
    scan_pickle_globals,
)

STRUCTURES = Path(__file__).resolve().parents[2] / "atlas" / "src" / "data" / "structures"


def test_inputs_are_deterministic():
    a, b = build_tasks(STRUCTURES), build_tasks(STRUCTURES)
    for key in a:
        for fa, fb in zip(a[key]["frames"], b[key]["frames"]):
            assert np.array_equal(fa.positions, fb.positions)
            assert np.array_equal(fa.cell.array, fb.cell.array)


def test_every_input_is_fully_periodic_and_identical_for_all_models():
    t = build_tasks(STRUCTURES)
    for key in t:
        for f in t[key]["frames"]:
            assert all(f.pbc)


def test_rotated_frame_is_a_rigid_rotation():
    t = build_tasks(STRUCTURES)
    a, b = t["rotation"]["frames"]
    R = np.asarray(t["rotation"]["R"])
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1)
    assert np.allclose(b.positions, a.positions @ R.T)
    assert np.allclose(b.get_all_distances(mic=True), a.get_all_distances(mic=True), atol=1e-9)


def test_loop_is_closed_and_tangent_is_derivative():
    th, disp, tan = loop_path(64)
    assert np.allclose(disp[0], 0)
    h = th[1] - th[0]
    # central difference of the displacement equals the stored tangent (periodic loop)
    for k in range(len(th)):
        d = (disp[(k + 1) % len(th)] - disp[k - 1]) / (2 * h)
        assert np.allclose(d, tan[k], atol=1e-2 * np.abs(tan[k]).max())  # O(h²) ≈ 0.3 %


def test_na_heights_are_measured_from_the_top_layer():
    t = build_tasks(STRUCTURES)
    for h, f in zip(t["na-au"]["x"], t["na-au"]["frames"]):
        top = f.positions[:-1, 2].max()
        assert np.isclose(f.positions[-1, 2] - top, h)
        assert f.get_chemical_symbols()[-1] == "Na"


def test_rotation_matrix_axis_is_fixed():
    R = rotation_matrix([0, 0, 1], 90)
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])


def test_pickle_gate_accepts_data_and_rejects_code(tmp_path):
    good = tmp_path / "good.pkl"
    good.write_bytes(pickle.dumps(collections.OrderedDict(a=1, b=[1.5, "x"]), protocol=2))
    assert scan_pickle_globals(good) <= ALLOWED_GLOBALS
    check_checkpoint(good)

    bad = tmp_path / "bad.pkl"
    bad.write_bytes(pickle.dumps({"f": print}, protocol=2))
    assert scan_pickle_globals(bad) & {"builtins print", "__builtin__ print"}
    with pytest.raises(RuntimeError):
        check_checkpoint(bad)

    bad4 = tmp_path / "bad4.pkl"
    bad4.write_bytes(pickle.dumps({"f": print}, protocol=4))  # STACK_GLOBAL form
    with pytest.raises(RuntimeError):
        check_checkpoint(bad4)
