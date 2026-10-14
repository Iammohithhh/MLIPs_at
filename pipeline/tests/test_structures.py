import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from atlas_pipeline.structures import RECIPES, build_all, validate


def by_id():
    return {s["id"]: s for s in build_all()}


def atoms(s):
    return Atoms(s["symbols"], positions=s["positions"], cell=s["cell"] or np.zeros((3, 3)), pbc=s["pbc"])


def test_every_recipe_builds_and_validates():
    out = build_all()
    assert [s["id"] for s in out] == [r.id for r in RECIPES]
    for s in out:
        validate(s)
        assert s["provenance"]["source"]


def test_nacl_is_octahedral_with_r0_2_82():
    a = atoms(by_id()["nacl-rocksalt"])
    i, j, d = neighbor_list("ijd", a, 3.0)
    for k in range(len(a)):
        assert (i == k).sum() == 6
    assert np.allclose(d, 2.82, atol=1e-6)


def test_si_is_tetrahedral():
    a = atoms(by_id()["si-diamond"])
    i, j, d = neighbor_list("ijd", a, 2.5)
    assert all((i == k).sum() == 4 for k in range(len(a)))
    assert np.allclose(d, np.sqrt(3) * 5.431 / 4, atol=1e-6)


def test_slab_is_periodic_in_plane_only_with_vacuum():
    s = by_id()["au111-slab"]
    assert s["pbc"] == [True, True, False]
    z = np.array(s["positions"])[:, 2]
    assert len(z) == 64
    assert s["cell"][2][2] - (z.max() - z.min()) >= 15.9


def test_wurtzite_cell_is_hexagonal():
    a = atoms(by_id()["zno-wurtzite"])
    assert np.isclose(a.cell.angles()[2], 120.0)


def test_molecules_are_open():
    for mid in ("h2o", "benzene", "ethanol"):
        s = by_id()[mid]
        assert s["cell"] is None and s["pbc"] == [False, False, False]
