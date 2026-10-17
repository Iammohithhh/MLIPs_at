"""Analysis maths of the Representation Map (numpy only; the MACE features need torch and are not tested here)."""

import numpy as np
import pytest

from atlas_pipeline.representation import (
    boxed,
    invariance,
    knn,
    linear_cka,
    mutual_knn,
    nearest_distinct,
    pca2,
    procrustes,
    reflected,
    relative,
    rotated,
    rotation_matrix,
    stratified_anchors,
)

rng = np.random.default_rng(3)
X = rng.normal(size=(60, 8)) @ np.diag([5, 3, 1, 1, 0.5, 0.5, 0.2, 0.1])


def rot(d, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    return q


def test_pca_is_centred_ordered_and_sign_fixed():
    xy, frac = pca2(X)
    assert np.allclose(xy.mean(axis=0), 0)
    assert frac[0] > frac[1] > 0 and sum(frac) < 1
    assert np.var(xy[:, 0]) > np.var(xy[:, 1])
    perm = rng.permutation(len(X))  # reordering the points must not move them
    xy2, _ = pca2(X[perm])
    assert np.allclose(xy[perm], xy2)


def test_cka_is_rotation_and_scale_invariant_and_detects_unrelated_spaces():
    assert linear_cka(X, X) == pytest.approx(1)
    assert linear_cka(X, 3.7 * X @ rot(8, 1)) == pytest.approx(1)
    assert linear_cka(X, rng.normal(size=X.shape)) < 0.5


def test_relative_representation_ignores_rotation_and_scale_of_the_latent_space():
    anchors = np.array([0, 7, 19, 33])
    Z = relative(X, anchors)
    assert Z.shape == (60, 4)
    assert np.allclose(Z[anchors[1], 1], 1)
    assert np.allclose(relative(2.5 * X @ rot(8, 2), anchors), Z)


def test_procrustes_recovers_a_rotated_reflected_scaled_copy():
    ref, _ = pca2(X)
    R = np.array([[0.6, 0.8], [0.8, -0.6]])  # rotation + reflection
    aligned, sim = procrustes(4 * ref @ R + 3, ref)
    assert sim == pytest.approx(1)
    assert np.allclose(aligned, ref - ref.mean(axis=0))
    _, low = procrustes(rng.normal(size=ref.shape), ref)
    assert low < 0.3


def test_knn_and_mutual_knn():
    pts = np.array([[0.0], [1.0], [3.0], [10.0]])
    assert knn(pts, 1)[:, 0].tolist() == [1, 0, 1, 2]
    assert mutual_knn(X, X @ rot(8, 4), 5) == pytest.approx(1)
    assert mutual_knn(X, rng.normal(size=X.shape), 5) < 0.4


def test_anchors_are_stratified_deterministic_and_unique():
    groups = ["Si"] * 10 + ["O"] * 20 + ["H"] * 2
    a = stratified_anchors(groups, 6)
    assert a.tolist() == stratified_anchors(groups, 6).tolist()
    assert len(set(a.tolist())) == 6
    assert sorted({groups[i] for i in a}) == ["H", "O", "Si"]
    assert len(stratified_anchors(groups, 100)) == 32


def test_rotation_is_proper_and_preserves_every_distance():
    R = rotation_matrix()
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1)
    assert not np.allclose(R, np.eye(3))
    from ase.build import bulk, molecule

    for a in (bulk("ZnO", "wurtzite", a=3.25, c=5.21) * (2, 2, 1), molecule("CH3CH2OH")):
        b = rotated(a)
        assert np.allclose(a.get_all_distances(mic=a.pbc.any()), b.get_all_distances(mic=b.pbc.any()))
        assert not np.allclose(a.positions, b.positions)
        if a.pbc.any():
            assert np.allclose(a.cell.cellpar(), b.cell.cellpar())


def test_boxed_puts_molecules_in_a_periodic_box_and_leaves_crystals_alone():
    from ase.build import bulk, molecule

    m = boxed(molecule("H2O"))
    assert m.pbc.all() and np.allclose(m.cell.lengths(), 20)
    assert np.allclose(m.positions.mean(axis=0), 10, atol=0.5)
    si = bulk("Si")
    assert boxed(si) is si
    from ase.build import fcc111

    slab = fcc111("Cu", size=(2, 2, 4), a=3.615, vacuum=8.0)
    b = boxed(slab)
    assert b.pbc.all() and np.allclose(b.positions, slab.positions)
    gap = b.cell[2, 2] - np.ptp(b.positions[:, 2])
    assert gap > 2 * 6.0  # more than any cutoff used here: no edge crosses the vacuum


def test_invariance_is_zero_for_an_invariant_model_and_measured_in_neighbour_units():
    rep = np.array([[0.0, 0], [1, 0], [3, 0], [3, 2]])
    assert nearest_distinct(rep).tolist() == [1, 1, 2, 2]  # median 1.5
    copies = [rep[[0]], np.array([[1.0, 0], [1, 0]]), rep[[2]], rep[[3]]]
    inv = invariance(rep, copies, rep.copy())
    assert inv["d_ref"] == pytest.approx(1.5)
    assert inv["equiv_summary"] == {"n": 1, "median": 0.0, "p95": 0.0, "max": 0.0}
    assert inv["rot_summary"]["max"] == 0
    moved = [rep[[0]], np.array([[1.0, 0], [1, 0.3]]), rep[[2]], rep[[3]]]
    rot = rep + np.array([[0, 0.15], [0, 0], [0, 0], [0, 0]])
    inv = invariance(rep, moved, rot)
    assert inv["equiv_summary"]["max"] == pytest.approx(0.2)
    assert inv["rot"].tolist() == pytest.approx([0.1, 0, 0, 0])


def test_equivalence_needs_more_than_one_cutoff():
    """A bulk-like atom inside a thick MgO slab matches the ideal crystal within 6 Å but not within 25 Å."""
    from ase.build import bulk, surface

    from atlas_pipeline.representation import env_signatures

    mgo = bulk("MgO", "rocksalt", a=4.21, cubic=True)
    slab = surface(mgo, (1, 0, 0), layers=4, vacuum=8.0)
    mid = int(np.argmin(np.abs(slab.positions[:, 2] - slab.positions[:, 2].mean())))
    ref = [i for i in range(len(mgo)) if mgo.numbers[i] == slab.numbers[mid]][0]
    assert env_signatures(slab, 6.0)[mid] == env_signatures(mgo, 6.0)[ref]
    assert env_signatures(slab)[mid] != env_signatures(mgo)[ref]
    sig = env_signatures(mgo)  # every Mg of the perfect crystal is equivalent at any range
    assert len({sig[i] for i in range(len(mgo)) if mgo.numbers[i] == 12}) == 1


def test_reflection_keeps_distances_reverses_handedness_and_keeps_a_right_handed_cell():
    from ase.build import bulk, molecule

    q = bulk("ZnO", "wurtzite", a=3.25, c=5.21) * (2, 2, 1)
    for a in (q, molecule("CH3CH2OH")):
        b = reflected(a)
        assert np.allclose(a.get_all_distances(mic=a.pbc.any()), b.get_all_distances(mic=b.pbc.any()))
        assert np.allclose(b.positions[:, 2], -a.positions[:, 2])
    assert np.linalg.det(np.array(reflected(q).cell)) > 0
    # a chiral quadruple: the signed volume of (r1−r0, r2−r0, r3−r0) changes sign
    m = molecule("CH3CH2OH")
    vol = lambda x: np.linalg.det(x.positions[[1, 2, 3]] - x.positions[0])
    assert vol(reflected(m)) == pytest.approx(-vol(m))
    rep = np.zeros((3, 2)) + np.arange(3)[:, None]
    inv = invariance(rep, [rep[[i]] for i in range(3)], rep, rep + [[0, 0.5], [0, 0], [0, 0]])
    assert inv["ref"].tolist() == pytest.approx([0.5 / inv["d_ref"], 0, 0])
