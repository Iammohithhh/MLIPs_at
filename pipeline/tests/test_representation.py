"""Analysis maths of the Representation Map (numpy only; the MACE features need torch and are not tested here)."""

import numpy as np
import pytest

from atlas_pipeline.representation import knn, linear_cka, mutual_knn, pca2, procrustes, relative, stratified_anchors

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
