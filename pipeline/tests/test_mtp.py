import itertools

import numpy as np
import pytest

from atlas_pipeline.mtp import (
    basis_values,
    canonical,
    chebyshev_radial,
    enumerate_basis,
    moment_tensor,
    neighbour_vectors,
    parse_template,
)

RMIN, RCUT = 1.9, 5.0


def diamond_env(a: float = 5.431, rattle: float = 0.0, seed: int = 0) -> np.ndarray:
    lat = a * np.eye(3)
    frac = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0], [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]])
    pos = frac @ lat
    if rattle:
        pos = pos + np.random.default_rng(seed).uniform(-rattle, rattle, pos.shape)
    return neighbour_vectors(lat, pos, RCUT)[0]


def test_enumeration_matches_mlip2_template_sizes():
    # alpha_scalar_moments of untrained_mtps/02.mtp … 28.mtp in mlip-2 (commit fc3b0721)
    expected = [1, 2, 5, 9, 16, 29, 52, 92, 163, 288, 500, 864, 1464, 2445]
    assert [len(enumerate_basis(L)) for L in range(2, 29, 2)] == expected


def test_canonical_label_ignores_node_order():
    assert canonical((0, 0, 0), (1, 1, 0)) == canonical((0, 0, 0), (0, 1, 1)) == canonical((0, 0, 0), (1, 0, 1))
    assert canonical((1, 0), (1,)) == canonical((0, 1), (1,))


def test_diamond_neighbours():
    v = diamond_env()
    r = np.sort(np.linalg.norm(v, axis=1))
    assert np.allclose(r[:4], 5.431 * np.sqrt(3) / 4)
    assert np.allclose(r[4:16], 5.431 / np.sqrt(2))


def test_radial_basis_is_chebyshev_times_cutoff_envelope():
    r = np.array([2.2, 3.7, 4.99, 5.0, 6.0])
    q = chebyshev_radial(r, RMIN, RCUT, 4)
    ksi = (2 * r - RMIN - RCUT) / (RCUT - RMIN)
    env = np.where(r < RCUT, (r - RCUT) ** 2, 0)
    assert np.allclose(q[2], (2 * ksi**2 - 1) * env)
    assert np.allclose(q[3], (4 * ksi**3 - 3 * ksi) * env)
    assert np.all(q[:, 3:] == 0)


def test_basis_values_invariant_and_equal_to_direct_sums():
    v = diamond_env(rattle=0.12, seed=4)
    basis = enumerate_basis(12)
    b0 = basis_values(v, basis, RMIN, RCUT)
    Q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    assert np.allclose(basis_values(v @ Q.T, basis, RMIN, RCUT), b0, rtol=1e-11, atol=1e-9)
    assert np.allclose(basis_values(v[::-1], basis, RMIN, RCUT), b0, rtol=1e-11, atol=1e-9)
    r = np.linalg.norm(v, axis=1)
    f = chebyshev_radial(r, RMIN, RCUT, 1)[0]
    u = v / r[:, None]
    G = u @ u.T
    by = {b["key"]: t for t, b in enumerate(basis)}
    assert b0[by["2|0,0|2"]] == pytest.approx(float(f @ (G**2) @ f), rel=1e-11)
    assert b0[by["3|0,0,0|0,1,1"]] == pytest.approx(float(np.einsum("i,j,k,ij,ik->", f, f, f, G, G)), rel=1e-11)


def test_moment_tensor_is_symmetric():
    v = diamond_env(rattle=0.1, seed=2)
    T = moment_tensor(v, 1, 3, RMIN, RCUT)
    for p in itertools.permutations(range(3)):
        assert np.allclose(T, np.transpose(T, p))


def test_parse_template():
    text = """MTP
version = 1.1.0
species_count = 1
radial_basis_type = RBChebyshev
	radial_basis_size = 8
	radial_funcs_count = 2
alpha_moments_count = 18
alpha_index_basic_count = 3
alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 2, 0}}
alpha_scalar_moments = 9
"""
    t = parse_template(text)
    assert t["alpha_scalar_moments"] == 9 and t["radial_funcs_count"] == 2 and t["mu_max"] == 1 and t["nu_max"] == 2
    with pytest.raises(ValueError):
        parse_template(text.replace("alpha_index_basic_count = 3", "alpha_index_basic_count = 4"))
