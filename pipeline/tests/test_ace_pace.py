"""PACE Cu precompute. Parsing, the embedding and the core repulsion are pure Python; the written data file
is checked for internal consistency. Re-evaluating the potential needs python-ace (see README)."""

import json
import math
from pathlib import Path

from atlas_pipeline import ace_pace as ap

DATA = Path(__file__).resolve().parents[2] / "atlas" / "src" / "data" / "models" / "ace-pace-cu.json"

SNIPPET = """nelements=1
elements: Cu
lmax=6
embedding-function: FinnisSinclairShiftedScaled
4 FS parameters:  1.000000 1.000000 1.000000 0.500000
core energy-cutoff parameters: 100000.0 250.0
E0: 0.0
radbasename=ChebPow
nradbase=16
nradmax=5
cutoff= 7.4
dcut= 0.01
core repulsion parameters: 400.0 7.0
radparameter= 2.0
rankmax=2
ctilde_basis_func: rank=1 ndens=2 mu0=0 mu=( 0 )
n=( 1 )
l=( 0 )
num_ms=1
< 0 >:  21.47  -13.79
ctilde_basis_func: rank=2 ndens=2 mu0=0 mu=( 0  0 )
n=( 1  1 )
l=( 1  1 )
num_ms=2
< 0  0 >:  -0.11  -0.16
< 1  -1 >:  0.22  0.32
"""


def test_parse_and_summarise():
    p = ap.parse_ace(SNIPPET)
    assert [f["rank"] for f in p["functions"]] == [1, 2]
    assert p["functions"][1]["l"] == [1, 1] and p["functions"][1]["num_ms"] == 2
    assert p["functions"][0]["coeffs"] == [[21.47, -13.79]]
    s = ap.summarise(p)
    assert s["densities"] == [{"weight": 1.0, "m": 1.0}, {"weight": 1.0, "m": 0.5}]
    assert s["core_repulsion"] == [400.0, 7.0] and s["cutoff"] == 7.4 and s["radial_basis"] == "ChebPow"
    assert [r["functions"] for r in s["by_rank"]] == [1, 1]


def test_zero_density_keeps_one_column():
    t = ap.zero_density(SNIPPET, (1,))
    c = ap.parse_ace(t)["functions"]
    assert c[0]["coeffs"] == [[21.47, 0.0]] and c[1]["coeffs"] == [[-0.11, 0.0], [0.22, 0.0]]
    t0 = ap.zero_density(SNIPPET, (0,))
    assert ap.parse_ace(t0)["functions"][1]["coeffs"][1] == [0.0, 0.32]


def test_embedding_closed_form():
    assert ap.fs_shifted_scaled(0.0, 0.5) == 0.0
    assert ap.fs_shifted_scaled(2.5, 1.0) == 2.5
    # x_off = 1/4 e^{-|rho|}, y_off = 1/2 e^{-|rho|}: smooth through zero with slope
    # (1/2)(1/4)^(-1/2)(1 - 1/4) + 1/2 = 5/4, and -> sign(rho) sqrt|rho| for large |rho|
    h = 1e-6
    assert abs((ap.fs_shifted_scaled(h, 0.5) - ap.fs_shifted_scaled(-h, 0.5)) / (2 * h) - 1.25) < 1e-5
    assert abs(ap.fs_shifted_scaled(-400.0, 0.5) + 20.0) < 1e-9
    assert ap.fs_shifted_scaled(-3.0, 0.5) == -ap.fs_shifted_scaled(3.0, 0.5)


def test_core_repulsion_closed_form():
    r = 1.0
    assert math.isclose(ap.core_repulsion(r, 400, 7, 7.4), 400 * math.exp(-7) * 0.5 * (1 + math.cos(math.pi / 7.4)))
    assert ap.core_repulsion(7.4, 400, 7, 7.4) == 0.0


def test_data_file():
    d = json.loads(DATA.read_text())
    s = d["summary"]
    assert d["source"]["md5"] == ap.FILE_MD5 and d["source"]["doi"] == ap.ZENODO_DOI
    assert sum(r["functions"] for r in s["by_rank"]) == s["total_functions"] == 742
    assert [r["rank"] for r in s["by_rank"]] == [1, 2, 3, 4, 5, 6]
    assert d["closure_max_eV"] < 1e-8
    for p in d["dimer"]:
        assert abs(p["total"] - (p["linear"] + p["embedding"] + p["core"])) < 1e-8
        assert abs(p["core"] - 2 * ap.core_repulsion(p["r"], 400, 7, 7.4)) < 1e-6 * max(1, p["core"])
    mins = {k: min(v, key=lambda p: p["total"]) for k, v in d["eos"].items()}
    assert mins["fcc"]["total"] < mins["hcp"]["total"] < mins["bcc"]["total"]
    assert d["dimer"][-1]["r"] == 7.4 and abs(d["dimer"][-1]["total"]) < 1e-6
