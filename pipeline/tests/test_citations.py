import numpy as np

from atlas_pipeline.citations import build, equalise, layout


def test_layout_is_seeded_finite_and_equalised():
    edges = [(0, 1), (1, 2), (2, 0), (3, 4)]
    a = layout(6, edges, iters=40)
    assert np.allclose(a, layout(6, edges, iters=40))
    assert np.isfinite(a).all()
    r = np.sort(np.linalg.norm(a, axis=1))
    assert np.allclose(r, ((np.arange(6) + 0.5) / 6) ** (1 / 3))


def test_equalise_keeps_directions():
    p = np.array([[1.0, 0, 0], [0, 10.0, 0], [0, 0, -0.1]])
    q = equalise(p)
    for u, v in zip(p, q):
        assert np.allclose(u / np.linalg.norm(u), v / np.linalg.norm(v))


def test_build_links_only_works_in_the_universe():
    bib = {"papers": {"atlas-a": {"openalex": {"id": "https://openalex.org/W1", "referenced_works": ["https://openalex.org/W2"]}}}}
    works = {
        "W1": {"id": "https://openalex.org/W1", "display_name": "A", "publication_year": 2020, "cited_by_count": 5, "referenced_works": ["https://openalex.org/W2", "https://openalex.org/W99"]},
        "W2": {"id": "https://openalex.org/W2", "display_name": "B", "publication_year": 2010, "cited_by_count": 50, "referenced_works": []},
    }
    d = build(bib, works)
    assert [n["id"] for n in d["nodes"]] == ["W1", "W2"]
    assert d["nodes"][0]["atlas"] == "atlas-a" and d["nodes"][1]["atlas"] is None
    assert d["edges"] == [(0, 1)]
