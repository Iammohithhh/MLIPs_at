from pathlib import Path

from atlas_pipeline.migration_barriers import build, read_barrier


def write(p: Path, barrier: float):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"Image,Energy (eV)\n0,-1.0\n1,-0.5\n\nMigration Barrier,{barrier}\n", encoding="utf-8")


def test_read_barrier_takes_the_last_barrier_line(tmp_path: Path):
    f = tmp_path / "energies.csv"
    write(f, 0.42)
    assert read_barrier(f) == 0.42


def test_build_joins_dft_targets_with_model_barriers(tmp_path: Path):
    write(tmp_path / "Dataset-2" / "A" / "MACE" / "neb_results.csv", 0.5)
    write(tmp_path / "Dataset-2" / "A" / "CHGNET" / "energies.csv", 0.2)
    write(tmp_path / "Dataset-2" / "B" / "CHGNET" / "energies.csv", 1.3)
    ds = [
        {"sys_name": "A", "formula": "X", "crystal_class": "SPINEL", "target": 0.4},
        {"sys_name": "B", "formula": "Y", "crystal_class": "OLIVINE", "target": 1.0},
    ]
    d = build(ds, tmp_path, "abc")
    a, b = d["systems"]
    assert a["dft"] == 0.4 and a["models"]["MACE-MP-0"] == 0.5 and a["models"]["SevenNet"] is None
    s = d["summary"]["CHGNet"]
    assert s["n"] == 2 and abs(s["mae"] - 0.25) < 1e-12 and (s["under"], s["over"]) == (1, 1)
    assert d["summary"]["MACE-MP-0"]["n"] == 1


def test_hop_finds_the_moving_atom_through_the_periodic_boundary():
    from atlas_pipeline.migration_barriers import hop

    lat = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
    ini = {"lattice_mat": lat, "elements": ["O", "Li", "O"], "coords": [[0.1, 0.1, 0.1], [0.95, 0.5, 0.5], [0.5, 0.5, 0.1]], "cartesian": False}
    fin = {"lattice_mat": lat, "elements": ["O", "Li", "O"], "coords": [[0.1, 0.1, 0.11], [0.25, 0.5, 0.5], [0.5, 0.5, 0.1]], "cartesian": False}
    h = hop({"sys_name": "t", "structure_ini": ini, "structure_fin": fin})
    assert h["mover"] == 1
    assert abs(h["hop"] - 3.0) < 1e-9  # 0.95 → 1.25 across the boundary, not 7 Å back
    assert h["end"] == [12.5, 5.0, 5.0]
