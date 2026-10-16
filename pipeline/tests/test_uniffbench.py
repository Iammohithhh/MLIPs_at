import pytest

from atlas_pipeline.uniffbench import MIN_RUNS, MODELS, build, parse_run_name, read_runs

HEADER = "Filename,Exp_Density (g/cm³),Exp_a (Å),Sim_Density (g/cm³),Sim_a (Å)\n"


def test_parse_run_name():
    assert parse_run_name("0_Thorianite_298.00_1.01_") == ("Thorianite", 298.0, 1.01)
    assert parse_run_name("0_Hydroxylbastnasite-(Ce)_4.20_0.00_") == ("Hydroxylbastnasite-(Ce)", 4.2, 0.0)
    assert parse_run_name("0_Calcybeborosilite_(Y)_298.00_610000.00_")[0] == "Calcybeborosilite_(Y)"
    with pytest.raises(ValueError):
        parse_run_name("Thorianite.cif")


def test_read_runs_copies_densities(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text(HEADER + "0_Quartz_298.00_1.01_,2.648,4.91,2.5012345678,5.0\n", encoding="utf-8")
    (r,) = read_runs(f)
    assert r == {"mineral": "Quartz", "T": 298.0, "p_field": 1.01, "rho_exp": 2.648, "rho_sim": 2.50123}


def test_read_runs_rejects_duplicates_and_bad_values(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text(HEADER + "0_Q_298.00_1.01_,2.6,4.9,2.5,5\n0_Q_298.00_1.01_,2.6,4.9,2.5,5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_runs(f)
    f.write_text(HEADER + "0_Q_298.00_1.01_,0,4.9,2.5,5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_runs(f)


def test_build_requires_complete_files(tmp_path):
    for m in MODELS:
        p = tmp_path / m["file"]
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = "".join(f"0_M{i}_298.00_1.01_,3.0,5,2.9,5.1\n" for i in range(MIN_RUNS))
        p.write_text(HEADER + rows, encoding="utf-8")
    d = build(tmp_path, "abc", "2026-07-25")
    assert d["source"]["commit"] == "abc" and [len(m["runs"]) for m in d["models"]] == [MIN_RUNS] * len(MODELS)
    (tmp_path / MODELS[0]["file"]).write_text(HEADER + "0_M_298.00_1.01_,3,5,2.9,5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build(tmp_path, "abc", "2026-07-25")
