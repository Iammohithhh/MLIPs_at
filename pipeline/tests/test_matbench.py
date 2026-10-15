from pathlib import Path

from atlas_pipeline.matbench import model_record, snapshot

YML = """
model_name: Test-Model
model_key: test-model
model_version: v1
dates: { benchmark_added: '2024-01-02' }
openness: OSOD
training_sets: [MPtrj]
model_params: 1234
license: { code: MIT, checkpoint: MIT }
metrics:
  discovery:
    unique_prototypes: { F1: 0.5, DAF: 2.0, MAE: 0.07, TP: 10 }
    full_test_set: { F1: 0.45 }
  geo_opt:
    symprec=1e-5: { rmsd: 0.09 }
  phonons:
    kappa_103: { κ_SRME: 0.6 }
"""


def test_model_record_copies_numbers_verbatim(tmp_path: Path):
    d = tmp_path / "models" / "fam"
    d.mkdir(parents=True)
    f = d / "test-model.yml"
    f.write_text(YML, encoding="utf-8")
    r = model_record(f)
    assert r is not None
    assert r["metrics"]["discovery_unique"] == {"F1": 0.5, "DAF": 2.0, "MAE": 0.07}
    assert r["metrics"]["discovery_full"] == {"F1": 0.45}
    assert r["metrics"]["rmsd"] == 0.09
    assert r["metrics"]["kappa_srme"] == 0.6
    assert r["training_sets"] == ["MPtrj"]
    assert r["family_dir"] == "fam"


def test_models_without_discovery_metrics_are_skipped(tmp_path: Path):
    d = tmp_path / "models" / "fam"
    d.mkdir(parents=True)
    (d / "x.yml").write_text("model_name: X\nmetrics: {}\n", encoding="utf-8")
    assert model_record(d / "x.yml") is None
    assert snapshot(tmp_path)["models"] == []
