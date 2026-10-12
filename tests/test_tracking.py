"""Run with the isolated MLflow interpreter; no training dependencies are needed."""
import json

import numpy as np
import pytest

pytest.importorskip("mlflow")
from mlflow import MlflowClient
from sydney_tracking.importer import Importer, confusion_matrix, digest, sha256, source_snapshot


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")


def complete(directory, **extra):
    write(directory / "complete.json", {**extra, "files": {p.name: sha256(p) for p in directory.iterdir() if p.name != "complete.json"}})


def source_fixture(source):
    source.mkdir()
    identity = {"code": "historical-code", "environment": {"python": "historical-version"}}
    write(source / "identity.json", identity)
    write(source / "inputs.json", [])
    write(source / "cohort.json", {"participant_count": 4, "sessions": [1, 2, 3]})
    write(source / "splits.json", {"outer": [{"participant": "p1"}]})
    write(source / "prepared.json", {"identity": digest(identity), "files": {n: sha256(source / n) for n in ("cohort.json", "splits.json")}})
    write(source / "evaluated.json", {"prepared": sha256(source / "prepared.json"), "bundles": []})
    (source / "fit-times.jsonl").write_text('{}\n')
    scores = {"macro_f1": .7, "balanced_accuracy": .8}
    write(source / "reports" / "study-summary.json", {"bootstrap_draws": 5000, "results": {"100": {
        "mean": scores, "seed_scores": {"42": scores}, "intervals": {"macro_f1": {"low": .5, "high": .8}}}}})
    (source / "reports" / "confusion-matrices.csv").write_text('result,seed,actual,predicted,windows\n' +
        '\n'.join(f'100,42,{a},{p},{5 if a == p else 1}' for a in range(3) for p in range(3)))
    (source / "reports" / "per-class.csv").write_text('result,seed,label,precision,recall,f1,support\n100,42,0,0.7,0.7,0.7,7\n')
    complete(source / "reports", evaluation_sha256=sha256(source / "evaluated.json"))
    write(source / "studies" / "final-w100-s42.json", {"completed": 2, "trials": [
        {"state": "COMPLETE", "number": i, "value": v, "params": {"max_depth": 3+i}, "seconds": .1}
        for i, v in enumerate((.75, .8))]})
    write(source / "final" / "metadata.json", {"parameters": {"max_depth": 4}, "seed": 42,
          "window_seconds": 100, "selection_scores": {"100": .8}})
    (source / "final" / "pipeline.joblib").write_bytes(b'not-loaded-by-importer')
    complete(source / "final")


def test_import_metrics_history_artifacts_and_idempotence(tmp_path):
    source = tmp_path / "source"
    source_fixture(source)
    before = source_snapshot(source)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    result = Importer(source, uri, "test").run()
    assert result["imported"] == 5
    runs = client.search_runs([experiment_id])
    by_key = {r.data.tags["import.entity"]: r for r in runs}
    assert by_key["result/100"].data.metrics["heldout.macro_f1"] == .7
    assert by_key["final"].data.metrics["selection.window_100.macro_f1"] == .8
    assert not any(k.startswith("heldout") for k in by_key["final"].data.metrics)
    study = by_key["studies/final-w100-s42.json"]
    assert study.data.params["max_depth"] == "4"
    assert [(m.step, m.value) for m in client.get_metric_history(study.info.run_id, "inner.best_macro_f1")] == [(0, .75), (1, .8)]
    assert len(client.list_artifacts(by_key["result/100"].info.run_id, "charts")) == 2
    model = client.download_artifacts(by_key["final"].info.run_id, "final/pipeline.joblib", str(tmp_path))
    from pathlib import Path
    assert Path(model).read_bytes() == b'not-loaded-by-importer'
    again = Importer(source, uri, "test").run()
    assert again["imported"] == 0 and again["reused"] == 5
    assert len(client.search_runs([experiment_id])) == 5
    assert source_snapshot(source) == before


def test_import_shuffled_fractional_run_and_preserve_legacy_support(tmp_path):
    source = tmp_path / "source"
    source_fixture(source)
    config = {"schema_version": 1, "evaluation_strategy": "shuffled_session_holdout",
              "calibration_fraction": {"bus": .5, "car": .5, "train": .5}}
    write(source / "experiment.json", config)
    identity = json.loads((source / "identity.json").read_text())
    identity["configuration"] = config
    write(source / "identity.json", identity)
    write(source / "splits.json", {"evaluation_strategy": "shuffled_session_holdout",
                                   "calibration_fraction": config["calibration_fraction"],
                                   "outer": [{"id": "shuffled", "test_sessions": ["s1"]}]})
    write(source / "prepared.json", {"identity": digest(identity),
                                     "files": {n: sha256(source / n) for n in ("cohort.json", "splits.json", "experiment.json")}})
    write(source / "evaluated.json", {"prepared": sha256(source / "prepared.json"), "bundles": []})
    complete(source / "reports", evaluation_sha256=sha256(source / "evaluated.json"))
    metadata = json.loads((source / "final" / "metadata.json").read_text())
    metadata["calibration_fraction"] = config["calibration_fraction"]
    metadata["evaluation_strategy"] = config["evaluation_strategy"]
    write(source / "final" / "metadata.json", metadata)
    complete(source / "final")
    (source / "studies" / "outer-000-w100-s42.json").write_bytes(
        (source / "studies" / "final-w100-s42.json").read_bytes())
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment("shuffled", artifact_location=(tmp_path / "artifacts").as_uri())
    result = Importer(source, uri, "shuffled").run()
    assert result["imported"] == 6
    by_key = {run.data.tags["import.entity"]: run for run in client.search_runs([experiment_id])}
    assert by_key["summary"].data.params["evaluation_strategy"] == "shuffled_session_holdout"
    assert by_key["studies/outer-000-w100-s42.json"].data.params["outer_fold"] == "shuffled"
    assert "outer_participant" not in by_key["studies/outer-000-w100-s42.json"].data.params
    assert by_key["final"].data.tags["mlflow.runName"] == "Final model — configured cohort"


def test_corrupt_artifact_rejected_before_mlflow_access(tmp_path):
    source = tmp_path / "source"
    source_fixture(source)
    (source / "reports" / "per-class.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        Importer(source, "http://127.0.0.1:1")


def test_failed_import_resumes_same_run(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source_fixture(source)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment("resume", artifact_location=(tmp_path / "artifacts").as_uri())
    original = MlflowClient.log_artifact

    def fail_upload(*args, **kwargs):
        raise OSError("interrupted upload")

    monkeypatch.setattr(MlflowClient, "log_artifact", fail_upload)
    with pytest.raises(OSError, match="interrupted upload"):
        Importer(source, uri, "resume").run()
    failed = client.search_runs([experiment_id])
    assert len(failed) == 1 and failed[0].info.status == "FAILED"
    monkeypatch.setattr(MlflowClient, "log_artifact", original)
    result = Importer(source, uri, "resume").run()
    assert result["run_id"] == failed[0].info.run_id
    assert len(client.search_runs([experiment_id])) == 5
    assert all(r.info.status == "FINISHED" for r in client.search_runs([experiment_id]))


def test_confusion_cells_preserve_seed_and_orientation():
    rows = [{"result": "100", "seed": str(s), "actual": str(a), "predicted": str(p), "windows": str(s+a*3+p)}
            for s in (42, 43) for a in range(3) for p in range(3)]
    np.testing.assert_array_equal(confusion_matrix(rows, "100", 42), np.arange(42, 51).reshape(3, 3))
    with pytest.raises(ValueError, match="nine"):
        confusion_matrix(rows[:-1], "100", 43)
