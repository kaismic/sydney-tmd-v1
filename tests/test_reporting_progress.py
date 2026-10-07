import json
import time
import warnings

import numpy as np
import pandas as pd
import pytest

from sydney_tmd.core import FEATURES, PROTOCOL, code_fingerprint, digest, environment, file_hash, read_json, write_json
from sydney_tmd.data import make_splits
from sydney_tmd.progress import Attempt, Progress
from sydney_tmd import reporting, training


def predictions():
    return pd.DataFrame([dict(seed=s, participant_id=p, session_id=f"{p}-{c}", label=c,
                              prediction=c if p == "a" else 0)
                         for s in (42, 43, 44) for p in ("a", "b") for c in (0, 1, 2) for _ in range(c+1)])


def test_metrics_bootstrap_cluster_determinism():
    frame = predictions()
    intervals, draws = reporting.bootstrap(frame, draws=100)
    intervals2, draws2 = reporting.bootstrap(frame.sample(frac=1, random_state=7), draws=100)
    assert intervals == intervals2
    np.testing.assert_array_equal(draws, draws2)
    # Repeating every window preserves cluster metrics and bootstrap outcomes.
    _, repeated = reporting.bootstrap(pd.concat([frame, frame]), draws=100)
    np.testing.assert_allclose(draws, repeated)
    assert reporting.pooled_metrics(frame)["macro_f1"] == training.score(frame.label, frame.prediction)
    assert reporting.matrix_metrics(np.diag([2, 0, 0]))["macro_f1"] == pytest.approx(1/3)


def test_progress_heartbeat_terminal_file_failure_and_interruption(tmp_path, capsys):
    with Progress(tmp_path, "run", "evaluation", interval=.02) as log:
        log.task("long fit", seed=42, outer_participant="a", window=100)
        warnings.warn("test sensor warning", UserWarning)
        time.sleep(.07)
        log.event("fit complete", completed=1, total=3, output="model")
    lines = (tmp_path / "evaluation.log").read_text(encoding="utf-8").splitlines()
    assert capsys.readouterr().out.splitlines() == lines
    events = [json.loads(line) for line in lines]
    assert any(e["message"].startswith("heartbeat") and e["task"] == "long fit" for e in events)
    assert any(e["severity"] == "WARNING" and e["message"] == "test sensor warning" for e in events)
    assert all("+" in e["timestamp"][10:] or "-" in e["timestamp"][10:] for e in events)
    for error in (ValueError("failed"), KeyboardInterrupt()):
        with pytest.raises(type(error)):
            with Progress(tmp_path, "run", "ingestion") as log:
                log.task("bad input")
                raise error
    text = (tmp_path / "ingestion.log").read_text()
    assert "stage failed" in text and "stage interrupted" in text and "traceback" in text
    assert "resum" in text.lower()


def test_log_creation_and_write_failure(tmp_path, capsys):
    (tmp_path / "blocked").write_text("file")
    with pytest.raises(OSError):
        with Progress(tmp_path / "blocked", "run", "reporting"):
            pass
    assert "cannot create log" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="logging failed"):
        with Progress(tmp_path, "run", "reporting") as log:
            log.stream.close()
            log.event("cannot save")
    assert "cannot write log" in capsys.readouterr().err


def prepared_fixture(run):
    run.mkdir()
    cohort = [dict(participant_id=f"p{p}", session_id=f"p{p}-{c}", label=c)
              for p in range(4) for c in range(3)]
    identity = dict(protocol=PROTOCOL, code=code_fingerprint(), environment=environment(), threads=1)
    write_json(run / "identity.json", identity)
    write_json(run / "cohort.json", {"sessions": cohort})
    write_json(run / "splits.json", make_splits(cohort))
    files = ["cohort.json", "splits.json"]
    for w in (100, 120, 140):
        data = pd.DataFrame(cohort)
        for i, feature in enumerate(FEATURES):
            data[feature] = data.label + i / 10
        data["window_start_ms"], data["window_end_ms"] = 0, w*1000
        name = f"features-{w}.parquet"
        data.to_parquet(run / name, index=False)
        files.append(name)
    write_json(run / "prepared.json", {"identity": digest(identity), "files": {n: file_hash(run / n) for n in files}, "planned_fits": 13537})


def test_end_to_end_real_xgboost_nested_report_final_resume(tmp_path, monkeypatch):
    run = tmp_path / "run"
    prepared_fixture(run)
    # Test-only reductions; the CLI exposes no reduced-budget production experiment.
    original_search, original_bootstrap = training.search, reporting.bootstrap
    monkeypatch.setattr(training, "search", lambda *a, **kw: original_search(*a, **kw, budget=1))
    monkeypatch.setattr(training, "suggest", lambda trial: {"n_estimators": trial.suggest_int("n_estimators", 2, 2)})
    monkeypatch.setattr(reporting, "bootstrap", lambda rows, **kw: original_bootstrap(rows, draws=20, **kw))
    attempt = Attempt(tmp_path / "logs", "smoke")
    with attempt.stage("evaluation") as log:
        training.evaluate(run, log)
    with attempt.stage("reporting") as log:
        reporting.report(run, log)
    with attempt.stage("evaluation") as log:
        training.train_final(run, log)
    summary = read_json(run / "reports" / "study-summary.json")
    assert set(summary["results"]) == {"100", "120", "140", "nested"}
    metadata = read_json(run / "final" / "metadata.json")
    assert metadata["seed"] == 42 and metadata["window_seconds"] == 100
    assert metadata["ordered_features"] == FEATURES
    assert metadata["reload_parity"].startswith("exact")
    assert metadata["independent_test_cohort"] is False
    predictions = pd.read_parquet(run / "reports" / "held-out-predictions.parquet")
    nested = predictions[predictions.result == "nested"].reset_index(drop=True)
    fixed = predictions[predictions.result == "100"].reset_index(drop=True)
    pd.testing.assert_frame_equal(nested.drop(columns="result"), fixed.drop(columns="result"))
    fits_before = (run / "fit-times.jsonl").read_text()
    with Attempt(tmp_path / "logs", "smoke").stage("evaluation") as log:
        training.evaluate(run, log)
        training.train_final(run, log)
    with attempt.stage("reporting") as log:
        reporting.report(run, log)
    assert (run / "fit-times.jsonl").read_text() == fits_before
    # Reject tampered caches before using any data for training.
    (run / "features-100.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Prepared artifact changed"):
        with attempt.stage("evaluation") as log:
            training.evaluate(run, log)
