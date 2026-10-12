import gzip
import json

import numpy as np
import pandas as pd
import pytest

from sydney_tmd.config import load_config, validate_config
from sydney_tmd.cli import main as cli_main
from sydney_tmd.core import FEATURES, SENSORS, read_json
from sydney_tmd.data import extract, load_session, make_splits, prepare
from sydney_tmd.progress import Attempt


def sensor_frame(seconds=140):
    ts = np.arange(0, seconds * 1000 + 1, 20)
    return pd.DataFrame({"ts": ts, **{c: np.full(len(ts), i + 1, dtype=float)
                                   for i, c in enumerate(c for cols in SENSORS.values() for c in cols)}})


def payload(path, **changes):
    data = dict(session_id="session", participant_id="person", collection_country_code="AU",
                vehicle_type="bus", samples=sensor_frame().to_dict("records"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


@pytest.mark.parametrize("country,reason", [(None, "missing_country"), ("KR", "non_AU"), ("au", "non_AU")])
def test_strict_country(tmp_path, country, reason):
    path = tmp_path / "s.json"
    payload(path, collection_country_code=country)
    assert load_session(path)[2] == reason


def test_classes_duration_trim_and_no_maximum(tmp_path):
    path = tmp_path / "s.json"
    payload(path, vehicle_type="walk")
    assert load_session(path)[2] == "unsupported_class"
    payload(path, trimmed_start_ms=0, trimmed_end_ms=139999)
    assert load_session(path)[2] == "duration_below_140"
    payload(path, samples=sensor_frame(601).to_dict("records"))
    record, _, reason = load_session(path)
    assert reason is None and record["duration_seconds"] == 601


def test_compression_sidecar_conflict_and_missing_identity(tmp_path):
    path = tmp_path / "s.json"
    data = payload(path)
    del data["collection_country_code"]
    compressed = tmp_path / "s.json.gz"
    with gzip.open(compressed, "wt", encoding="utf-8") as stream:
        json.dump(data, stream)
    sidecar = tmp_path / "s.json.gz.metadata.json"
    sidecar.write_text(json.dumps({"collection_country_code": "AU"}))
    assert load_session(compressed)[2] is None
    sidecar.write_text(json.dumps({"collection_country_code": "AU", "participant_id": "different"}))
    with pytest.raises(ValueError, match="Conflicting participant"):
        load_session(compressed)
    payload(path, participant_id=None)
    assert load_session(path)[2] == "missing_participant"


def test_reference_features_overlap_and_quality():
    frame = sensor_frame(240)
    rows, generated = extract(frame, 100)
    assert generated == 3
    assert rows.window_start_ms.tolist() == [0, 50000, 100000]
    assert rows.columns.tolist()[2:] == FEATURES
    magnitudes = [np.linalg.norm([1, 2, 3]), np.linalg.norm([4, 5, 6]), np.linalg.norm([7, 8, 9])]
    expected = [magnitudes[0], 0, 0, magnitudes[0], magnitudes[1], 0, 0, magnitudes[1], 0, 0, magnitudes[2]]
    np.testing.assert_allclose(rows[FEATURES].iloc[0], expected, atol=1e-6)
    frame.loc[frame.ts.between(1000, 1600), ["gx", "gy", "gz"]] = np.nan
    assert len(extract(frame, 140)[0]) == 1  # second window starts after the gap
    frame[["mx", "my", "mz"]] = np.nan
    assert extract(frame, 100)[0].empty


def test_population_std_duplicate_and_boundary_gap():
    frame = sensor_frame()
    frame["ax"] = np.sin(frame.ts / 1000)
    expected = np.linalg.norm(frame.loc[frame.ts < 100000, ["ax", "ay", "az"]], axis=1)
    rows, _ = extract(frame, 100)
    assert rows.iloc[0][FEATURES[1]] == pytest.approx(expected.std(ddof=0), rel=1e-6)
    duplicate = frame.iloc[[1]].copy()
    duplicate["ax"] = 999
    doubled = pd.concat([frame, duplicate]).sort_values("ts", kind="stable")
    np.testing.assert_array_equal(extract(doubled, 100)[0][FEATURES], rows[FEATURES])
    frame.loc[frame.ts < 520, ["mx", "my", "mz"]] = np.nan
    assert extract(frame, 140)[0].empty
    frame = sensor_frame().iloc[::2]  # 25Hz accelerometer, below minimum
    assert extract(frame, 100)[0].empty


def cohort(people=4):
    return [dict(participant_id=f"p{p}", session_id=f"p{p}c{c}", label=c)
            for p in range(people) for c in range(3)]


def test_splits_disjoint_deterministic_and_diagnostic():
    records = cohort()
    splits = make_splits(records)
    assert splits == make_splits(list(reversed(records)))
    assert splits["evaluation_strategy"] == "participant_lopo"
    assert splits["final_train_sessions"] == sorted(r["session_id"] for r in records)
    for outer in splits["outer"]:
        assert outer["train_sessions"] == outer["train_pool_sessions"]
        assert sorted(outer["test_sessions"] + outer["train_sessions"]) == sorted(r["session_id"] for r in records)
        validation = []
        for inner in outer["inner"]:
            assert not set(inner["train"]) & set(inner["validation"])
            assert outer["participant"] not in inner["train"] + inner["validation"]
            validation += inner["validation"]
        assert sorted(validation) == sorted(outer["train"])
    with pytest.raises(ValueError, match="at least 4"):
        make_splits(cohort(3))
    with pytest.raises(ValueError, match="Missing training classes"):
        make_splits([r for r in records if r["label"] != 2 or r["participant_id"] == "p0"])


def test_experiment_config_validation():
    default = load_config()
    assert default["evaluation_strategy"] == "participant_lopo"
    assert set(default["calibration_fraction"].values()) == {1.0}
    for invalid in ({**default, "unknown": 1},
                    {**default, "evaluation_strategy": "window_shuffle"},
                    {**default, "calibration_fraction": {"bus": .5, "car": .5}},
                    {**default, "calibration_fraction": {"bus": 0, "car": .5, "train": .5}},
                    {**default, "calibration_fraction": {"bus": True, "car": .5, "train": .5}}):
        with pytest.raises(ValueError):
            validate_config(invalid)


def test_session_holdout_and_training_fractions_are_frozen_and_disjoint():
    records = [dict(participant_id=f"p{p}", session_id=f"p{p}c{c}n{n}", label=c)
               for p in range(5) for c in range(3) for n in range(2)]
    config = {"schema_version": 1, "evaluation_strategy": "shuffled_session_holdout",
              "calibration_fraction": {"bus": .5, "car": .6, "train": .7}}
    manifest = make_splits(records, config)
    assert manifest == make_splits(list(reversed(records)), config)
    fold = manifest["outer"][0]
    assert fold["test_counts"] == {"bus": 2, "car": 2, "train": 2}
    assert fold["training_pool_counts"] == {"bus": 8, "car": 8, "train": 8}
    assert fold["training_counts"] == {"bus": 4, "car": 4, "train": 5}
    assert not set(fold["test_sessions"]) & set(fold["train_pool_sessions"])
    assert not set(fold["test_sessions"]) & set(fold["train_sessions"])
    validation = []
    for inner in fold["inner"]:
        assert not set(inner["train_sessions"]) & set(inner["validation_sessions"])
        assert not set(inner["train_sessions"]) & set(fold["test_sessions"])
        validation += inner["validation_sessions"]
    assert sorted(validation) == fold["train_pool_sessions"]
    assert manifest["final_training_counts"] == {"bus": 5, "car": 6, "train": 7}


def test_participant_fraction_keeps_all_validation_sessions():
    records = [dict(participant_id=f"p{p}", session_id=f"p{p}c{c}n{n}", label=c)
               for p in range(5) for c in range(3) for n in range(2)]
    config = {"schema_version": 1, "evaluation_strategy": "participant_lopo",
              "calibration_fraction": {"bus": .5, "car": .5, "train": .5}}
    manifest = make_splits(records, config)
    for fold in manifest["outer"]:
        assert fold["training_counts"] == {"bus": 4, "car": 4, "train": 4}
        assert {r["participant_id"] for r in records if r["session_id"] in fold["test_sessions"]} == {fold["participant"]}
        for inner in fold["inner"]:
            training_people = {r["participant_id"] for r in records if r["session_id"] in inner["train_sessions"]}
            validation_people = {r["participant_id"] for r in records if r["session_id"] in inner["validation_sessions"]}
            assert not training_people & validation_people
            assert validation_people == set(inner["validation"])


def test_holdout_rejects_sparse_mode():
    records = [dict(participant_id=f"p{p}", session_id=f"p{p}c{c}", label=c)
               for p in range(3) for c in range(3)]
    config = {"schema_version": 1, "evaluation_strategy": "shuffled_session_holdout",
              "calibration_fraction": {"bus": 1, "car": 1, "train": 1}}
    with pytest.raises(ValueError, match="at least 3 sessions per mode"):
        make_splits(records, config)


def test_prepare_common_cohort_resume_and_identity(tmp_path):
    source, run = tmp_path / "input", tmp_path / "run"
    source.mkdir()
    run.mkdir()
    samples = sensor_frame(200).to_dict("records")
    for record in cohort():
        payload(source / f"{record['session_id']}.json", **{k: record[k] for k in ("session_id", "participant_id")},
                vehicle_type=["bus", "car", "train"][record["label"]], samples=samples)
    # This session has a valid 100-second window but none at 120/140 seconds.
    bad = sensor_frame(140)
    bad.loc[bad.ts >= 110000, ["gx", "gy", "gz"]] = np.nan
    payload(source / "bad.json", session_id="bad", samples=bad.to_dict("records"))
    attempt = Attempt(tmp_path / "logs", "test")
    prepare(source, run, attempt, 1)
    assert read_json(run / "cohort.json")["quality_exclusions"][0]["session_id"] == "bad"
    for window in (100, 120, 140):
        assert set(pd.read_parquet(run / f"features-{window}.parquet").session_id) == {r["session_id"] for r in cohort()}
    marker = read_json(run / "prepared.json")
    assert read_json(run / "experiment.json") == load_config()
    prepare(source, run, Attempt(tmp_path / "logs", "test"), 1)
    assert marker == read_json(run / "prepared.json")
    assert len(list((tmp_path / "logs" / "test").iterdir())) == 2
    with pytest.raises(ValueError, match="new run ID"):
        prepare(source, run, Attempt(tmp_path / "logs", "test"), 2)
    changed = {**load_config(), "calibration_fraction": {"bus": .5, "car": 1, "train": 1}}
    with pytest.raises(ValueError, match="new run ID"):
        prepare(source, run, Attempt(tmp_path / "logs", "test"), 1, changed)
    shuffled = {**changed, "evaluation_strategy": "shuffled_session_holdout"}
    shuffled_path = tmp_path / "shuffled-experiment.json"
    shuffled_path.write_text(json.dumps(shuffled), encoding="utf-8")
    assert cli_main(["prepare", "--run-id", "shuffled", "--runs-dir", str(tmp_path),
                     "--logs-dir", str(tmp_path / "logs"), "--input", str(source),
                     "--config", str(shuffled_path)]) == 0
    custom_splits = read_json(tmp_path / "shuffled" / "splits.json")
    assert custom_splits["evaluation_strategy"] == "shuffled_session_holdout"
    assert custom_splits["outer"][0]["test_counts"] == {"bus": 1, "car": 1, "train": 1}
    config_path = tmp_path / "changed-experiment.json"
    config_path.write_text(json.dumps(changed), encoding="utf-8")
    assert cli_main(["evaluate", "--run-id", "run", "--runs-dir", str(tmp_path),
                     "--logs-dir", str(tmp_path / "logs"), "--config", str(config_path)]) == 1
    (source / "new.json").write_text("{}")
    with pytest.raises(ValueError, match="new run ID"):
        prepare(source, run, Attempt(tmp_path / "logs", "test"), 1)
