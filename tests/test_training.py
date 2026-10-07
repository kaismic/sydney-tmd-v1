import json

import numpy as np
import optuna
import pandas as pd
import pytest

from sydney_tmd import training
from sydney_tmd.core import FEATURES, read_json, write_json
from sydney_tmd.progress import Progress


def features():
    frame = pd.DataFrame(np.tile(np.arange(11), (12, 1)), columns=FEATURES, dtype=float)
    frame["participant_id"] = np.repeat(["a", "b", "c", "d"], 3)
    frame["label"] = np.tile([0, 1, 2], 4)
    frame["session_id"] = [f"s{i}" for i in range(12)]
    frame["window_start_ms"], frame["window_end_ms"] = 0, 100000
    frame[FEATURES[0]] = frame.label.astype(float)
    return frame


def test_real_model_training_imputation_weights_and_reload(tmp_path, monkeypatch):
    frame = features()
    frame.loc[0, FEATURES[1]] = np.nan
    weights = []
    original = training.compute_sample_weight
    monkeypatch.setattr(training, "compute_sample_weight", lambda policy, labels: weights.append(list(labels)) or original(policy, labels))
    with Progress(tmp_path, "t", "evaluation") as log:
        estimator = training.fit(frame, {"n_estimators": 2, "max_depth": 3}, 42, 1, log, tmp_path / "times.jsonl")
    assert weights == [frame.label.tolist()]
    assert estimator.named_steps["imputer"].statistics_[1] == 1
    assert estimator.named_steps["imputer"].indicator_.features_.tolist() == [1]
    validation = frame.copy()
    validation[FEATURES[1]] = 999
    estimator.predict(validation[FEATURES])
    assert estimator.named_steps["imputer"].statistics_[1] == 1
    import joblib
    joblib.dump(estimator, tmp_path / "model.joblib")
    training.verify_artifact(tmp_path / "model.joblib", frame, estimator)


def test_search_100_completed_resume_and_failure(tmp_path, monkeypatch):
    write_json(tmp_path / "prepared.json", {"test": True})
    calls = []

    def objective(trial, *args):
        calls.append(trial.number)
        params = training.suggest(trial)
        if trial.number == 0:
            raise ValueError("test failure")
        return params["learning_rate"]

    monkeypatch.setattr(training, "inner_objective", objective)
    with Progress(tmp_path / "logs", "t", "evaluation") as log:
        with pytest.raises(ValueError, match="test failure"):
            training.search(tmp_path, "study", features(), [], 42, 1, log)
        winner = training.search(tmp_path, "study", features(), [], 42, 1, log)
        assert len(calls) == 101
        assert training.search(tmp_path, "study", features(), [], 42, 1, log) == winner
        assert len(calls) == 101
    history = read_json(tmp_path / "studies" / "study.json")
    assert history["completed"] == 100
    assert history["trials"][0]["state"] == "FAIL"
    assert set(history["milestones"]) == {"45", "75", "100"}


def test_interrupted_trial_and_search_sequence_reproducible(tmp_path, monkeypatch):
    for name in ("resumed", "continuous"):
        write_json(tmp_path / name / "prepared.json", {"test": True})
    params_before = []

    def interrupt(trial, *args):
        params_before.append(training.suggest(trial))
        raise KeyboardInterrupt()

    monkeypatch.setattr(training, "inner_objective", interrupt)
    with Progress(tmp_path / "logs", "t", "evaluation") as log:
        with pytest.raises(KeyboardInterrupt):
            training.search(tmp_path / "resumed", "study", features(), [], 42, 1, log, budget=12)
        monkeypatch.setattr(training, "inner_objective", lambda trial, *args: training.suggest(trial)["learning_rate"])
        for name in ("resumed", "continuous"):
            training.search(tmp_path / name, "study", features(), [], 42, 1, log, budget=12)
    resumed = read_json(tmp_path / "resumed" / "studies" / "study.json")["trials"]
    continuous = read_json(tmp_path / "continuous" / "studies" / "study.json")["trials"]
    assert resumed[0]["params"] == params_before[0]
    assert [t["params"] for t in resumed] == [t["params"] for t in continuous]


def test_selection_ties_and_fixed_final_seed():
    assert training.choose_window({140: {"score": .8}, 100: {"score": .8}, 120: {"score": .7}}) == 100
    winners = {w: {s: {"score": .8} for s in (42, 43, 44)} for w in (100, 120, 140)}
    assert training.final_window(winners)[0] == 100
    study = optuna.create_study(direction="maximize")
    for _ in range(2):
        trial = study.ask()
        study.tell(trial, .5)
    assert training.winner(study)["trial"] == 0


def test_inner_pools_predictions_and_excludes_outer(tmp_path, monkeypatch):
    frame = features()
    frame = pd.concat([frame, frame[(frame.participant_id == "a") & (frame.label == 0)].loc[lambda x: x.index.repeat(20)]], ignore_index=True)
    fits = []

    class Estimator:
        def predict(self, x):
            return np.zeros(len(x), dtype=int)

    def fake_fit(frame, *args):
        fits.append(set(frame.participant_id))
        return Estimator()

    monkeypatch.setattr(training, "fit", fake_fit)
    folds = [{"train": ["b", "c"], "validation": ["a"]},
             {"train": ["a", "c"], "validation": ["b"]},
             {"train": ["a", "b"], "validation": ["c"]}]
    study = optuna.create_study()
    with Progress(tmp_path, "t", "evaluation") as log:
        value = training.inner_objective(study.ask(), frame[frame.participant_id != "d"], folds, 42, 1, log, tmp_path / "times")
    selected = frame[frame.participant_id != "d"]
    assert value == training.score(selected.label, np.zeros(len(selected)))
    assert value != pytest.approx(np.mean([training.score(part.label, np.zeros(len(part)))
                                          for _, part in selected.groupby("participant_id")]))
    assert fits == [{"b", "c"}, {"a", "c"}, {"a", "b"}]
