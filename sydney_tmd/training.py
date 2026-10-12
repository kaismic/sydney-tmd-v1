from __future__ import annotations

import json
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .core import (FEATURES, LABELS, PROTOCOL, SEEDS, WINDOWS, atomic_write, digest,
                   environment, file_hash, read_json, verify_prepared, write_json)


def score(y, pred):
    return float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))


def suggest(trial):
    return dict(n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
                learning_rate=trial.suggest_float("learning_rate", .02, .2, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                min_child_weight=trial.suggest_float("min_child_weight", 1, 10),
                subsample=trial.suggest_float("subsample", .6, 1),
                colsample_bytree=trial.suggest_float("colsample_bytree", .5, 1),
                gamma=trial.suggest_float("gamma", 0, 5),
                reg_alpha=trial.suggest_float("reg_alpha", 0, 2),
                reg_lambda=trial.suggest_float("reg_lambda", .5, 10))


def model(params, seed, threads):
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                     ("model", XGBClassifier(**params, objective="multi:softprob", eval_metric="mlogloss",
                                             tree_method="hist", device="cpu", random_state=seed,
                                             n_jobs=threads))])


def fit(frame, params, seed, threads, log, timing_path):
    if set(frame.label) != {0, 1, 2}:
        raise ValueError("Training partition must contain all three classes")
    estimator = model(params, seed, threads)
    start = time.monotonic()
    estimator.fit(frame[FEATURES], frame.label,
                  model__sample_weight=compute_sample_weight("balanced", frame.label))
    seconds = time.monotonic() - start
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    with timing_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({**log.context, "fit_seconds": seconds, "training_windows": len(frame)}) + "\n")
        stream.flush()
    log.event("model fit complete", fit_seconds=seconds, training_windows=len(frame))
    return estimator


def inner_objective(trial, frame, folds, seed, threads, log, timing_path):
    params = suggest(trial)
    actual, predicted = [], []
    for index, fold in enumerate(folds, 1):
        if "train" in fold and set(fold["train"]) & set(fold["validation"]):
            raise ValueError("Participant leakage in inner split")
        if "train_sessions" in fold:
            if set(fold["train_sessions"]) & set(fold["validation_sessions"]):
                raise ValueError("Session leakage in inner split")
            train = frame[frame.session_id.isin(fold["train_sessions"])]
            validation = frame[frame.session_id.isin(fold["validation_sessions"])]
        else:
            train = frame[frame.participant_id.isin(fold["train"])]
            validation = frame[frame.participant_id.isin(fold["validation"])]
        if validation.empty:
            raise ValueError("Empty inner validation partition")
        log.task("inner fit", inner_fold=index, inner_folds=3, trial=trial.number)
        estimator = fit(train, params, seed, threads, log, timing_path)
        log.task("inner prediction", inner_fold=index)
        actual.extend(validation.label.tolist())
        predicted.extend(estimator.predict(validation[FEATURES]).tolist())
        log.event("inner fold complete", completed=index, total=3)
    return score(actual, predicted)


def winner(study):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise ValueError("Study has no completed trials")
    best = min(completed, key=lambda t: (-t.value, t.number))
    return {"trial": best.number, "score": best.value, "params": best.params}


def choose_window(winners):
    return min(winners, key=lambda w: (-winners[w]["score"], int(w)))


def study_history(study, path):
    trials, completed, milestones = [], [], {}
    elapsed = 0.0
    for t in study.trials:
        duration = t.duration.total_seconds() if t.duration else 0.0
        elapsed += duration
        trials.append(dict(number=t.number, state=t.state.name, value=t.value, params=t.params,
                           seconds=duration, user_attrs=t.user_attrs))
        if t.state == optuna.trial.TrialState.COMPLETE:
            completed.append(t)
            if len(completed) in (45, 75, 100):
                milestones[str(len(completed))] = {"best_score": max(x.value for x in completed),
                                                    "elapsed_trial_seconds": elapsed}
    write_json(path, {"trials": trials, "completed": len(completed), "milestones": milestones})


def search(run, name, frame, folds, seed, threads, log, budget=100):
    """One sequential study. Per-trial sampler seeds make process restarts reproducible."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_dir = run / "studies"
    study_dir.mkdir(exist_ok=True)
    storage = f"sqlite:///{(study_dir / 'studies.sqlite3').resolve().as_posix()}"
    study = optuna.create_study(storage=storage, study_name=name, direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10),
                                pruner=optuna.pruners.NopPruner())
    identity = {"seed": seed, "folds": folds, "threads": threads, "budget": budget,
                "prepared": file_hash(run / "prepared.json")}
    if study.user_attrs.get("identity", identity) != identity:
        raise ValueError(f"Study identity mismatch: {name}")
    study.set_user_attr("identity", identity)
    complete_state = optuna.trial.TrialState.COMPLETE
    completed = sum(t.state == complete_state for t in study.trials)
    if completed > budget:
        raise ValueError(f"Study exceeds fixed budget: {name}")
    log.event("study resumed" if study.trials else "study starting", study=name,
              completed_trials=completed, trial_budget=budget, previously_completed=completed, newly_completed=0)
    initial_completed = completed
    while completed < budget:
        active = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        if len(active) > 1:
            raise ValueError(f"Multiple active trials in sequential study {name}")
        number = active[0].number if active else len(study.trials)
        sampler_seed = int(digest([seed, name, number])[:8], 16)
        study.sampler = optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=10)
        trial = optuna.trial.Trial(study, active[0]._trial_id) if active else study.ask()
        log.task("trial", trial=trial.number, completed_trials=completed, trial_budget=budget)
        try:
            value = inner_objective(trial, frame, folds, seed, threads, log, run / "fit-times.jsonl")
        except KeyboardInterrupt:
            # Keep RUNNING trial and its parameters for an exact retry on resume.
            study_history(study, study_dir / f"{name}.json")
            raise
        except Exception as exc:
            trial.set_user_attr("failure", repr(exc))
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            study_history(study, study_dir / f"{name}.json")
            raise
        study.tell(trial, value)
        completed += 1
        study_history(study, study_dir / f"{name}.json")
        log.event("trial complete", completed_trials=completed, trial_budget=budget,
                  trial_score=value, best_inner_score=winner(study)["score"],
                  newly_completed=completed-initial_completed)
    study_history(study, study_dir / f"{name}.json")
    return winner(study)


def prediction_rows(estimator, frame, seed, window):
    result = frame[["session_id", "participant_id", "label", "window_start_ms", "window_end_ms"]].copy()
    result["prediction"] = estimator.predict(frame[FEATURES]).astype(int)
    probabilities = estimator.predict_proba(frame[FEATURES])
    for label in range(3):
        result[f"probability_{label}"] = probabilities[:, label]
    result["seed"], result["window"] = seed, window
    return result


def read_completed_bundle(directory):
    marker = directory / "complete.json"
    if not marker.exists():
        return None
    info = read_json(marker)
    for name, expected in info["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Completed artifact changed: {directory / name}")
    return info


def evaluate(run, log):
    identity, _ = verify_prepared(run)
    splits = read_json(run / "splits.json")
    frames = {w: pd.read_parquet(run / f"features-{w}.parquet") for w in WINDOWS}
    total, completed, hits = len(splits["outer"]) * len(SEEDS), 0, 0
    paths = []
    for index, fold in enumerate(splits["outer"]):
        participant = fold.get("participant")
        fold_id = fold.get("id", participant)
        for seed in SEEDS:
            directory = run / "outer" / f"fold-{index:03d}-seed-{seed}"
            log.task("outer selection", phase="nested", outer_fold=fold_id, outer_participant=participant, seed=seed,
                     completed_outer_folds=completed, total_outer_folds=total)
            if read_completed_bundle(directory):
                hits += 1
                log.event("outer predictions cache hit", previously_completed=hits)
            else:
                winners = {}
                for w in WINDOWS:
                    log.task("tune window", window=w)
                    train = (frames[w][frames[w].session_id.isin(fold["train_pool_sessions"])]
                             if "train_pool_sessions" in fold else frames[w][frames[w].participant_id.isin(fold["train"])])
                    winners[w] = search(run, f"outer-{index:03d}-w{w}-s{seed}", train, fold["inner"],
                                        seed, identity["threads"], log)
                selected = choose_window(winners)
                # Select before seeing any held-out predictions. Reuse the selected window's refit.
                write_json(directory / "selection.json", {"fold_id": fold_id, "participant": participant, "seed": seed,
                                                           "selected_window": selected, "winners": winners})
                predictions = []
                for w in WINDOWS:
                    refit = directory / f"window-{w}"
                    if read_completed_bundle(refit):
                        log.event("outer refit predictions cache hit", window=w, output=str(refit))
                        rows = pd.read_parquet(refit / "predictions.parquet")
                    else:
                        log.task("outer refit", window=w, selected_window=selected, inner_fold=None)
                        train = (frames[w][frames[w].session_id.isin(fold["train_sessions"])]
                                 if "train_sessions" in fold else frames[w][frames[w].participant_id.isin(fold["train"])])
                        heldout = (frames[w][frames[w].session_id.isin(fold["test_sessions"])]
                                   if "test_sessions" in fold else frames[w][frames[w].participant_id == participant])
                        estimator = fit(train, winners[w]["params"], seed, identity["threads"], log, run / "fit-times.jsonl")
                        log.task("outer prediction", window=w)
                        rows = prediction_rows(estimator, heldout, seed, w)
                        rows["result"] = str(w)
                        atomic_write(refit / "predictions.parquet", lambda p: rows.to_parquet(p, index=False))
                        write_json(refit / "complete.json", {"files": {"predictions.parquet": file_hash(refit / "predictions.parquet")}})
                    predictions.append(rows)
                    if w == selected:
                        predictions.append(rows.assign(result="nested"))
                combined = pd.concat(predictions, ignore_index=True)
                atomic_write(directory / "predictions.parquet", lambda p: combined.to_parquet(p, index=False))
                write_json(directory / "complete.json", {"files": {n: file_hash(directory / n)
                           for n in ("selection.json", "predictions.parquet")}})
            paths.append(directory.relative_to(run).as_posix())
            completed += 1
            log.event("outer fold complete", completed_outer_folds=completed, total_outer_folds=total,
                      newly_completed=completed-hits, output=str(directory))
    write_json(run / "evaluated.json", {"prepared": file_hash(run / "prepared.json"), "bundles": paths})
    log.event("evaluation outputs saved", output=str(run / "evaluated.json"))


def final_window(winners):
    means = {w: float(np.mean([winners[w][s]["score"] for s in SEEDS])) for w in WINDOWS}
    return min(WINDOWS, key=lambda w: (-means[w], w)), means


def verify_artifact(path, frame, original):
    reloaded = joblib.load(path)
    np.testing.assert_array_equal(original.predict(frame[FEATURES]), reloaded.predict(frame[FEATURES]))
    np.testing.assert_allclose(original.predict_proba(frame[FEATURES]), reloaded.predict_proba(frame[FEATURES]),
                               rtol=0, atol=0)


def train_final(run, log):
    identity, _ = verify_prepared(run)
    if not (run / "reports" / "complete.json").exists():
        raise ValueError("Run nested evaluation and reporting before final selection")
    read_completed_bundle(run / "reports")
    directory = run / "final"
    if read_completed_bundle(directory):
        log.event("final artifact cache hit", previously_completed=1, newly_completed=0, output=str(directory))
        return
    split_manifest = read_json(run / "splits.json")
    splits = split_manifest["final"]
    frames = {w: pd.read_parquet(run / f"features-{w}.parquet") for w in WINDOWS}
    winners = {}
    for w in WINDOWS:
        winners[w] = {}
        for seed in SEEDS:
            log.task("full-data search", phase="final-selection", outer_participant=None, window=w, seed=seed)
            winners[w][seed] = search(run, f"final-w{w}-s{seed}", frames[w], splits, seed, identity["threads"], log)
    selected, means = final_window(winners)
    selected_params = winners[selected][42]["params"]
    log.task("final fit", window=selected, seed=42, inner_fold=None)
    training_frame = (frames[selected][frames[selected].session_id.isin(split_manifest["final_train_sessions"])]
                      if "final_train_sessions" in split_manifest else frames[selected])
    estimator = fit(training_frame, selected_params, 42, identity["threads"], log, run / "fit-times.jsonl")
    artifact = directory / "pipeline.joblib"
    atomic_write(artifact, lambda p: joblib.dump(estimator, p))
    log.task("verify artifact prediction and probability parity")
    verify_artifact(artifact, training_frame, estimator)
    metadata = dict(protocol=PROTOCOL, ordered_features=FEATURES, labels=LABELS, window_seconds=selected,
                    step_seconds=selected//2, seed=42, parameters=selected_params, dependency_versions=environment(),
                    provenance=identity, prepared_sha256=file_hash(run / "prepared.json"),
                    selection_scores=means, winners=winners, artifact_sha256=file_hash(artifact),
                    evaluation_strategy=split_manifest.get("evaluation_strategy", "participant_lopo"),
                    calibration_fraction=split_manifest.get("calibration_fraction", {mode: 1.0 for mode in LABELS}),
                    final_training_sessions=split_manifest.get("final_train_sessions", sorted(frames[selected].session_id.unique())),
                    final_training_counts=split_manifest.get("final_training_counts"),
                    independent_test_cohort=False, reload_parity="exact predictions and probabilities on all training windows")
    write_json(directory / "metadata.json", metadata)
    write_json(directory / "complete.json", {"files": {n: file_hash(directory / n)
               for n in ("pipeline.joblib", "metadata.json")}})
    log.event("final artifact saved and verified", output=str(directory), completed=1, total=1)
