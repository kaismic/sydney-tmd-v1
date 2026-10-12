from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .core import LABELS, SEEDS, WINDOWS, atomic_write, file_hash, read_json, verify_prepared, write_json

LIMITATION = ("Intervals resample saved held-out predictions, not the full training/search process; "
              "they do not remove effects of prior exploratory choices, including candidate windows. "
              "Primary scores pool windows within each seed, then average seeds. Longer sessions "
              "contribute more windows; macro F1 balances classes, not sessions or participants. "
              "Nested results evaluate the selection procedure, not the exact final fitted artifact.")


def matrix_metrics(cm):
    cm = np.asarray(cm, dtype=float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    true = np.diag(cm)
    precision = np.divide(true, predicted, out=np.zeros(3), where=predicted > 0)
    recall = np.divide(true, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * true, support + predicted, out=np.zeros(3), where=(support + predicted) > 0)
    return dict(macro_f1=float(f1.mean()), balanced_accuracy=float(recall[support > 0].mean()),
                per_class=[dict(label=i, precision=float(precision[i]), recall=float(recall[i]),
                                f1=float(f1[i]), support=int(support[i])) for i in range(3)])


def pooled_metrics(rows):
    return matrix_metrics(confusion_matrix(rows.label, rows.prediction, labels=[0, 1, 2]))


def bootstrap(rows, draws=5000, seed=42, log=None):
    """Resample seeds -> participants -> sessions; each cluster retains all its windows."""
    clusters = {}
    for (s, p, session), group in rows.groupby(["seed", "participant_id", "session_id"], sort=True):
        clusters.setdefault(int(s), {}).setdefault(p, []).append(
            confusion_matrix(group.label, group.prediction, labels=[0, 1, 2]))
    seeds = sorted(clusters)
    rng, results = np.random.default_rng(seed), []
    for draw in range(draws):
        seed_scores = []
        for selected_seed in rng.choice(seeds, len(seeds), replace=True):
            people = sorted(clusters[selected_seed])
            pooled = np.zeros((3, 3), dtype=np.int64)
            for participant in rng.choice(people, len(people), replace=True):
                sessions = np.asarray(clusters[selected_seed][participant])
                pooled += sessions[rng.integers(0, len(sessions), len(sessions))].sum(axis=0)
            metrics = matrix_metrics(pooled)
            seed_scores.append([metrics["macro_f1"], metrics["balanced_accuracy"]])
        results.append(np.mean(seed_scores, axis=0))
        if log and ((draw + 1) % 100 == 0 or draw + 1 == draws):
            log.event("bootstrap progress", completed=draw+1, total=draws)
    results = np.asarray(results)
    intervals = np.percentile(results, [2.5, 97.5], axis=0)
    return {metric: {"low": float(intervals[0, i]), "high": float(intervals[1, i])}
            for i, metric in enumerate(("macro_f1", "balanced_accuracy"))}, results


def plot_macro_f1(summary, path, include_nested=True):
    if include_nested:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 1]}, sharey=True)
        fixed_ax, nested_ax = axes
    else:
        fig, fixed_ax = plt.subplots(figsize=(8, 4))
    xvalues = list(WINDOWS)
    fixed = [summary[str(window)] for window in WINDOWS]
    means = [item["mean"]["macro_f1"] for item in fixed]
    lower = [item["intervals"]["macro_f1"]["low"] for item in fixed]
    upper = [item["intervals"]["macro_f1"]["high"] for item in fixed]
    fixed_ax.fill_between(xvalues, lower, upper, color="#2266aa", alpha=.2)
    fixed_ax.plot(xvalues, lower, color="#2266aa", alpha=.5, linewidth=1)
    fixed_ax.plot(xvalues, upper, color="#2266aa", alpha=.5, linewidth=1)
    fixed_ax.plot(xvalues, means, "-o", color="#2266aa")
    fixed_ax.set_xticks(xvalues, [str(window) for window in WINDOWS])
    fixed_ax.set(xlabel="Window (seconds)", ylabel="Pooled macro F1, mean over seeds", title="Fixed windows")
    fixed_ax.set_ylim(0, 1)
    fixed_ax.grid(alpha=.2)
    if include_nested:
        nested = summary["nested"]
        bounds = nested["intervals"]["macro_f1"]
        nested_ax.vlines(0, bounds["low"], bounds["high"], color="#2266aa", linewidth=2)
        nested_ax.plot(0, nested["mean"]["macro_f1"], "o", color="#2266aa")
        nested_ax.set_xticks([0], ["nested"])
        nested_ax.set_title("Nested selection")
        nested_ax.set_ylim(0, 1)
        nested_ax.grid(alpha=.2)
    fig.suptitle("95% hierarchical bootstrap intervals (saved predictions)")
    fig.tight_layout()
    atomic_write(path, lambda p: fig.savefig(p, format="png", dpi=180))
    plt.close(fig)


def report(run, log):
    from .training import read_completed_bundle

    _, prepared = verify_prepared(run)
    evaluation = read_json(run / "evaluated.json")
    if evaluation["prepared"] != file_hash(run / "prepared.json"):
        raise ValueError("Evaluation does not match prepared data")
    splits = read_json(run / "splits.json")
    expected = [f"outer/fold-{i:03d}-seed-{s}" for i, _ in enumerate(splits["outer"])
                for s in SEEDS]
    if evaluation["bundles"] != expected:
        raise ValueError("Incomplete or unexpected evaluation bundles")
    directory = run / "reports"
    directory.mkdir(exist_ok=True)
    frames, selections = [], []
    for i, bundle in enumerate(expected, 1):
        log.task("load held-out predictions", bundle=bundle)
        if read_completed_bundle(run / bundle) is None:
            raise ValueError(f"Incomplete outer bundle: {bundle}")
        frames.append(pd.read_parquet(run / bundle / "predictions.parquet"))
        selections.append(read_json(run / bundle / "selection.json"))
        log.event("predictions loaded", completed=i, total=len(expected))
    previous = read_completed_bundle(directory)
    if previous:
        if previous["evaluation_sha256"] != file_hash(run / "evaluated.json"):
            raise ValueError("Reporting cache does not match evaluation")
        log.event("report cache hit", previously_completed=4, newly_completed=0, output=str(directory))
        return
    rows = pd.concat(frames, ignore_index=True)
    atomic_write(directory / "held-out-predictions.parquet", lambda p: rows.to_parquet(p, index=False))
    cohort = read_json(run / "cohort.json")["sessions"]
    expected_sessions = (set().union(*(set(fold["test_sessions"]) for fold in splits["outer"]))
                         if all("test_sessions" in fold for fold in splits["outer"])
                         else {record["session_id"] for record in cohort})
    if all("test_sessions" in fold for fold in splits["outer"]):
        total_test_sessions = sum(len(fold["test_sessions"]) for fold in splits["outer"])
        if len(expected_sessions) != total_test_sessions:
            raise ValueError("A test session appears in multiple outer folds")
    summary, seed_metrics, per_class, participants, matrices = {}, [], [], [], []
    output_names = ["held-out-predictions.parquet"]
    for result in [str(w) for w in WINDOWS] + ["nested"]:
        log.task("pooled metrics", result=result)
        group = rows[rows.result == result]
        by_seed = {}
        for seed in SEEDS:
            data = group[group.seed == seed]
            if set(data.session_id) != expected_sessions:
                raise ValueError(f"Missing or unexpected held-out sessions: {result}, {seed}")
            if data.duplicated(["session_id", "window_start_ms"]).any():
                raise ValueError("Duplicate held-out windows")
            metrics = pooled_metrics(data)
            by_seed[str(seed)] = {k: metrics[k] for k in ("macro_f1", "balanced_accuracy")}
            seed_metrics.append(dict(result=result, seed=seed, **by_seed[str(seed)]))
            per_class.extend(dict(result=result, seed=seed, **record) for record in metrics["per_class"])
            cm = confusion_matrix(data.label, data.prediction, labels=[0, 1, 2])
            matrices.extend(dict(result=result, seed=seed, actual=a, predicted=p, windows=int(cm[a, p]))
                            for a in range(3) for p in range(3))
            for participant, subset in data.groupby("participant_id"):
                metric = pooled_metrics(subset)
                participants.append(dict(result=result, seed=seed, participant=participant, windows=len(subset),
                                         macro_f1=metric["macro_f1"], balanced_accuracy=metric["balanced_accuracy"]))
        log.task("hierarchical bootstrap", result=result)
        intervals, samples = bootstrap(group, log=log)
        name = f"bootstrap-{result}.csv"
        atomic_write(directory / name, lambda p: pd.DataFrame(samples, columns=["macro_f1", "balanced_accuracy"]).to_csv(p, index=False))
        output_names.append(name)
        summary[result] = {"seed_scores": by_seed,
                           "mean": {m: float(np.mean([v[m] for v in by_seed.values()]))
                                    for m in ("macro_f1", "balanced_accuracy")}, "intervals": intervals}
        log.event("bootstrap table saved", output=str(directory / name))
    frequencies = pd.Series([s["selected_window"] for s in selections]).value_counts()
    tables = {"seed-metrics.csv": seed_metrics, "per-class.csv": per_class,
              "participant-metrics.csv": participants, "confusion-matrices.csv": matrices,
              "window-selection.csv": [{"window": w, "count": int(frequencies.get(w, 0))} for w in WINDOWS],
              "metrics.csv": [{"result": r, **v["mean"],
                                "macro_f1_low": v["intervals"]["macro_f1"]["low"],
                                "macro_f1_high": v["intervals"]["macro_f1"]["high"]} for r, v in summary.items()]}
    for name, records in tables.items():
        log.task("write metric table", output=str(directory / name))
        atomic_write(directory / name, lambda p: pd.DataFrame(records).to_csv(p, index=False))
        output_names.append(name)
        log.event("table saved", output=str(directory / name))
    log.task("macro F1 chart")
    plot_macro_f1(summary, directory / "macro-f1-with-nested.png")
    output_names.append("macro-f1-with-nested.png")
    plot_macro_f1(summary, directory / "macro-f1.png", include_nested=False)
    output_names.append("macro-f1.png")
    timings = [json.loads(line)["fit_seconds"] for line in (run / "fit-times.jsonl").read_text().splitlines()]
    study_summary = dict(results=summary, labels=LABELS, bootstrap_draws=5000, analysis_seed=42,
                         limitations=LIMITATION, window_selection_frequency={str(w): int(frequencies.get(w, 0)) for w in WINDOWS},
                         evaluation_strategy=splits.get("evaluation_strategy", "participant_lopo"),
                         calibration_fraction=splits.get("calibration_fraction", {mode: 1.0 for mode in LABELS}),
                         test_sessions=sorted(expected_sessions),
                         outer_folds=[{"id": fold.get("id", fold.get("participant")),
                                       "training_pool_counts": fold.get("training_pool_counts"),
                                       "training_counts": fold.get("training_counts"),
                                       "test_counts": fold.get("test_counts"),
                                       "test_sessions": fold.get("test_sessions")} for fold in splits["outer"]],
                         final_training_counts=splits.get("final_training_counts"),
                         planned_fits=prepared["planned_fits"], measured_fits=len(timings),
                         measured_fit_seconds=sum(timings), mean_fit_seconds=float(np.mean(timings)),
                         median_fit_seconds=float(np.median(timings)),
                         estimated_remaining_fit_seconds=max(0, prepared["planned_fits"]-len(timings))*float(np.mean(timings)),
                         runtime_estimate_note="Measured fit throughput only; excludes feature/report overhead and assumes similar future fits.")
    write_json(directory / "study-summary.json", study_summary)
    output_names.append("study-summary.json")
    atomic_write(directory / "README.md", lambda p: p.write_text(
        "# Sydney TMD evaluation\n\n" + LIMITATION + "\n\nSee `metrics.csv`, `seed-metrics.csv`, "
        "`participant-metrics.csv`, `per-class.csv`, `confusion-matrices.csv`, `window-selection.csv`, "
        "`macro-f1-with-nested.png`, `macro-f1.png`, and `study-summary.json`. "
        "The split strategy, requested fractions, actual session counts, and test cohort are in `study-summary.json`. "
        "The final artifact is selected and fitted separately in `../final/` after evaluation; it is not independently tested.\n",
        encoding="utf-8"))
    output_names.append("README.md")
    write_json(directory / "complete.json", {"evaluation_sha256": file_hash(run / "evaluated.json"),
                                             "files": {n: file_hash(directory / n) for n in output_names}})
    log.event("report outputs saved", output=str(directory), completed=4, total=4)
