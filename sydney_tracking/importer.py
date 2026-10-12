from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mlflow import MlflowClient
from mlflow.entities import Metric, Param

EXPERIMENT = "Sydney TMD V1"
LABELS = ("bus", "car", "train")
SCHEMA = "1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checked_bundle(directory):
    manifest = read_json(directory / "complete.json")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != expected:
            raise ValueError(f"Artifact checksum/path mismatch: {path}")
    return manifest


def source_snapshot(source):
    """Verify persisted outputs without loading the model or requiring its old environment."""
    prepared = read_json(source / "prepared.json")
    if digest(read_json(source / "identity.json")) != prepared["identity"]:
        raise ValueError("Prepared identity mismatch")
    for name, expected in prepared["files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or sha256(path) != expected:
            raise ValueError(f"Prepared artifact checksum/path mismatch: {path}")
    evaluated = read_json(source / "evaluated.json")
    if evaluated["prepared"] != sha256(source / "prepared.json"):
        raise ValueError("Evaluation/preparation mismatch")
    reports = checked_bundle(source / "reports")
    if reports["evaluation_sha256"] != sha256(source / "evaluated.json"):
        raise ValueError("Report/evaluation mismatch")
    names = ["identity.json", "inputs.json", "cohort.json", "splits.json", "prepared.json", "evaluated.json", "fit-times.jsonl"]
    if (source / "experiment.json").exists():
        names.append("experiment.json")
    files = [source / n for n in names]
    files.extend((source / "reports").iterdir())
    files.extend(sorted((source / "studies").glob("*.json")))
    for bundle in evaluated["bundles"]:
        directory = (source / bundle).resolve()
        if not directory.is_relative_to(source):
            raise ValueError("Outer bundle outside source run")
        checked_bundle(directory)
        files.append(directory / "selection.json")
    if (source / "final").exists():
        checked_bundle(source / "final")
        files.extend((source / "final").iterdir())
    fingerprints = {p.relative_to(source).as_posix(): sha256(p) for p in sorted(files) if p.is_file()}
    return fingerprints, digest({"schema": SCHEMA, "files": fingerprints})


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def confusion_matrix(rows, result, seed):
    matrix = np.zeros((3, 3), dtype=int)
    selected = [r for r in rows if r["result"] == result and int(r["seed"]) == seed]
    if len(selected) != 9 or len({(r["actual"], r["predicted"]) for r in selected}) != 9:
        raise ValueError(f"Expected nine confusion-matrix cells for {result}, seed {seed}")
    for row in selected:
        matrix[int(row["actual"]), int(row["predicted"])] = int(row["windows"])
    return matrix


def draw_confusion(matrices, title, path, normalize=False):
    """Keep seeds separate: the same evaluation windows recur across seeds."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(4 * len(matrices), 4), squeeze=False)
    for ax, (seed, counts) in zip(axes.flat, sorted(matrices.items())):
        totals = counts.sum(axis=1, keepdims=True)
        values = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0) if normalize else counts
        ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalize else max(1, int(counts.max())))
        for a in range(3):
            for p in range(3):
                label = f"{values[a, p]:.1%}" if normalize else str(counts[a, p])
                threshold = .5 if normalize else counts.max() / 2
                ax.text(p, a, label, ha="center", va="center", color="white" if values[a, p] > threshold else "black")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
               xlabel="Predicted", ylabel="Actual", title=f"Seed {seed}")
    fig.suptitle(title + (" — row-normalized" if normalize else " — window counts"))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


class Importer:
    def __init__(self, source, uri, experiment=EXPERIMENT):
        self.source = Path(source).resolve()
        self.files, self.snapshot = source_snapshot(self.source)
        self.client = MlflowClient(tracking_uri=uri)
        exp = self.client.get_experiment_by_name(experiment)
        self.experiment_id = exp.experiment_id if exp else self.client.create_experiment(experiment)
        self.existing = {}
        token = None
        while True:
            page = self.client.search_runs([self.experiment_id],
                filter_string=f"tags.`import.snapshot` = '{self.snapshot}'", max_results=1000, page_token=token)
            for run in page:
                key = run.data.tags.get("import.entity")
                if key in self.existing:
                    raise ValueError(f"Duplicate imported entity: {key}")
                self.existing[key] = run
            token = page.token
            if not token:
                break
        self.created = self.skipped = 0
        self.timestamp = int((self.source / "reports" / "complete.json").stat().st_mtime * 1000)

    def entity(self, key, name, kind, parent=None, params=None, metrics=None, history=None,
               artifacts=(), generated=()):
        existing = self.existing.get(key)
        if existing and existing.data.tags.get("import.complete") == "true" and existing.info.status == "FINISHED":
            self.skipped += 1
            print(f"Reused {name}", flush=True)
            return existing.info.run_id
        tags = {"import.snapshot": self.snapshot, "import.entity": key, "import.schema": SCHEMA,
                "study.run_id": self.source.name, "study.kind": kind,
                "mlflow.note.content": "Imported saved Sydney study results. No retraining. "
                "Held-out metrics and final-training selection scores are separate.", "import.complete": "false"}
        if parent:
            tags["mlflow.parentRunId"] = parent
        run_id = existing.info.run_id if existing else self.client.create_run(
            self.experiment_id, tags=tags, run_name=name).info.run_id
        self.client.update_run(run_id, status="RUNNING")
        try:
            values = [Metric(k, float(v), self.timestamp, 0) for k, v in (metrics or {}).items()]
            values.extend(Metric(k, float(v), self.timestamp, step) for k, v, step in (history or []))
            parameters = [Param(k, str(v)) for k, v in (params or {}).items()]
            for offset in range(0, max(len(values), 1), 500):
                self.client.log_batch(run_id, metrics=values[offset:offset+500], params=parameters if offset == 0 else [])
            for relative in artifacts:
                path = self.source / relative
                parent_path = Path(relative).parent.as_posix()
                self.client.log_artifact(run_id, str(path), artifact_path=None if parent_path == "." else parent_path)
            for path in generated:
                self.client.log_artifact(run_id, str(path), artifact_path="charts")
            self.client.set_tag(run_id, "import.complete", "true")
            self.client.set_terminated(run_id, status="FINISHED")
            self.created += 1
            print(f"Imported {name}", flush=True)
        except BaseException:
            self.client.set_terminated(run_id, status="FAILED")
            raise
        return run_id

    def run(self):
        summary = read_json(self.source / "reports" / "study-summary.json")
        cohort = read_json(self.source / "cohort.json")
        splits = read_json(self.source / "splits.json")
        strategy = splits.get("evaluation_strategy", "participant_lopo")
        root_metrics = {f"heldout.{result}.{metric}": value
                        for result, record in summary["results"].items() for metric, value in record["mean"].items()}
        root = self.entity("summary", self.source.name, "summary", params={"participants": cohort["participant_count"],
                           "sessions": len(cohort["sessions"]), "source_identity": read_json(self.source / "prepared.json")["identity"],
                           "evaluation_strategy": strategy,
                           "calibration_fraction": json.dumps(splits.get("calibration_fraction", {mode: 1.0 for mode in LABELS}), sort_keys=True)},
                           metrics=root_metrics,
                           artifacts=[n for n in self.files if n.startswith("reports/") or "/" not in n])
        confusion = csv_rows(self.source / "reports" / "confusion-matrices.csv")
        per_class = csv_rows(self.source / "reports" / "per-class.csv")
        with tempfile.TemporaryDirectory(prefix="sydney-mlflow-") as temporary:
            for result, record in summary["results"].items():
                matrices = {int(seed): confusion_matrix(confusion, result, int(seed)) for seed in record["seed_scores"]}
                charts = []
                for normalized in (False, True):
                    path = Path(temporary) / f"confusion-{result}-{'normalized' if normalized else 'counts'}.png"
                    draw_confusion(matrices, "Nested selection" if result == "nested" else f"{result}-second windows", path, normalized)
                    charts.append(path)
                metrics = {f"heldout.{m}": v for m, v in record["mean"].items()}
                metrics.update({f"heldout.{m}.ci95_{bound}": v for m, bounds in record["intervals"].items() for bound, v in bounds.items()})
                result_id = self.entity(f"result/{result}", f"Result — {result}", "heldout_result", root,
                    params={"window": result, "seeds": "42,43,44", "bootstrap_draws": summary["bootstrap_draws"]},
                    metrics=metrics, generated=charts)
                for seed, scores in record["seed_scores"].items():
                    metrics = {f"heldout.{k}": v for k, v in scores.items()}
                    for row in per_class:
                        if row["result"] == result and row["seed"] == seed:
                            metrics.update({f"heldout.{LABELS[int(row['label'])]}.{m}": float(row[m])
                                            for m in ("precision", "recall", "f1", "support")})
                    self.entity(f"result/{result}/seed/{seed}", f"{result} — seed {seed}", "heldout_seed", result_id,
                                params={"window": result, "seed": seed}, metrics=metrics)
        for name in sorted(n for n in self.files if n.startswith("studies/")):
            study = read_json(self.source / name)
            trials = [t for t in study["trials"] if t["state"] == "COMPLETE"]
            if len(trials) != study["completed"] or not trials:
                raise ValueError(f"Invalid completed trial history: {name}")
            winner = min(trials, key=lambda t: (-t["value"], t["number"]))
            match = re.fullmatch(r"(final|outer-(\d+))-w(\d+)-s(\d+)", Path(name).stem)
            if not match:
                raise ValueError(f"Unknown study name: {name}")
            phase, index, window, seed = match.groups()
            params = {**winner["params"], "window_seconds": window, "seed": seed, "winning_trial": winner["number"],
                      "phase": "final_selection" if phase == "final" else "nested_selection"}
            if index is not None:
                fold = splits["outer"][int(index)]
                params["outer_fold"] = fold.get("id", fold.get("participant", f"fold-{index}"))
                if fold.get("participant") is not None:
                    params["outer_participant"] = fold["participant"]
            history, best = [], -float("inf")
            for trial in trials:
                best = max(best, trial["value"])
                history.extend([("inner.macro_f1", trial["value"], trial["number"]),
                                ("inner.best_macro_f1", best, trial["number"]),
                                ("trial.seconds", trial["seconds"], trial["number"])])
            self.entity(name, Path(name).stem, "tuning_study", root, params=params,
                        metrics={"inner.winning_macro_f1": winner["value"], "trials.completed": len(trials),
                                 "trials.failed": sum(t["state"] == "FAIL" for t in study["trials"])},
                        history=history, artifacts=[name])
        if (self.source / "final" / "metadata.json").exists():
            metadata = read_json(self.source / "final" / "metadata.json")
            fractions = metadata.get("calibration_fraction", {mode: 1.0 for mode in LABELS})
            full_cohort = all(value == 1.0 for value in fractions.values())
            self.entity("final", "Final model — full cohort" if full_cohort else "Final model — configured cohort", "final_model", root,
                        params={**metadata["parameters"], "seed": metadata["seed"], "window_seconds": metadata["window_seconds"],
                                "independent_test_cohort": False, "evaluation_strategy": metadata.get("evaluation_strategy", strategy),
                                "calibration_fraction": json.dumps(fractions, sort_keys=True)},
                        metrics={f"selection.window_{w}.macro_f1": v for w, v in metadata["selection_scores"].items()},
                        artifacts=[n for n in self.files if n.startswith("final/")])
        # Detect any source change during an import rather than accepting a mixed snapshot.
        if any(sha256(self.source / n) != value for n, value in self.files.items()):
            raise ValueError("Source changed during import; rerun after the study finishes")
        return {"experiment_id": self.experiment_id, "run_id": root, "snapshot": self.snapshot,
                "imported": self.created, "reused": self.skipped}
