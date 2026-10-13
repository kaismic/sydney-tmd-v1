# Sydney TMD V1

Local Sydney-only XGBoost training for bus (`0`), car (`1`), and train (`2`).
The pipeline implements [PLAN.md](PLAN.md): frozen common-cohort preparation,
configurable participant or session evaluation, resumable tuning,
hierarchical-bootstrap reports, and separate final model selection. No NOR data, NOR-selected parameters,
cloud configuration, or adjacent Python package is used.

The default experiment uses the participant-LOPO protocol. Each user supplies
their own run ID and input directory; results are saved under `runs/<run-id>/`.

## MLflow UI

Open the [local MLflow UI](http://127.0.0.1:5000) and select the `Sydney TMD V1`
experiment after importing a completed run.
The local server stores its SQLite database and served artifacts under `.mlflow/`
and listens on `127.0.0.1:5000`. It continues running after the starting terminal
closes; restart it after a reboot. Database/artifacts persist when stopped.

From this project directory in PowerShell:

```powershell
./scripts/mlflow.ps1 setup   # first-time setup of .venv-mlflow
./scripts/mlflow.ps1 start
./scripts/mlflow.ps1 import -RunId my-study
./scripts/mlflow.ps1 status
./scripts/mlflow.ps1 stop
```

Setup uses `requirements-mlflow.lock` in an isolated environment (MLflow 3.17.0,
validated on Windows/Python 3.14). It does not change the training interpreter or
dependencies. Use `-Port 5001` consistently for start/import if port 5000 is busy.
The script never stops another application's listener. Startup diagnostics are in
`.mlflow/server.stdout.log` and `.mlflow/server.stderr.log`; import logs are saved
as `.mlflow/import-*.log`. Keep `.mlflow/` to retain the imported results.

Replace `my-study` with the run ID used during training. Import after reporting
and final selection to include all available outputs. An imported study can
contain these MLflow runs:

| Run | What to inspect |
|---|---|
| Parent study run | Overall held-out metrics; Artifacts → `reports/` for the macro-F1 chart, CSVs, predictions, and study summary |
| `Result — 100/120/140/nested` | Comparable mean metrics and 95% intervals; Artifacts → `charts/` for count and row-normalized confusion heatmaps |
| Result seed children | Per-seed scores and bus/car/train precision, recall, F1, and support |
| `outer-...` / `final-w...` studies | Winning hyperparameters; metric histories `inner.macro_f1` and `inner.best_macro_f1` versus trial step; full trial parameters in JSON artifacts |
| `Final model — full cohort` | Full-data selection scores and downloadable `final/pipeline.joblib` with its metadata |

Expand the parent run to see children. Filter on `tags.study.kind = 'heldout_result'`
to compare the four result runs, or `tags.study.kind = 'tuning_study'` for searches.
Confusion charts keep seeds separate so repeated predictions of the same windows
are not presented as additional independent observations. Held-out metrics use
`heldout.*`; tuning uses `inner.*`; final selection uses `selection.*`. The final
pipeline is a downloadable artifact, not an independently evaluated or registered
MLflow model. Trial scores are histories within tuning studies rather than
individual MLflow runs.

Importing reads completed saved outputs and never retrains or rewrites source
artifacts. It checks manifest checksums without requiring the original training
environment, so the tracking package lives separately in `sydney_tracking/` and
does not change the training code fingerprint. Repeating an unchanged import
reuses existing runs; failed imports can resume. Changed exported content creates
a new snapshot, preserving the previous import. Run one importer at a time.
New studies need an explicit import after reporting; this setup does not stream
live training events automatically.

For a different output directory or experiment name:

```powershell
.venv-mlflow/Scripts/python -m sydney_tracking --run-dir runs/my-study --tracking-uri http://127.0.0.1:5000 --experiment "Sydney TMD V1"
.venv-mlflow/Scripts/python -m pytest -q tests/test_tracking.py --basetemp=.pytest_cache/tracking
```

Server configuration follows the [MLflow tracking-server documentation](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).

## Setup

Python 3.12+ is required. The dependency lock was validated on CPython 3.14.5 on
Windows. From this directory in PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock
.venv/Scripts/python -m pip install --no-deps -e .
.venv/Scripts/python -m pytest -q --basetemp=.pytest_cache/acceptance
```

On other platforms use `.venv/bin/python`. The package also runs directly from
this directory with `python -m sydney_tmd` without an editable installation.

## Local workflow

The tracked [`experiment.json`](experiment.json) defines one experiment. Its
default `participant_lopo` strategy and calibration fractions of `1.0` for bus,
car, and train reproduce the original study. To compare another condition,
copy the file outside the session input tree and edit it, for example:

```json
{
  "schema_version": 1,
  "evaluation_strategy": "shuffled_session_holdout",
  "calibration_fraction": {"bus": 0.8, "car": 0.5, "train": 0.5}
}
```

Fractions must be finite numbers greater than zero and at most one, with all
three modes present. Each fraction selects that share of **whole training
sessions** per mode in every inner training fold, outer refit, and final fit;
at least one session per mode is retained. Validation and test sessions are not
sampled. Session selections use split seed 42 and are shared across windows and
model seeds. The saved split manifest records requested fractions, actual counts,
and chosen IDs.

`participant_lopo` tests each participant in turn, with participant-disjoint
three-fold inner validation. `shuffled_session_holdout` reserves 20% of each
mode's whole sessions for one fixed test set, then uses session-disjoint
three-fold inner validation on the remaining pool. A participant may appear on
both sides of the shuffled split; no window from a test session enters training.
The two strategies estimate different generalization settings.

Prepare and review cohort exclusions and coverage before expensive training:

```powershell
python -m sydney_tmd prepare --input D:/tmd-data/downloaded_sessions --run-id my-study --threads 4
python -m sydney_tmd evaluate --run-id my-study
python -m sydney_tmd report --run-id my-study
python -m sydney_tmd final --run-id my-study
```

For a custom config, pass `--config path/to/experiment.json` to `prepare` or
`all`. Later commands load the copy saved under `runs/<run-id>/experiment.json`;
an explicitly supplied config must match it. Alternatively,
`python -m sydney_tmd all --input <directory> --run-id <new-id> --threads 4
--config path/to/experiment.json` executes all four stages. There is no
reduced-budget production CLI mode. The planned total is `2,709F + 2,701`
fits for `F` outer folds (21,664 for seven-person LOPO; 5,410 for one shuffled
holdout). Threads default to one on preparation, are frozen,
and are inherited thereafter. A different explicit thread value is rejected.
Measured fit durations are saved; the report estimates remaining fit time from
observed throughput, excluding other pipeline work.

`--runs-dir` and `--logs-dir` relocate outputs; keep both outside the input tree.
Downloading and synchronization are separate prerequisites.

## Input and preparation

Recursively reads non-hidden `*.json` and `*.json.gz` sessions. Payloads contain
`samples` with millisecond `ts` and vector columns `ax/ay/az`, `gx/gy/gz`, `mx/my/mz`.
Metadata can be in the payload or reference collector sidecars. For compressed
files, `name.json.gz.metadata.json` takes precedence over `name.json.metadata.json`.
Non-session files should be outside the input tree.

Require explicit `collection_country_code: "AU"`, nonempty `participant_id`, and
supported `vehicle_type`. Missing/non-AU countries, unsupported classes, absent
samples/participants, and sessions shorter than 140 seconds after trimming are
recorded as exclusions. There is no maximum duration. Conflicting participant,
session, or country identities, duplicate sessions, and malformed input stop
with a diagnostic. Optional trim bounds are inclusive, prefer payload values,
and accept zero as a valid timestamp.

All window sizes (100/120/140 seconds, stepped by 50/60/70 seconds) must have at
least one valid window for a session to enter the common cohort. Quality requires
30/4/2 Hz for accelerometer/gyroscope/magnetometer and at most a 500 ms gap,
including window boundaries. Features preserve reference Euclidean magnitudes,
population standard deviation, ordering, and float32 storage. Collector units
remain m/s², rad/s, and microtesla. Windows are half-open; the reference endpoint
includes the last sample's median positive sampling interval. Quality counts
valid rows; features use the first finite vector per timestamp.

The default outer LOPO and shuffled three-fold participant inner splits (split
seed 42) share the same cohort across windows and seeds. The alternate strategy
uses a single session holdout and three session-disjoint inner folds. Every
training partition must contain all three classes. Fewer than four participants
for LOPO, or insufficient session/class coverage for either strategy, stops
preparation. The diagnostic cohort is saved even if splitting is infeasible.

## Training and selection

Each window/outer fold/seed (42, 43, 44) has an independent Optuna TPE
study: 100 **completed** trials, 10 startup trials, and the nine-parameter search
space in the plan. Trials run sequentially on CPU, without pruning or early
stopping. The objective pools inner out-of-fold predictions before calculating
macro F1 over `[0,1,2]`. Median imputation, missing indicators, and balanced class
weights fit only the current training partition. No duration balancing is used.

Trial ties choose the earliest trial; window ties choose the shorter window.
Window selection precedes all held-out predictions. Nested predictions reuse the
selected window's refit. Final selection runs nine new studies on the eligible
cohort, chooses the window by mean best validation F1 across seeds, and fits its
fixed **seed-42** winning configuration on the configured fraction of the full
cohort. Run reporting before final selection. The final artifact includes data
previously used for evaluation and has no independent test score.

## Resume and progress

Rerun the same command with the same run ID. Inputs and sidecars, saved experiment settings,
Python/dependency versions, and implementation source hashes form an immutable
identity; changed values require a new run ID. All commands verify original
inputs, so keep them available. Prepared artifacts and completed output bundles
have SHA-256 checksums. An operating-system lock prevents concurrent runs with
the same identity and releases automatically when a process exits.

Atomic checkpoints retain ingestion/features, individual outer refits, completed
folds, reports, and final artifacts. SQLite retains Optuna trials. Failed trials
are saved and do not consume the completed-trial budget; errors stop execution.
Ctrl+C retains an active trial's parameters for retry. Unfinished fits may repeat;
completed trials and saved refits are reused. TPE receives a deterministic seed
per study/trial derived from the experiment seed, preserving uninterrupted/resumed
trial sequences. Convergence checkpoints are exported at trials 45/75/100.
Trial durations are wall-clock spans (including pauses for retried trials);
`fit-times.jsonl` records actual completed fit durations.

Each invocation prints its unique `logs/<run-id>/<attempt-id>/` directory. Invoked
stages create `ingestion.log`, `feature-extraction.log`, `evaluation.log`, and/or
`reporting.log`. Flushed UTF-8 JSON-lines events appear identically in the terminal
and file, with timezone timestamps, severity, run/stage/task context, counts,
elapsed seconds, warnings, and failure tracebacks. An independent heartbeat runs
every 20 seconds during long tasks, including fits. Earlier attempts are retained;
cache hits and new work are distinguished. Saved-log failures stop execution.

## Outputs

Under `runs/<run-id>/`:

| Path | Contents |
|---|---|
| `identity.json`, `experiment.json`, `inputs.json` | Input fingerprints, frozen settings, dependencies, code hash, ingestion decisions |
| `cohort.json`, `splits.json`, `prepared.json` | Common cohort, exclusions, participant/class coverage, frozen session and fold selections, checksums |
| `events/`, `ingested/`, `features-cache/`, `features-*.parquet` | Resumable normalized events and ordered features |
| `studies/` | Optuna SQLite database and trial/parameter/score/timing histories |
| `outer/`, `evaluated.json` | Inner-only selections and held-out predictions/probabilities |
| `fit-times.jsonl` | Fit durations and task context |
| `reports/` | Pooled/seed/class/participant metrics, confusion matrices, selection frequencies, bootstrap samples, macro-F1 charts, summary with strategy, calibration counts, and test sessions |
| `final/pipeline.joblib`, `final/metadata.json` | Complete preprocessing/model pipeline and feature/label/window/unit/quality/provenance metadata |

Reports separate fixed windows from nested selection. Primary scores pool windows
within each seed, then average seeds. The 95% percentile intervals use 5,000 draws
(seed 42): resample seeds, participants within each sampled seed, then sessions
within each sampled participant, retaining all windows of each sampled session.
Participant metrics remain separate from pooled-window metrics.
`macro-f1.png` connects fixed-window means and shades the area between their
lower and upper 95% confidence bounds, with nested selection in a separate panel.
`macro-f1.png` shows only the fixed-window curve and shaded interval.

Intervals reuse predictions; they do not rerun training/search or undo earlier
exploratory choices. Longer sessions contribute more windows. Nested evaluation
estimates the selection procedure, not independent performance of the exact
full-data artifact. Mobile conversion, integration, and device validation remain
outside this project. Load joblib artifacts only from trusted runs.

## Implementation

`data.py` adapts collector ingestion and sensor semantics from
`../all-tmd-v2/all_tmd/{ingest,features,windowing}.py`; the adjacent
`sydney-window-size-comparison` provides study/reporting context. This project
imports neither checkout and reuses neither snapshots nor selected parameters.
`config.py` validates experiment settings, `training.py` owns nested/final searches,
`reporting.py` reconstructs metrics,
`progress.py` saves events and heartbeats, and `cli.py` orchestrates stages.
