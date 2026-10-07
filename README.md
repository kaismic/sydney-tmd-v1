# Sydney TMD V1

Local Sydney-only XGBoost training for bus (`0`), car (`1`), and train (`2`).
The pipeline implements [PLAN.md](PLAN.md): frozen common-cohort preparation,
nested participant validation, resumable tuning, hierarchical-bootstrap reports,
and separate full-data model selection. No NOR data, NOR-selected parameters,
cloud configuration, or adjacent Python package is used.

**Status:** implemented with synthetic acceptance tests, including real CPU
XGBoost fitting and exact prediction/probability parity after serialization.
Production 100-trial searches and the final production artifact have not been run.
Synthetic scores are not evidence of transport-mode accuracy.

Local preparation on 7 October 2026 (`local-validation-20261007`, one model
thread) scanned 186 files and retained **149 sessions across seven participants**:
38 bus, 66 car, and 45 train sessions. Exclusions were 20 non-AU sessions, nine
unsupported classes, and eight sessions below 140 seconds. All retained sessions
had valid windows at every size; the 100/120/140-second feature tables contain
5,107/4,217/3,580 windows. All outer, inner, and final training partitions passed
class-coverage checks. This snapshot requires 21,664 planned fits. Its local
manifests, features, and logs are git-ignored; production tuning remains pending.

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

Prepare and review cohort exclusions and coverage before expensive training:

```powershell
python -m sydney_tmd prepare --input D:/tmd-data/downloaded_sessions --run-id sydney-20261007 --threads 4
python -m sydney_tmd evaluate --run-id sydney-20261007
python -m sydney_tmd report --run-id sydney-20261007
python -m sydney_tmd final --run-id sydney-20261007
```

Alternatively, `python -m sydney_tmd all --input <directory> --run-id <new-id>
--threads 4` executes all four stages. There is no reduced-budget production CLI
mode. The planned total is `2,709P + 2,701` fits for `P` eligible participants
(21,664 at seven participants). Threads default to one on preparation, are frozen,
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

Outer LOPO and shuffled three-fold inner splits (split seed 42) share the same
cohort across windows and seeds. Every training partition must contain all three
classes. Fewer than four participants or insufficient coverage stops preparation.
The diagnostic cohort is saved even if splitting is infeasible.

## Training and selection

Each window/outer participant/seed (42, 43, 44) has an independent Optuna TPE
study: 100 **completed** trials, 10 startup trials, and the nine-parameter search
space in the plan. Trials run sequentially on CPU, without pruning or early
stopping. The objective pools inner out-of-fold predictions before calculating
macro F1 over `[0,1,2]`. Median imputation, missing indicators, and balanced class
weights fit only the current training partition. No duration balancing is used.

Trial ties choose the earliest trial; window ties choose the shorter window.
Window selection precedes all held-out predictions. Nested predictions reuse the
selected window's refit. Final selection runs nine new full-cohort studies,
chooses the window by mean best validation F1 across seeds, and fits its fixed
**seed-42** winning configuration. Run reporting before final selection.

## Resume and progress

Rerun the same command with the same run ID. Inputs and sidecars, settings,
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
| `identity.json`, `inputs.json` | Input fingerprints, settings, dependencies, code hash, ingestion decisions |
| `cohort.json`, `splits.json`, `prepared.json` | Common cohort, exclusions, participant/class coverage, frozen folds, checksums |
| `events/`, `ingested/`, `features-cache/`, `features-*.parquet` | Resumable normalized events and ordered features |
| `studies/` | Optuna SQLite database and trial/parameter/score/timing histories |
| `outer/`, `evaluated.json` | Inner-only selections and held-out predictions/probabilities |
| `fit-times.jsonl` | Fit durations and task context |
| `reports/` | Pooled/seed/class/participant metrics, confusion matrices, selection frequencies, bootstrap samples, chart, summary |
| `final/pipeline.joblib`, `final/metadata.json` | Complete preprocessing/model pipeline and feature/label/window/unit/quality/provenance metadata |

Reports separate fixed windows from nested selection. Primary scores pool windows
within each seed, then average seeds. The 95% percentile intervals use 5,000 draws
(seed 42): resample seeds, participants within each sampled seed, then sessions
within each sampled participant, retaining all windows of each sampled session.
Participant metrics remain separate from pooled-window metrics.

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
`training.py` owns nested/final searches, `reporting.py` reconstructs metrics,
`progress.py` saves events and heartbeats, and `cli.py` orchestrates stages.
