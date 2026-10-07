# Sydney TMD V1 implementation plan

## Purpose

Develop a Sydney-only XGBoost model for bus, car, and train classification,
optimizing macro F1 for eventual use in a mobile transport-mode inference app.
Evaluate generalization to unseen participants and produce a final model trained
on all eligible data.

This document specifies planned work. The project currently contains documentation
only; no pipeline, experiment results, or trained artifact is available. Future
implementation will adapt relevant ingestion, feature extraction, evaluation,
and reporting code from `../sydney-window-size-comparison` and its
`../all-tmd-v2` source into this project. Do not load NOR-TMD data or reuse
NOR-selected parameters. Run locally without AWS configuration. Mobile
conversion, app integration, and device validation are future work.

## Data

Freeze a new input manifest from the locally downloaded Sydney sessions available
when preparation runs. The input directory contains session JSON or compressed
JSON files; downloading or synchronizing data is a separate prerequisite.

- Require an explicit `collection_country_code == "AU"`. Exclude missing country
  labels and non-AU sessions, including legacy records that the reference study
  admitted without a country label.
- Keep only bus, car, and train, with the label mapping `bus: 0`, `car: 1`, and
  `train: 2`.
- Require at least 140 seconds of usable session duration, with no maximum.
- Use the same session cohort for all window sizes: retain a session only if it
  produces at least one quality-valid window at each of 100, 120, and 140 seconds.
  Window counts may differ across sizes even though session membership is fixed.
- Record input fingerprints, session and participant identities, exclusion
  reasons, class counts, and participant/class coverage. Derive participant counts
  from this cohort; do not inherit the previous snapshot or its seven-participant
  assumption.

Freeze the cohort and split manifests before tuning. Changed data or experiment
settings require a new run identity rather than resuming an incompatible run.

## Features

Use complete windows within each session, with 50% overlap:

| Window length | Step |
|---|---|
| 100 seconds | 50 seconds |
| 120 seconds | 60 seconds |
| 140 seconds | 70 seconds |

Preserve the reference pipeline's sensor-magnitude calculations and quality
checks. Minimum sampling rates are 30 Hz for accelerometer, 4 Hz for gyroscope,
and 2 Hz for magnetometer; the maximum permitted sample gap is 500 ms.

Compute the following 11 features in the reference ordering. Standard deviation
is the population standard deviation (`ddof=0`), and range is maximum minus
minimum.

| Sensor | Ordered features |
|---|---|
| Accelerometer magnitude | mean, standard deviation, range, minimum |
| Gyroscope magnitude | mean, standard deviation, range, minimum |
| Magnetometer magnitude | standard deviation, range, minimum |

Preserve sensor units and feature semantics in saved metadata so later app
integration can reproduce training-time features.

## Training and selection

### Nested participant validation

Use outer leave-one-participant-out (LOPO) evaluation. In each outer fold, all
sessions and windows belonging to the held-out participant are reserved for
evaluation and excluded from all tuning and preprocessing fits.

Within the remaining participants, create three participant-disjoint inner folds
using shuffled participant-grouped splitting with split seed 42. Freeze those
folds and share them across window sizes and training seeds. Validate participant
separation and coverage of all three classes in every inner and outer training
partition before tuning. If the cohort cannot support these folds, stop with a
diagnostic; do not silently change the split count or grouping policy.

Repeat the entire hyperparameter search independently for seeds 42, 43, and 44.
Each seed controls its Optuna sampler and XGBoost training; the split seed remains
42. These repeats capture both search and model-training variability.

### Trial budget and search space

For every window, outer fold, and seed, run **100 completed Optuna TPE trials**,
including 10 startup trials. This increases the previous 45-trial budget by about
2.2 times for an otherwise identical study. It is a practical starting budget,
not a guarantee of better macro F1. Record best-so-far inner-validation scores
and elapsed times at completed trials 45, 75, and 100. Any later budget increase
is a separately documented experiment, not an extension chosen from outer test
scores.

Use sequential trials, CPU XGBoost, and configurable model threads. Disable
pruning and early stopping. Preserve the existing nine-parameter search space:

| Parameter | XGBoost name | Search range |
|---|---|---|
| Trees | `n_estimators` | Integers 200–1,200, step 100 |
| Learning rate | `learning_rate` | 0.02–0.2, logarithmic |
| Maximum depth | `max_depth` | Integers 3–8 |
| Minimum child weight | `min_child_weight` | 1–10 |
| Row subsampling | `subsample` | 0.6–1.0 |
| Column subsampling | `colsample_bytree` | 0.5–1.0 |
| Gamma | `gamma` | 0–5 |
| L1 regularization | `reg_alpha` | 0–2 |
| L2 regularization | `reg_lambda` | 0.5–10 |

Float ranges are linear unless explicitly marked logarithmic. Preserve the
reference XGBoost histogram-tree multiclass classifier and median-imputation
pipeline, including missing-value indicators. Fit imputation and compute
class-balanced sample weights using only the current training partition. Apply
no duration balancing.

Each trial maximizes macro F1 calculated from pooled inner out-of-fold
predictions, using the fixed labels `[0, 1, 2]` and zero for undefined class
scores. Do not average the three fold-level F1 values as the tuning objective.

### Window selection and outer refits

For each outer fold and seed:

1. Select the highest-scoring trial separately for each window. Break equal
   scores by the earliest trial.
2. Select the window with the highest winning inner-validation score, breaking
   equal scores by the shorter window. Do not inspect outer evaluation scores
   when choosing trials or windows.
3. Refit each window's winner on all outer training participants and predict the
   held-out participant's windows.
4. Reuse the selected window's held-out predictions to evaluate the complete
   window-selection procedure. This requires no additional model fit.

Persist studies, input/cohort/split manifests, trial histories, timings, selected
parameters, and held-out predictions. Resume completed work without duplicating
trials or mixing experiment identities. Failed trials do not count toward the
100 completed trials; surface failures and preserve diagnostic state for recovery.

### Local compute budget

For `P` eligible participants, nested tuning requires
`3 windows × 3 seeds × 100 trials × 3 inner folds × P = 2,700P` fits.
Outer refits add `9P` fits. The final full-data search described below adds 2,700
fits, followed by one final fit. The planned total is therefore `2,709P + 2,701`,
excluding failures or reruns: **21,664 fits if P = 7**.

Record measured fit durations before estimating wall-clock runtime. Cache prepared
features, persist tuning progress, and avoid promising a runtime before local
benchmarking. Bootstrap reporting reuses predictions and adds no model fits.

## Evaluation

Report the three fixed-window experiments and the nested window-selection
procedure separately. For each result, pool outer held-out predictions across
participants within each seed, compute macro F1 over the fixed three labels,
then report the mean across seeds alongside the individual seed scores.

Include balanced accuracy, per-class scores, confusion matrices, participant
results, and window-selection frequencies. Keep participant-level summaries
separate from the primary pooled-window metric: long sessions contribute more
windows, and macro F1 balances classes rather than sessions or participants.

Generate 95% percentile confidence intervals from **5,000 hierarchical bootstrap
draws**, with analysis seed 42. Resample seeds, participants, and sessions within
sampled participants; retain all windows from a sampled session together. In each
draw, reconstruct pooled scores for sampled seeds and average those seed scores.
Use saved held-out predictions without retraining, for both fixed-window and
nested-selection results.

Produce a macro-F1-versus-window chart with intervals and a separate result for
nested window selection. Save supporting metric tables, predictions, trial
histories, and a machine-readable study summary. Clearly state that these
intervals do not reproduce the full training/search process or remove effects of
earlier exploratory study choices, including the choice of candidate windows.

## Final artifact

After evaluation, run a separate full-data selection stage on the same frozen
cohort. Create three participant-grouped folds with split seed 42, shared across
windows and seeds, and apply the same coverage checks and training-only
preprocessing rules. Repeat the 100-trial search independently for each window
and seed: nine studies in total.

Choose the window with the highest mean best-validation macro F1 across seeds;
break ties by the shorter window. Fit that window's seed-42 winning configuration
on all cohort data. Seed 42 is fixed in advance for the final artifact, rather
than selecting whichever seed happens to score highest.

Save the complete preprocessing/model pipeline with the ordered features, label
mapping, window and step settings, sensor units and quality rules, dependency
versions, selected parameters, and data/code provenance. Verify that reloading
the artifact reproduces its predictions and probabilities.

Keep final-training selection scores separate from outer held-out evaluation.
The full-data model has no remaining independent test cohort in this study;
nested results evaluate the selection procedure, not an independent test of that
exact fitted artifact. Mobile export, integration, and device validation remain
outside this implementation.

## Implementation workflow and acceptance criteria

### Terminal progress and saved logs

All ingestion, feature extraction, evaluation (including tuning and refits), and
reporting steps must emit live, human-readable progress to the terminal and save
the same events automatically. Logging must be enabled by default, without
requiring shell redirection or a separate monitoring tool.

- Log stage start, completion, failure, and interruption, plus progress updates
  at least every 30 seconds while work is active, including during a long model
  fit. Include an explicit heartbeat when no work unit has finished recently.
- Include a timestamp with timezone, severity, run ID, stage, current task,
  completed/total work units when known, and elapsed time. Show percentage and
  estimated time remaining only when supported by known totals and measured
  throughput; otherwise show counts and elapsed time without inventing an ETA.
- Flush each event promptly to both terminal and file. Use newline-delimited
  text in UTF-8 log files, without terminal control codes or progress-bar redraws,
  so redirected output and saved logs remain readable.
- Save separate `ingestion.log`, `feature-extraction.log`, `evaluation.log`, and
  `reporting.log` files under `logs/<run-id>/<attempt-id>/`. Assign a unique attempt
  ID to every invocation, print the log directory at startup, and retain previous
  attempts on resume. Record cache hits and previously completed work separately
  from newly processed work. Logs supplement the authoritative resume manifests
  and studies; they do not replace them.
- Save warnings, exclusion summaries, errors, and exception tracebacks with task
  context. On handled failure or interruption, flush logs and state what completed
  and where resumable state remains. A successful stage ends with counts, elapsed
  time, and output paths. If a log cannot be created or written, surface the error
  in the terminal and stop rather than silently running without saved logs.

| Step | Required progress details |
|---|---|
| Ingestion | Files/sessions scanned out of the discovered total, current session, accepted/excluded/failed counts, exclusion reasons, and manifest output path |
| Feature extraction | Window length, sessions processed/total, current session, windows generated/retained/rejected, cache hits, and feature/cohort output paths |
| Evaluation | Nested or final-selection phase, outer participant, window, seed, trial progress out of 100, inner-fold progress out of three, active fit/refit/prediction task, trial score and best-so-far inner score, completed outer folds, and saved prediction/metric paths |
| Reporting | Result sets loaded/total, active metric or chart, bootstrap draws completed out of 5,000 for each result, and each table/chart/summary output path |

### Implementation sequence

The future implementation sequence is:

1. Adapt relevant reference code into this project and establish reproducible
   local dependencies and shared terminal/file logging, without NOR data,
   NOR-selected parameters, or AWS setup.
2. Prepare and freeze inputs, the common cohort, features, and participant splits;
   review exclusions and class coverage before expensive training.
3. Run resumable nested tuning and outer evaluation, recording convergence and
   local compute measurements.
4. Reconstruct evaluation metrics and bootstrap intervals from saved predictions
   and generate the report artifacts.
5. Run full-data selection, fit and verify the final artifact, and document the
   implemented local workflow and measured results.

### Acceptance criteria

Acceptance tests must cover:

- Strict AU filtering, including missing and non-AU country labels; allowed
  classes; the 140-second minimum; and absence of a maximum-duration filter.
- Common-cohort membership at all window sizes, correct overlap, reference
  feature order/calculations, and sensor-quality exclusions.
- Participant isolation in every inner/outer split, shared split manifests,
  complete training-class coverage, and diagnostic failure for infeasible folds.
- Training-only imputation and class weights, independent seeded searches,
  100 completed trials per study, and inner-only window selection with fixed
  tie-breaking rules.
- Reproducible resume, preservation of completed work, and rejection of changed
  inputs or settings under an existing run identity.
- Reconstruction of pooled metrics and deterministic hierarchical bootstrap
  intervals from saved predictions, including the nested-selection result.
- Final window/seed selection and prediction/probability parity after artifact
  serialization and reload.
- Terminal and saved-log coverage for all four steps, timely progress/heartbeats
  during long tasks, accurate counters and task context, readable flushed files,
  completion/output summaries, failure/interruption diagnostics, log-write
  failures, and preservation of earlier logs and clear skip counts on resume.

The current documentation revision changes only this plan and `README.md`.
It introduces no executable APIs, schemas, types, training runs, or model artifacts.
