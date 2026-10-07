# Sydney TMD V1

A planned Sydney-only XGBoost project for bus, car, and train classification,
optimizing macro F1 for eventual mobile transport-mode inference.

**Status: documentation only.** The experiment and final model have not been
implemented or run. See [PLAN.md](PLAN.md) for the complete protocol and future
acceptance criteria.

## Planned experiment

| Setting | Decision |
|---|---|
| Data | New frozen manifest from current local Sydney downloads; explicit `collection_country_code == "AU"` required |
| Classes | Bus (`0`), car (`1`), train (`2`) |
| Cohort | Same sessions at every window size; at least 140 seconds and one valid window per size; no maximum duration |
| Windows / steps | 100 / 50, 120 / 60, and 140 / 70 seconds |
| Features | 11 sensor-magnitude features in reference order: mean, population standard deviation, range, minimum for accelerometer and gyroscope; standard deviation, range, minimum for magnetometer |
| Quality rules | Minimum sampling rates of 30/4/2 Hz for accelerometer/gyroscope/magnetometer; maximum gap 500 ms |
| Evaluation | Outer leave-one-participant-out; three participant-disjoint inner folds shared across windows and seeds, split seed 42 |
| Search | 100 completed Optuna TPE trials per window, outer fold, and seed, including 10 startup trials |
| Seeds | 42, 43, 44; repeat both tuning and training independently |
| Selection | Pooled inner out-of-fold macro F1; select window length inside each outer fold |
| Reporting | Pooled held-out macro F1 per seed, mean across seeds; fixed-window and nested-selection results reported separately |
| Uncertainty | 5,000 hierarchical bootstrap draws over seeds, participants, and sessions; 95% percentile intervals; analysis seed 42 |
| Execution | Local CPU XGBoost, configurable model threads, sequential trials, no pruning or early stopping |

The 100-trial budget is about 2.2 times the previous 45-trial budget for an
otherwise identical search. It allows more exploration but does not guarantee
better performance. Record convergence at trials 45, 75, and 100; later budget
increases require a separately documented experiment.

Nested tuning needs **2,700 fits per eligible participant**, plus nine outer
refits per participant, 2,700 fits for final full-data selection, and one final
fit. With seven eligible participants, that would be **21,664 fits**. The new
cohort determines the actual participant count; wall-clock runtime needs local
benchmarking. Studies will persist progress for resume, and bootstrap reporting
will reuse saved predictions without retraining.

## Local data and implementation approach

Provide locally downloaded Sydney session JSON or compressed JSON files and
their country, participant, and transport-mode metadata. Data download and
synchronization are separate prerequisites. Preparation will freeze a new
manifest, record exclusions and class coverage, and retain the common
quality-valid cohort. Missing country labels are excluded. Infeasible participant
folds or incomplete training-class coverage will stop preparation for review.

Future implementation will adapt relevant code from
[`sydney-window-size-comparison`](../sydney-window-size-comparison/README.md) and
its `all-tmd-v2` source into this project. It will use neither NOR data nor
NOR-selected parameters. No AWS configuration is planned. Runnable setup and
training commands will be documented when the implementation exists.

## Planned progress monitoring

Ingestion, feature extraction, evaluation (including tuning and refits), and
reporting will display live terminal progress and automatically save the same
events. Updates will show the active task, completed/total work where known,
elapsed time, and an ETA when measurable, with progress or a heartbeat at least
every 30 seconds during active work. Evaluation will identify the participant,
window, seed, trial, and inner fold; reporting will show bootstrap progress out
of 5,000 draws and generated artifact paths.

Logs will be saved as `ingestion.log`, `feature-extraction.log`, `evaluation.log`,
and `reporting.log` under `logs/<run-id>/<attempt-id>/`, with the directory printed
at startup. Timestamped UTF-8 logs will include warnings, errors, and completion
summaries. Resumed invocations will retain previous logs and identify reused work;
no shell redirection will be required.

## Expected outputs

- Frozen input/cohort/split manifests, cached features, resumable tuning studies,
  trial histories, timings, parameters, held-out predictions, and saved stage logs.
- Metric tables, confusion matrices, per-class and participant results,
  window-selection frequencies, a macro-F1-versus-window chart with intervals,
  a separate nested-selection result, and a machine-readable study summary.
- A final preprocessing/model artifact with feature order, label mapping,
  window settings, quality rules, dependency versions, and provenance.

For the final artifact, run separate 100-trial searches for every window and seed
using three participant-grouped folds on the full frozen cohort. Select the
window by mean best-validation macro F1 across seeds, breaking ties by shorter
window, then fit its seed-42 winning configuration on all cohort data. Keep these
selection scores separate from held-out evaluation. The final artifact itself
will not have an independent test cohort in this study.

Bootstrap intervals describe variation in the saved held-out predictions; they
do not rerun training or remove effects of earlier exploratory study choices.
Mobile model conversion, app integration, and device validation remain future
work.
