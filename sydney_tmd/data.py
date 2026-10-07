"""Collector-only ingestion and reference-compatible sensor features."""
from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .core import (FEATURES, LABELS, PROTOCOL, RATES, SENSORS, WINDOWS, atomic_write,
                   code_fingerprint, digest, environment, file_hash, read_json, write_json)


def discover(root):
    return sorted(p for p in root.rglob("*") if p.is_file()
                  and (p.name.endswith(".json") or p.name.endswith(".json.gz"))
                  and not p.name.endswith(".metadata.json")
                  and not any(part.startswith(".") for part in p.relative_to(root).parts))


def sidecars(path):
    return [p for p in (path.with_suffix(f"{path.suffix}.metadata.json"),
                       path.with_suffix(".metadata.json")) if p.exists()]


def load_session(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    metadata = read_json(sidecars(path)[0]) if sidecars(path) else {}
    if not isinstance(payload, dict) or not isinstance(metadata, dict):
        raise ValueError("Session and metadata must be JSON objects")
    # Reject conflicting identities instead of placing the same person on both sides of a fold.
    for field in ("participant_id", "session_id", "collection_country_code"):
        a, b = payload.get(field), metadata.get(field)
        if a is not None and b is not None and a != b:
            raise ValueError(f"Conflicting {field} in payload and sidecar")
    info = {**payload, **metadata}
    country = info.get("collection_country_code")
    record = {"session_id": str(info.get("session_id") or path.name.removesuffix(".gz").removesuffix(".json")),
              "participant_id": info.get("participant_id"), "country": country,
              "vehicle_type": str(payload.get("vehicle_type") or metadata.get("vehicle_type") or "").lower()}
    if country != "AU":
        return record, None, "missing_country" if country is None else "non_AU"
    if record["vehicle_type"] not in LABELS:
        return record, None, "unsupported_class"
    if not isinstance(record["participant_id"], str) or not record["participant_id"].strip():
        return record, None, "missing_participant"
    if "raw" in path.parts:
        index = path.parts.index("raw")
        if index + 2 < len(path.parts) and path.parts[index + 1] != record["participant_id"]:
            raise ValueError("Participant sidecar/path mismatch")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        return record, None, "no_samples"
    frame = pd.DataFrame(samples)
    if "ts" not in frame:
        raise ValueError("Session samples lack ts")
    frame = frame.reindex(columns=["ts", *[c for cols in SENSORS.values() for c in cols]])
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["ts"])
    frame["ts"] = frame.ts.astype("int64")
    # Payload trim bounds have the same precedence as the reference; zero is a valid bound.
    bounds = [payload.get(k) if payload.get(k) is not None else metadata.get(k)
              for k in ("trimmed_start_ms", "trimmed_end_ms")]
    if all(v is not None for v in bounds):
        frame = frame[frame.ts.between(int(bounds[0]), int(bounds[1]))]
    frame = frame.sort_values("ts", kind="stable").reset_index(drop=True)
    if frame.empty:
        return record, None, "no_usable_samples"
    record["duration_seconds"] = float((frame.ts.max() - frame.ts.min()) / 1000)
    if record["duration_seconds"] < 140:
        return record, None, "duration_below_140"
    record["label"] = LABELS[record["vehicle_type"]]
    return record, frame, None


def extract(frame, seconds):
    """Preserve reference boundaries, gap checks, feature ordering and float32 storage."""
    times = frame.ts.to_numpy(dtype=np.int64)
    diffs = np.diff(times)
    interval = int(np.median(diffs[diffs > 0])) if (diffs > 0).any() else seconds * 500
    end = int(times[-1]) + interval
    rows, generated = [], 0
    sensors = {}
    for sensor, columns in SENSORS.items():
        vectors = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
        valid = np.isfinite(vectors).all(axis=1)
        sensors[sensor] = (times[valid], np.linalg.norm(vectors[valid], axis=1))
    for start in range(int(times[0]), end - seconds * 1000 + 1, seconds * 500):
        generated += 1
        stop, features = start + seconds * 1000, []
        for sensor, (timestamps, magnitudes) in sensors.items():
            left, right = np.searchsorted(timestamps, [start, stop], side="left")
            t, values = timestamps[left:right], magnitudes[left:right]
            if len(t) < RATES[sensor] * seconds or np.diff(np.r_[start, t, stop]).max() > 500:
                break
            values = values[np.r_[True, np.diff(t) > 0]]
            stats = [np.mean(values), np.std(values, ddof=0), np.ptp(values), np.min(values)]
            features.extend(stats if sensor != "magnetometer" else stats[1:])
        else:
            rows.append(dict(window_start_ms=start, window_end_ms=stop,
                             **dict(zip(FEATURES, np.asarray(features, dtype=np.float32)))))
    return pd.DataFrame(rows, columns=["window_start_ms", "window_end_ms", *FEATURES]), generated


def grouped_folds(participants):
    participants = np.asarray(sorted(participants))
    if len(participants) < 3:
        raise ValueError(f"Three inner folds require at least 3 training participants; got {len(participants)}")
    return [{"train": participants[a].tolist(), "validation": participants[b].tolist()}
            for a, b in KFold(3, shuffle=True, random_state=42).split(participants)]


def make_splits(cohort):
    participants = sorted({r["participant_id"] for r in cohort})
    if len(participants) < 4:
        raise ValueError(f"Nested LOPO needs at least 4 participants; eligible={participants}")

    def validate(training, where):
        present = {r["label"] for r in cohort if r["participant_id"] in training}
        if present != {0, 1, 2}:
            raise ValueError(f"Missing training classes {sorted({0, 1, 2} - present)} in {where}; participants={training}")

    outer = []
    for heldout in participants:
        training = [p for p in participants if p != heldout]
        validate(training, f"outer {heldout}")
        folds = grouped_folds(training)
        for i, fold in enumerate(folds):
            validate(fold["train"], f"outer {heldout}, inner {i + 1}")
        outer.append({"participant": heldout, "train": training, "inner": folds})
    final = grouped_folds(participants)
    for i, fold in enumerate(final):
        validate(fold["train"], f"final inner {i + 1}")
    return {"split_seed": 42, "participants": participants, "outer": outer, "final": final}


def inventory(root, log):
    paths = discover(root)
    if not paths:
        raise ValueError(f"No session JSON/JSON.gz files found under {root}")
    result = []
    for i, path in enumerate(paths, 1):
        log.task("fingerprint input", current_session=str(path))
        result.append({"path": path.relative_to(root).as_posix(), "sha256": file_hash(path),
                       "sidecars": {p.relative_to(root).as_posix(): file_hash(p) for p in sidecars(path)}})
        log.event("input fingerprinted", completed=i, total=len(paths))
    return result


def verify_inputs(identity, log):
    if inventory(Path(identity["input_directory"]), log) != identity["inputs"]:
        raise ValueError("Input files changed: use a new run ID")


def prepare(root, run, attempt, threads):
    root = root.resolve()
    with attempt.stage("ingestion") as log:
        inputs = inventory(root, log)
        identity = dict(protocol=PROTOCOL, threads=threads, input_directory=str(root), inputs=inputs,
                        code=code_fingerprint(), environment=environment())
        identity_path = run / "identity.json"
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError("Inputs/settings/code/dependencies changed: use a new run ID")
        write_json(identity_path, identity)
        records, seen, excluded, hits = [], set(), Counter(), 0
        for i, item in enumerate(inputs, 1):
            path = root / item["path"]
            key = digest(item)
            cached = run / "ingested" / f"{key}.json"
            event_path = run / "events" / f"{key}.parquet"
            log.task("ingest session", current_session=item["path"])
            if cached.exists():
                record = read_json(cached)
                if not record["exclusion"] and file_hash(event_path) != record["event_sha256"]:
                    raise ValueError(f"Corrupt event cache: {event_path}")
                hits += 1
                log.event("ingestion cache hit", cache_hits=hits)
            else:
                try:
                    record, frame, reason = load_session(path)
                except Exception:
                    log.event("input failed", severity="ERROR", failed=1)
                    raise
                record.update(path=item["path"], key=key, exclusion=reason)
                if frame is not None:
                    atomic_write(event_path, lambda p: frame.to_parquet(p, index=False))
                    record["event_sha256"] = file_hash(event_path)
                write_json(cached, record)
            if record["session_id"] in seen:
                raise ValueError(f"Duplicate session identity: {record['session_id']}")
            seen.add(record["session_id"])
            records.append(record)
            if record["exclusion"]:
                excluded[record["exclusion"]] += 1
            log.event("session scanned", completed=i, total=len(inputs), accepted=i-sum(excluded.values()),
                      excluded=sum(excluded.values()), exclusion_reasons=dict(excluded), failed=0,
                      cache_hits=hits, newly_processed=i-hits)
        write_json(run / "inputs.json", records)
        log.event("ingestion outputs saved", output=str(run / "inputs.json"))

    with attempt.stage("feature-extraction") as log:
        eligible = [r for r in records if not r["exclusion"]]
        cohort, all_features, quality_excluded, hits = [], {w: [] for w in WINDOWS}, [], 0
        for i, record in enumerate(eligible, 1):
            log.task("extract session", current_session=record["session_id"])
            cache = run / "features-cache" / f"{record['key']}.joblib"
            cache_meta = cache.with_suffix(".json")
            import joblib
            if cache.exists() and cache_meta.exists():
                if file_hash(cache) != read_json(cache_meta)["sha256"]:
                    raise ValueError(f"Corrupt feature cache: {cache}")
                result = joblib.load(cache)
                hits += 1
                log.event("feature cache hit", cache_hits=hits)
            else:
                frame = pd.read_parquet(run / "events" / f"{record['key']}.parquet")
                result = {}
                for w in WINDOWS:
                    log.task("extract windows", current_session=record["session_id"], window=w)
                    features, generated = extract(frame, w)
                    result[w] = (features, generated)
                    log.event("window extraction complete", completed=i, total=len(eligible),
                              generated=generated, retained=len(features), rejected=generated-len(features))
                atomic_write(cache, lambda p: joblib.dump(result, p))
                write_json(cache_meta, {"sha256": file_hash(cache)})
            counts = {str(w): len(result[w][0]) for w in WINDOWS}
            if all(counts.values()):
                cohort.append({**record, "window_counts": counts})
                for w in WINDOWS:
                    features = result[w][0].copy()
                    for column in ("session_id", "participant_id", "label"):
                        features[column] = record[column]
                    all_features[w].append(features)
            else:
                quality_excluded.append({**record, "exclusion": "no_valid_window_at_every_size", "window_counts": counts})
            log.event("session features complete", completed=i, total=len(eligible), cache_hits=hits,
                      newly_processed=i-hits, cohort_sessions=len(cohort), excluded_sessions=len(quality_excluded),
                      window_counts=counts)
        coverage = {p: {label: sum(r["participant_id"] == p and r["label"] == code for r in cohort)
                        for label, code in LABELS.items()} for p in sorted({r["participant_id"] for r in cohort})}
        write_json(run / "cohort.json", {"sessions": cohort, "quality_exclusions": quality_excluded,
                   "participant_class_coverage": coverage, "participant_count": len(coverage),
                   "class_counts": dict(Counter(r["vehicle_type"] for r in cohort))})
        # Save diagnostic cohort even if the fixed splitting policy is infeasible.
        log.task("validate participant splits", output=str(run / "cohort.json"))
        splits = make_splits(cohort)
        write_json(run / "splits.json", splits)
        for w, frames in all_features.items():
            combined = pd.concat(frames, ignore_index=True)
            atomic_write(run / f"features-{w}.parquet", lambda p: combined.to_parquet(p, index=False))
        names = ["inputs.json", "cohort.json", "splits.json", *[f"features-{w}.parquet" for w in WINDOWS]]
        marker = {"identity": digest(identity), "files": {n: file_hash(run / n) for n in names},
                  "planned_fits": 2709 * len(coverage) + 2701}
        write_json(run / "prepared.json", marker)
        log.event("preparation outputs saved", output=str(run), participants=len(coverage),
                  planned_fits=marker["planned_fits"], completed=len(eligible), total=len(eligible))
