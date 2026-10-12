from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform
from uuid import uuid4

WINDOWS = (100, 120, 140)
SEEDS = (42, 43, 44)
LABELS = {"bus": 0, "car": 1, "train": 2}
SENSORS = {"accelerometer": ("ax", "ay", "az"),
           "gyroscope": ("gx", "gy", "gz"), "magnetometer": ("mx", "my", "mz")}
RATES = {"accelerometer": 30, "gyroscope": 4, "magnetometer": 2}
FEATURES = [f"{sensor}#{stat}" for sensor in SENSORS
            for stat in (("mean", "standard_deviation", "range", "minimum")
                         if sensor != "magnetometer" else
                         ("standard_deviation", "range", "minimum"))]
PROTOCOL = dict(schema=1, windows=list(WINDOWS), steps=[50, 60, 70], seeds=list(SEEDS),
                labels=LABELS, minimum_seconds=140, maximum_seconds=None,
                minimum_rates=RATES, maximum_gap_ms=500, features=FEATURES,
                split_seed=42, inner_folds=3, completed_trials=100, startup_trials=10,
                bootstrap_draws=5000, analysis_seed=42,
                sensor_units={"accelerometer": "m/s^2", "gyroscope": "rad/s",
                              "magnetometer": "microtesla"},
                preprocessing={"imputation": "training-partition median", "missing_indicators": True,
                               "sample_weights": "training-partition class-balanced", "duration_balancing": False},
                classifier={"family": "XGBoost", "tree_method": "hist", "device": "cpu",
                            "objective": "multi:softprob", "early_stopping": False, "pruning": False},
                magnitude="Euclidean norm; units unchanged from collector",
                standard_deviation_ddof=0, feature_dtype="float32",
                boundaries="[start,end); final sample plus median positive interval",
                duplicates="quality counts all rows; features use first finite vector per timestamp")
DEPENDENCIES = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "optuna",
                "joblib", "matplotlib", "pyarrow", "sqlalchemy")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    atomic_write(path, lambda temp: temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"))


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        writer(temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def environment():
    return {"python": platform.python_version(),
            **{name: version(name) for name in DEPENDENCIES}}


def code_fingerprint():
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})


@contextmanager
def run_lock(run):
    """OS lock releases automatically after crashes; never remove another process's lock."""
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".lock").open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another invocation is using {run}") from exc
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def verify_prepared(run):
    identity = read_json(run / "identity.json")
    if "configuration" in identity and read_json(run / "experiment.json") != identity["configuration"]:
        raise ValueError("Saved experiment configuration changed: prepare with a new run ID")
    if identity["code"] != code_fingerprint() or identity["environment"] != environment():
        raise ValueError("Code/dependencies changed: prepare with a new run ID")
    marker = read_json(run / "prepared.json")
    if marker["identity"] != digest(identity):
        raise ValueError("Prepared identity mismatch")
    for name, expected in marker["files"].items():
        if file_hash(run / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}")
    return identity, marker
