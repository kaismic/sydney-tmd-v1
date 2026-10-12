"""Validated, immutable controls for one Sydney experiment."""
from __future__ import annotations

import math
from pathlib import Path

from .core import LABELS, read_json


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "experiment.json"
STRATEGIES = ("participant_lopo", "shuffled_session_holdout")


def validate_config(raw):
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "evaluation_strategy", "calibration_fraction"}:
        raise ValueError("Experiment config must contain only schema_version, evaluation_strategy, and calibration_fraction")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("Experiment schema_version must be 1")
    strategy = raw["evaluation_strategy"]
    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported evaluation_strategy: {strategy!r}")
    fractions = raw["calibration_fraction"]
    if not isinstance(fractions, dict) or set(fractions) != set(LABELS):
        raise ValueError(f"calibration_fraction must contain exactly {sorted(LABELS)}")
    for mode, value in fractions.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"calibration_fraction.{mode} must be a finite number in (0, 1]")
    return {"schema_version": 1, "evaluation_strategy": strategy,
            "calibration_fraction": {mode: float(fractions[mode]) for mode in LABELS}}


def load_config(path=None):
    return validate_config(read_json(DEFAULT_CONFIG_PATH if path is None else path))
