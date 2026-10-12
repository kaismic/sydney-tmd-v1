from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

from .core import read_json, run_lock
from .config import load_config
from .data import prepare, verify_inputs
from .progress import Attempt


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local Sydney-only configurable transport mode study")
    parser.add_argument("command", choices=["prepare", "evaluate", "report", "final", "all"])
    parser.add_argument("--run-id", required=True, help="Immutable experiment identity; use a new ID for changed inputs/settings")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--input", type=Path, help="Local collector session directory (prepare/all)")
    parser.add_argument("--config", type=Path, help="Experiment JSON; defaults to experiment.json on prepare/all")
    parser.add_argument("--threads", type=int, default=None, help="XGBoost CPU threads; default 1 on prepare, frozen thereafter")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("--run-id must contain only letters, digits, underscores, dots, and hyphens")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    if args.command in ("prepare", "all") and args.input is None:
        parser.error("--input is required for prepare/all")
    run = args.runs_dir.resolve() / args.run_id
    if args.input and (run.is_relative_to(args.input.resolve()) or args.logs_dir.resolve().is_relative_to(args.input.resolve())):
        parser.error("Run/log output directories must be outside the input tree")
    try:
        attempt = Attempt(args.logs_dir, args.run_id)
        with run_lock(run):
            if args.command in ("prepare", "all"):
                prepare(args.input, run, attempt, args.threads or 1, load_config(args.config))
            else:
                with attempt.stage("ingestion") as log:
                    identity = read_json(run / "identity.json")
                    if args.config is not None and load_config(args.config) != identity.get("configuration"):
                        raise ValueError("Experiment configuration changed: use a new run ID")
                    if args.threads is not None and args.threads != identity["threads"]:
                        raise ValueError("Thread setting changed: use a new run ID")
                    verify_inputs(identity, log)
                    log.event("frozen inputs verified", output=str(run / "identity.json"))
            if args.command in ("evaluate", "all"):
                from .training import evaluate
                with attempt.stage("evaluation") as log:
                    evaluate(run, log)
            if args.command in ("report", "all"):
                from .reporting import report
                with attempt.stage("reporting") as log:
                    report(run, log)
            if args.command in ("final", "all"):
                from .training import train_final
                with attempt.stage("evaluation") as log:
                    train_final(run, log)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}. Resumable state: {run}", file=sys.stderr, flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
