"""Flushed terminal/file events and independent heartbeats during blocking model fits."""
from __future__ import annotations

import _thread
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time
import traceback
import warnings
from uuid import uuid4


class Progress:
    def __init__(self, directory: Path, run_id: str, stage: str, interval=20.0):
        self.path = directory / f"{stage}.log"
        self.run_id, self.stage, self.interval = run_id, stage, interval
        self.start = time.monotonic()
        self.context = {"task": "starting"}
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.error = None
        self.last = self.start

    def __enter__(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.path.open("a", encoding="utf-8", buffering=1)
        except OSError as exc:
            print(f"ERROR cannot create log {self.path}: {exc}", file=sys.stderr, flush=True)
            raise
        try:
            self.event("stage started", output=str(self.path))
        except BaseException:
            self.stream.close()
            raise
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.warning_context = warnings.catch_warnings()
        self.warning_context.__enter__()
        warnings.showwarning = self._warning
        self.thread.start()
        return self

    def _warning(self, message, category, filename, lineno, file=None, line=None):
        self.event(str(message), severity="WARNING", warning_category=category.__name__,
                   warning_location=f"{filename}:{lineno}")

    def event(self, message, severity="INFO", **context):
        with self.lock:
            if self.error:
                raise RuntimeError(f"Saved logging failed: {self.path}") from self.error
            self.context.update(context)
            elapsed = time.monotonic() - self.start
            event = dict(timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
                         severity=severity, run_id=self.run_id, stage=self.stage,
                         elapsed_seconds=round(elapsed, 3), **self.context, message=message)
            line = json.dumps(event, ensure_ascii=False, allow_nan=False)
            try:
                self.stream.write(line + "\n")
                self.stream.flush()
            except (OSError, ValueError) as exc:
                self.error = exc
                print(f"ERROR cannot write log {self.path}: {exc}", file=sys.stderr, flush=True)
                raise
            print(line, flush=True)
            self.last = time.monotonic()

    def task(self, task, **context):
        self.event("task started", task=task, completed=None, total=None, **context)

    def _heartbeat(self):
        while not self.stop.wait(self.interval / 2):
            try:
                with self.lock:
                    if time.monotonic() - self.last >= self.interval:
                        self.event("heartbeat: work active")
            except BaseException as exc:
                self.error = exc
                _thread.interrupt_main()
                return

    def __exit__(self, kind, value, tb):
        self.stop.set()
        self.thread.join()
        try:
            if kind:
                self.event("stage interrupted" if issubclass(kind, KeyboardInterrupt) else "stage failed",
                           severity="ERROR", traceback="".join(traceback.format_exception(kind, value, tb)),
                           recovery="Resumable artifacts and completed trials retained; rerun the same command")
            else:
                self.event("stage complete")
        finally:
            self.warning_context.__exit__(kind, value, tb)
            self.stream.close()


class Attempt:
    def __init__(self, root, run_id):
        self.run_id = run_id
        stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        self.directory = Path(root) / run_id / f"{stamp}-{uuid4().hex[:8]}"
        self.directory.mkdir(parents=True)
        print(f"Logs: {self.directory.resolve()}", flush=True)

    def stage(self, name):
        return Progress(self.directory, self.run_id, name)
