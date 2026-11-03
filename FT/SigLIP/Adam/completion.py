# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

"""Atomic output, locking, completion validation, and resume support."""

from __future__ import annotations

import contextlib
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence


class CompletionError(RuntimeError):
    pass


def _replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n").encode()
    _replace_bytes(Path(path), payload)


def atomic_text(path: Path, value: str) -> None:
    _replace_bytes(Path(path), value.encode())


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_csv_gz(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        with gzip.open(temporary, "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksums(run_dir: Path, relative_paths: Iterable[str]) -> Dict[str, str]:
    result = {}
    for relative in sorted(set(relative_paths)):
        path = Path(run_dir) / relative
        if path.is_file():
            result[relative] = sha256_file(path)
    return result


@contextlib.contextmanager
def run_lock(run_dir: Path, blocking: bool = False) -> Iterator[None]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".run.lock"
    stream = lock_path.open("a+")
    operation = fcntl.LOCK_EX
    if not blocking:
        operation |= fcntl.LOCK_NB
    try:
        try:
            fcntl.flock(stream.fileno(), operation)
        except BlockingIOError as exc:
            raise CompletionError(f"Run is already locked: {run_dir}") from exc
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_done(run_dir: Path, expected_epoch: int, zero_shot: bool = False) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    done_path = run_dir / "DONE.json"
    if not done_path.is_file():
        raise CompletionError(f"Missing {done_path}")
    done = json.loads(done_path.read_text())
    if done.get("status") != "complete":
        raise CompletionError("DONE.json does not report complete")
    if int(done.get("expected_final_epoch", -1)) != int(expected_epoch):
        raise CompletionError("DONE.json expected epoch mismatch")
    summary_name = "zero_shot_metrics.json" if zero_shot else "summary.json"
    summary_path = run_dir / summary_name
    if not summary_path.is_file():
        raise CompletionError(f"Missing {summary_name}")
    summary = json.loads(summary_path.read_text())
    metric_key = "test_accuracy" if zero_shot else "final_test_accuracy"
    if not _finite(summary.get(metric_key)):
        raise CompletionError(f"Non-finite or missing {metric_key}")
    if not zero_shot and int(summary.get("final_epoch", -1)) != int(expected_epoch):
        raise CompletionError("Summary final epoch mismatch")
    recorded = done.get("checksums", {})
    if not recorded:
        raise CompletionError("DONE.json has no output checksums")
    for relative, digest in recorded.items():
        path = run_dir / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise CompletionError(f"Checksum failure: {relative}")
    return done


def write_done(
    run_dir: Path,
    expected_epoch: int,
    required_outputs: Iterable[str],
    zero_shot: bool = False,
) -> None:
    run_dir = Path(run_dir)
    output_checksums = checksums(run_dir, required_outputs)
    missing = sorted(set(required_outputs) - set(output_checksums))
    if missing:
        raise CompletionError("Cannot complete run; missing outputs: " + ", ".join(missing))
    payload = {
        "status": "complete",
        "expected_final_epoch": int(expected_epoch),
        "zero_shot": bool(zero_shot),
        "checksums": output_checksums,
    }
    atomic_json(run_dir / "DONE.json", payload)
    validate_done(run_dir, expected_epoch, zero_shot=zero_shot)


__all__ = [
    "CompletionError",
    "atomic_json",
    "atomic_text",
    "atomic_csv",
    "atomic_csv_gz",
    "sha256_file",
    "checksums",
    "run_lock",
    "validate_done",
    "write_done",
]
