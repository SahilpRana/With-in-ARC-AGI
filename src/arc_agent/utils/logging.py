"""Structured JSON Lines logging for reproducible episode runs."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping


def append_run(log_path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one JSON-serializable episode record, creating parent folders."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def git_revision(cwd: str | Path | None = None) -> str | None:
    """Return the current short Git revision, or None outside a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def write_run_manifest(
    log_path: str | Path,
    run_id: str,
    config: Mapping[str, Any],
    *,
    cwd: str | Path | None = None,
) -> Path:
    """Write run metadata beside logs and return its unique manifest path."""
    path = Path(log_path)
    manifest = path.with_name(f"{path.stem}.{run_id}.manifest.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "git_revision": git_revision(cwd),
        "python_version": platform.python_version(),
        "config": config,
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
