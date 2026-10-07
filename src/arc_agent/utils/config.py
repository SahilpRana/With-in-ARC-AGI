"""Validated YAML configuration loading and nested override merging."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject empty or non-mapping documents."""
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"configuration at {path} must have a mapping at its root")
    return value


def merge_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge non-None CLI overrides over a loaded configuration."""
    result = dict(base)
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = merge_overrides(result[key], value)
        else:
            result[key] = value
    return result