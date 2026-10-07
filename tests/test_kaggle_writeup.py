"""Tests for the Kaggle writeup word-count guard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "writeup" / "count_words.py"
WRITEUP = ROOT / "writeup" / "kaggle_writeup.md"
if not SCRIPT.is_file() or not WRITEUP.is_file():
    pytest.skip("Kaggle writeup files are not included in the code-only export",
                allow_module_level=True)

from writeup.count_words import count_words


def test_word_counter_enforces_submission_margin(tmp_path: Path) -> None:
    assert count_words("Title: don't over-count **bold** words.") == 5

    under_limit = subprocess.run(
        [sys.executable, str(SCRIPT), str(WRITEUP)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert under_limit.returncode == 0, under_limit.stderr
    assert "limit 1400" in under_limit.stdout

    too_long = tmp_path / "too-long.md"
    too_long.write_text("one two three four", encoding="utf-8")
    over_limit = subprocess.run(
        [sys.executable, str(SCRIPT), str(too_long), "--limit", "3"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert over_limit.returncode == 1
    assert "word limit exceeded" in over_limit.stderr
