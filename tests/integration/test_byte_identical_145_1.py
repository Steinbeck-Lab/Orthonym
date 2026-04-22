"""Phase 145.1 byte-identical integration tests (SC-4).

Two tiers:
  - test_byte_identical_subset (default tier): 300-row subset, ~3 min
  - test_byte_identical_full (slow marker): all 7,500 rows, ~25 min

Both subprocess to  Skipped if OPSIN
jar is missing (mirrors test_benchmark_multi_corpus.py guard).

Per D-11: subset is per-commit gate; full is per-merge gate.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "check_byte_identical.py"
BASELINE_CSV = (
    PROJECT_ROOT
    / ""
)
OPSIN_JAR = PROJECT_ROOT / "opsin-cli-2.9.0-jar-with-dependencies.jar"

pytestmark = pytest.mark.skipif(
    not OPSIN_JAR.exists() or not BASELINE_CSV.exists(),
    reason="OPSIN jar + Phase 145 baseline required for byte-identical tests",
)


def _run_check(subset: bool, timeout: int):
    cmd = [sys.executable, str(SCRIPT)]
    cmd.append("--subset" if subset else "--full")
    cmd.append("--strict")
    return subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=timeout,
    )


@pytest.mark.integration
def test_byte_identical_subset():
    """Phase 145.1 SC-4 fast tier: 100/corpus subset must be byte-identical
    to baseline_v17_all_corpora.csv. Per D-11 per-commit gate."""
    result = _run_check(subset=True, timeout=600)
    assert result.returncode == 0, (
        f"Subset byte-identical drift detected:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_byte_identical_full():
    """Phase 145.1 SC-4 ship gate: all 7,500 rows must be byte-identical.
    Per D-11 per-merge gate. Marked @pytest.mark.slow — opt-in via
    `pytest -m slow`."""
    result = _run_check(subset=False, timeout=3600)
    assert result.returncode == 0, (
        f"Full byte-identical drift detected:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
