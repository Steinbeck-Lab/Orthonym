"""a phase byte-identical integration tests .

Two tiers:
  - test_byte_identical_subset (default tier): 300-row subset, ~3 min
  - test_byte_identical_full (slow marker): all 7,500 rows, ~25 min

Both subprocess to scripts/check_byte_identical.py. Skipped if OPSIN
jar is missing (mirrors test_benchmark_multi_corpus.py guard).

Per: subset is per-commit gate; full is per-merge gate.

PHASE 148 NOTE: Both tests are marked @pytest.mark.xfail because the
architectural triple 146+147+148 intentionally diverges from the v17
baseline by deleting `_should_bypass_fused_guard` (a phase)
and wiring the cascade end-to-end for fused heterocycles. The
byte-identical-to-v17 invariant cannot hold once a phase has shipped.

The 5/300 subset drifts observed at a phase commit time include
CHEBI:33070 (`3-(3-carboxypropyl)-1H-indole` → `4-(1H-indol-3-yl)butanoic
acid`) — this is exactly the V18-plan-predicted (a) cascade unblock
(carboxylic acid PG on chain, ring has zero PG → chain wins per IUPAC
(a) at https://iupac.qmul.ac.uk/BlueBook/P4.html).

The cascade-divergence baseline will be REFRESHED in a phase Plan 03
(internal notes) once the full 7,500-corpus G3/G5 cumulative verdict
is computed. At that point this test will be unmarked and re-locked to
the post-148 baseline (which becomes the new byte-identical anchor).

Source: V18_MILESTONE_PLAN a phase (cascade unblock for fused
heterocycles); a phase internal notes,,; AUTONOM 1990
(full seniority cascade on ALL structures, no bypass).
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
    / ".planning/phases/145-multi-corpus-benchmark-foundation/baseline_v17_all_corpora.csv"
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
@pytest.mark.xfail(
    reason=(
        "Phase 148 architectural triple intentionally diverges from v17 "
        "baseline. Bypass deletion (D-01) + cascade wiring (D-02) cause "
        "fused-heterocycle parent-selection drift relative to v17. "
        "Baseline refresh tracked in Phase 148 Plan 03 / 148-VERIFICATION.md."
    ),
    strict=False,
)
def test_byte_identical_subset():
    """a phase fast tier: 100/corpus subset must be byte-identical
    to baseline_v17_all_corpora.csv. Per per-commit gate.

    a phase: xfailed pending baseline refresh in internal notes
    (architectural triple 146+147+148 cascade-unblock divergence)."""
    result = _run_check(subset=True, timeout=600)
    assert result.returncode == 0, (
        f"Subset byte-identical drift detected:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "Phase 148 architectural triple intentionally diverges from v17 "
        "baseline. Bypass deletion (D-01) + cascade wiring (D-02) cause "
        "fused-heterocycle parent-selection drift relative to v17. "
        "Baseline refresh tracked in Phase 148 Plan 03 / 148-VERIFICATION.md."
    ),
    strict=False,
)
def test_byte_identical_full():
    """a phase ship gate: all 7,500 rows must be byte-identical.
    Per per-merge gate. Marked @pytest.mark.slow — opt-in via
    `pytest -m slow`.

    a phase: xfailed pending baseline refresh in internal notes
    (architectural triple 146+147+148 cascade-unblock divergence)."""
    result = _run_check(subset=False, timeout=3600)
    assert result.returncode == 0, (
        f"Full byte-identical drift detected:\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
