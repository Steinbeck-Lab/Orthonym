"""eval/harness.py gives every run its own file.

The run file used to be named by the UTC second alone, so evals started together
(the four breadth evals of a task, run in parallel) wrote one file and the last
writer won; only their stdout logs were right (TRIAGE.md ' outcome'). The run
id, which is also the file stem, now carries the tag and the process id.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_EVAL = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(_EVAL))
pytestmark = pytest.mark.skipif(not (_EVAL / "harness.py").exists(),
                                reason="eval/harness.py is not in this checkout")

_T = datetime(2026, 9, 26, 5, 13, 48, tzinfo=timezone.utc)


def test_parallel_runs_in_one_second_get_distinct_ids():
    import harness
    ids = {harness.run_id_for(_T, tag, pid) for tag, pid in
           [("t10-m-be", 100), ("t10-m-pin", 101), ("t10-d-be", 102), ("t10-d-pin", 103)]}
    assert len(ids) == 4
    # Same tag, same second, different process: still distinct.
    assert harness.run_id_for(_T, "x", 1) != harness.run_id_for(_T, "x", 2)


def test_run_id_is_a_safe_file_stem():
    import harness
    assert harness.run_id_for(_T, "t12-m-be", 4242) == "20260926T051348Z-t12-m-be-p4242"
    assert harness.run_id_for(_T, "", 7) == "20260926T051348Z-p7"
    assert harness.run_id_for(_T, "a b/../c", 7) == "20260926T051348Z-a-b-..-c-p7"
    assert "/" not in harness.run_id_for(_T, "../../etc/passwd", 7)
