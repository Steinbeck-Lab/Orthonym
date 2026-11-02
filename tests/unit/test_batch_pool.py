"""``orthonym.batch_pool.run_ordered``: N spawn workers, results in input order, a crashed row isolated."""
import pytest

from orthonym import batch_pool
from tests.support import batch_pool_rows as rows

pytestmark = pytest.mark.unit


def test_results_come_back_in_input_order():
    items = [f"r{i}" for i in range(20)]
    assert list(batch_pool.run_ordered(items, 4, rows.upper, rows.noop_init)) == [i.upper() for i in items]


def test_a_killed_worker_costs_only_its_own_row():
    out = list(batch_pool.run_ordered(["a", "EXIT", "b", "c"], 2, rows.exit_on_marker, rows.noop_init,
                                      error_row=rows.error_line))
    assert out == ["a", rows.ERROR_LINE, "b", "c"]


def test_without_an_error_row_the_break_is_raised():
    from concurrent.futures.process import BrokenProcessPool
    with pytest.raises(BrokenProcessPool):
        list(batch_pool.run_ordered(["EXIT"], 1, rows.exit_on_marker, rows.noop_init))


def test_jobs_must_be_positive():
    with pytest.raises(ValueError):
        list(batch_pool.run_ordered(["a"], 0, rows.upper, rows.noop_init))


def test_a_crash_reruns_alone_only_the_rows_in_flight(monkeypatch):
    """A crash fails every unfinished row of the pool; only the rows in flight (at most ``jobs``) are named again
    alone, and the rest go on in a new pool (a crash early in a long file re-ran every later row alone before)."""
    alone = []
    real = batch_pool._alone
    monkeypatch.setattr(batch_pool, "_alone", lambda item, *a: alone.append(item) or real(item, *a))
    items = ["a", "EXIT"] + [f"r{i}" for i in range(20)]
    out = list(batch_pool.run_ordered(items, 2, rows.exit_on_marker, rows.noop_init, error_row=rows.error_line))
    assert out == ["a", rows.ERROR_LINE] + [f"r{i}" for i in range(20)]
    assert "EXIT" in alone and len(alone) <= 2, alone


def test_a_worker_ends_when_its_parent_is_killed(tmp_path):
    """A parent killed alone (not the process group) took no worker with it: they named on as orphans."""
    import os
    import signal
    import subprocess
    import sys
    import time
    marker = tmp_path / "pid"
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from orthonym import batch_pool\nfrom tests.support import batch_pool_rows as rows\n"
            "list(batch_pool.run_ordered([%r], 1, rows.report_pid_and_sleep, rows.noop_init))\n"
            % (os.getcwd(), str(marker)))
    parent = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        for _ in range(300):
            if marker.exists() and marker.read_text():
                break
            time.sleep(0.1)
        worker = int(marker.read_text())
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=10)
        for _ in range(50):
            if not os.path.exists(f"/proc/{worker}"):
                break
            try:
                os.kill(worker, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(worker, signal.SIGKILL)
            pytest.fail("the worker outlived its parent by 5 s")
    finally:
        if parent.poll() is None:
            parent.kill()
