"""The in-process JVM must never keep the interpreter from exiting.

A worker thread that calls into Java (the per-molecule ThreadPoolExecutor
timeout thread of ``scripts/benchmark_multi_corpus.py``) used to be attached as
a USER thread: explicitly by ``jvm_bridge._attach_thread``, and implicitly when
it was the thread that called ``startJVM``. JPype does not detach a thread when
it ends, so the dead worker stayed alive in the JVM's view, and ``DestroyJavaVM``
(run by JPype's atexit hook ``_JTerminate``) waited for it forever: the script
wrote all its outputs in ~6 s and then hung at exit.

The probe runs in a subprocess because the hang only shows at interpreter exit.
Thread 1 starts the JVM (the creating-thread path); thread 2 finds it running and
attaches (the ``_attach_thread`` path). Different names, so the process-wide
OPSIN memo cannot serve thread 2 without a Java call.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.jars import jar_or_none

PROJECT_ROOT = Path(__file__).resolve().parents[3]

pytest.importorskip("jpype")

_PROBE = r"""
import threading
import jpype
from orthonym import jvm_bridge

flags = []

def work(name):
    stdout, served = jvm_bridge.opsin_stdout(name, False)
    flags.append((name, served, bool(jpype.java.lang.Thread.currentThread().isDaemon())))

for name in ("ethane", "propane"):
    t = threading.Thread(target=work, args=(name,))
    t.start()
    t.join()
print("FLAGS", flags, flush=True)
"""


def test_worker_threads_attach_as_daemon_and_the_process_exits():
    if jar_or_none() is None:
        pytest.skip("OPSIN jar required: the probe needs the in-process JVM")
    env = {k: v for k, v in os.environ.items() if k != "ORTHONYM_DISABLE_JPYPE"}
    try:
        r = subprocess.run(
            [sys.executable, "-c", _PROBE],
            cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=90,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        pytest.fail(
            "the interpreter did not exit within 90 s after its Java-calling worker "
            "threads ended (DestroyJavaVM waits for a thread attached as a USER "
            f"thread); probe output: {out!r}"
        )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "FLAGS [('ethane', True, True), ('propane', True, True)]" in r.stdout, r.stdout
