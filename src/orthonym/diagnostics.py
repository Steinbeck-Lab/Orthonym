"""Crash diagnostics shared by every entry point (CLI, eval harness, batch shards).

Why this exists (audit 2026-09-03): between Aug 30 and Sep 2 the batch runs left
72 ``hs_err_pid*.log`` files. Every one was a SIGSEGV in native code (RDKit's
boost.python layer, ``_jpype.so`` or libc) on the Python main thread, and none
named a Python line, because nothing had enabled:mod:`faulthandler`. With it on,
the interpreter prints the Python stack of every thread to stderr on SIGSEGV,
SIGFPE, SIGABRT and SIGBUS before the process dies, so the molecule and the
call site are recoverable from the log.

Interaction with the in-process JVM: HotSpot installs its own SIGSEGV handler
when ``jvm_bridge`` starts it and chains to the handler that was installed
before, so call this BEFORE the first naming call (the JVM starts lazily on the
first OPSIN round-trip). Reproducing a crash with ``ORTHONYM_DISABLE_JPYPE=1``
removes the JVM from the picture entirely and gives the cleanest traceback.
"""
from __future__ import annotations

import faulthandler
import sys


def enable_crash_traceback() -> None:
    """Turn on ``faulthandler`` for this process (idempotent, never raises).

    Writes to ``sys.stderr``. If stderr is unavailable (embedded interpreters,
    closed streams) the call is a no-op rather than an error: diagnostics must
    never change whether a name is produced.
    """
    if faulthandler.is_enabled():
        return
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except (AttributeError, ValueError, OSError, RuntimeError):
        # sys.stderr is None or has no fileno (e.g. some embedded hosts).
        return


def strict_mode() -> bool:
    """``ORTHONYM_STRICT=1``: re-raise an exception that escaped a producer
    instead of degrading to an abstention at the top of ``name`` /
    ``name_tiered``. Read on every call so a test or harness can flip it
    without restarting the process. Default off: the always-emit contract holds.
    """
    import os
    return os.environ.get("ORTHONYM_STRICT", "").strip().lower() in ("1", "true", "on", "yes")
