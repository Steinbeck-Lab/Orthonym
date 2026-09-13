"""Process-level tuning for batch naming workers (perf lever A1 + A11, 2026-09-13).

Two knobs, both invisible to the names and tiers the engine emits:

* **Python cyclic GC.** Naming allocates millions of small objects per molecule (RDKit
  wrappers, tuples, dicts). With the default thresholds ``(700, 10, 10)`` the collector
  runs every ~700 net allocations and rescans the long-lived module data each time it
  reaches generation 2. Measured on 100 fixed-seed molecules, one core, twice each
  (2026-09-12): default 36.3 s; ``gc.freeze`` + threshold ``(50_000, 20, 20)`` 31.8 s
  (-12 %); freeze + ``gc.disable`` 30.1 s (-17 %). Names identical in every mode.
  RSS grew LESS with the collector frozen (+42 MB vs +185 MB over 150 molecules), because
  the growth is lexicon warm-up, not garbage. The collector stays ON here (high
  threshold) so a 16,000-molecule shard cannot accumulate cyclic garbage unbounded.

* **BLAS/OpenMP thread pools.** numpy's bundled OpenBLAS spawns one thread per CPU
  (59 idle threads per worker on this 60-vCPU host, 2,400 across a 41-worker run). They
  use no CPU but cost memory and scheduler noise; one thread is the right size for a
  process that never calls BLAS in parallel.

Opt-in. The repo's batch runners (eval/harness.py, scripts/perf/name_sample.py) call
:func:`tune_batch_process` explicitly; any other runner sets ``ORTHONYM_GC_TUNE=on`` and
the first ``Orthonym`` constructed in the process applies it via
:func:`maybe_tune_from_env`. ``ORTHONYM_GC_TUNE=off`` disables both paths (A/B runs).
A library user who imports orthonym and never sets the variable is untouched.
"""
from __future__ import annotations

import gc
import os
from typing import Tuple

_ENV = "ORTHONYM_GC_TUNE"
_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_state = {"tuned": False}

DEFAULT_THRESHOLD: Tuple[int, int, int] = (50_000, 20, 20)


def _disabled_by_env() -> bool:
    return os.environ.get(_ENV, "").strip().lower() in ("off", "0", "false", "no")


def tune_batch_process(threshold: Tuple[int, int, int] = DEFAULT_THRESHOLD, *,
                       freeze: bool = True, blas_threads: int = 1) -> bool:
    """Apply the batch-worker tuning once per process. Returns True if applied (now or
    earlier), False if ``ORTHONYM_GC_TUNE=off``. Idempotent; safe to call per molecule."""
    if _disabled_by_env():
        return False
    if _state["tuned"]:
        return True
    for var in _THREAD_VARS:
        # Only effective if set before the library that reads it is loaded; harmless later.
        os.environ.setdefault(var, str(blas_threads))
    gc.collect()
    if freeze:
        gc.freeze()
    gc.set_threshold(*threshold)
    _state["tuned"] = True
    return True


def maybe_tune_from_env() -> bool:
    """Apply:func:`tune_batch_process` iff ``ORTHONYM_GC_TUNE`` is ``on``/``1``/``true``."""
    if os.environ.get(_ENV, "").strip().lower() in ("on", "1", "true", "yes"):
        return tune_batch_process()
    return False


def is_tuned() -> bool:
    return _state["tuned"]
