"""W7 regression: name_with_tree is concurrent-safe via contextvars.ContextVar.

Per internal notes part B resolution: the inner-dispatch NamingResult is
captured into a contextvars.ContextVar slot (PEP 567). ContextVar gives
thread-local AND asyncio-task-local isolation, so two threads invoking
Orthonym.name_with_tree on different SMILES never see each other's slots.
"""
from __future__ import annotations

import threading

from orthonym import Orthonym


def test_name_with_tree_concurrent_threads_isolated():
    """Two threads call Orthonym.name_with_tree on different SMILES;
    each must receive the correct name (no cross-thread state leak)."""
    errors = []

    def worker(thread_id: int, smi: str, expected: str):
        try:
            namer = Orthonym()
            for _ in range(50):
                r = namer.name_with_tree(smi)
                if r.name != expected:
                    errors.append(
                        f"thread {thread_id} got {r.name!r} "
                        f"(expected {expected!r}) on smi={smi!r}"
                    )
        except Exception as e:
            errors.append(f"thread {thread_id}: {e!r}")

    t1 = threading.Thread(target=worker, args=(1, "CCO", "ethanol"))
    t2 = threading.Thread(target=worker, args=(2, "CC(C)O", "propan-2-ol"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not errors, f"Concurrency safety violations: {errors}"
