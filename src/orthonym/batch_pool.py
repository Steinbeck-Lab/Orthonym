"""Name a list of inputs in worker processes and hand the results back in input order.

``run_ordered`` is the parallel path of ``orthonym --batch --jobs N`` and of ``scripts/speed_bench.py``. Each worker is a
fresh ``spawn`` process: ``init_fn`` runs once in it (build the namer, limit the BLAS threads), then ``row_fn`` runs once
per input; a worker ends itself when its parent is gone (``_worker_start``). The parent keeps at most ``jobs`` inputs
in flight, one per worker, sends the next as one comes back, and yields the results in index order, so the output of
N workers is the output of one, line for line.

A worker that dies in the middle of a row (a crash of the JVM or of RDKit, ``os._exit``) breaks the whole pool:
``concurrent.futures`` then fails every unfinished future with ``BrokenProcessPool``, the rows that only shared the pool
included. The rows in flight at the break (at most ``jobs``) are sent again, each in a pool of its own, so a second
break can only be the row's own; that index yields ``error_row(item)``. The rows not yet sent go on in a new pool.
(``multiprocessing.Pool.imap`` is not used: it never reports a worker killed mid-task and blocks for ever.)

The JVM budget (``orthonym.jvm_budget``) is the caller's: a caller that reserves slots reserves the N once, and the
workers reserve none (a worker that asks for a slot of a full budget waits until its timeout).
"""
import collections
import concurrent.futures as cf
import functools
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures.process import BrokenProcessPool
from typing import Callable, Iterator, List, Optional, TypeVar

T = TypeVar("T")


def _worker_start(init_fn) -> None:
    """The start of a worker: a watch that ends it within a second of its parent's end, then ``init_fn``.

    A worker learns that its parent is gone only when it next reads its queue, after the row it is naming; a parent
    that is killed (a Ctrl-C reaches the whole process group, a ``kill`` only the parent) left its workers naming on
    as orphans at full CPU, one for 49 minutes and counting."""
    parent = os.getppid()

    def watch():
        while True:
            time.sleep(1.0)
            if os.getppid() != parent:
                os._exit(1)
    threading.Thread(target=watch, name="batch-pool-parent-watch", daemon=True).start()
    init_fn()


def _pool(jobs: int, init_fn, ctx):
    return cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx,
                                  initializer=functools.partial(_worker_start, init_fn))


def _alone(item: str, row_fn, init_fn, ctx):
    """``row_fn(item)`` in a pool of one fresh worker; raises ``BrokenProcessPool`` when the row kills it."""
    with _pool(1, init_fn, ctx) as pool:
        return pool.submit(row_fn, item).result()


def run_ordered(items: List[str], jobs: int, row_fn: Callable[[str], T], init_fn: Callable[[], None],
                error_row: Optional[Callable[[str], T]] = None) -> Iterator[T]:
    """Yield ``row_fn(item)`` for every item, in input order, computed in ``jobs`` spawn workers.

    ``row_fn`` and ``init_fn`` must be importable module-level functions (spawn pickles them by name). An item whose
    row kills its worker when it runs alone yields ``error_row(item)``, or raises ``BrokenProcessPool`` when no
    ``error_row`` is given."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    ctx = mp.get_context("spawn")
    jobs = max(1, min(jobs, len(items)))
    results: dict = {}
    todo = collections.deque(range(len(items)))
    next_out = 0
    while todo:
        broken: List[int] = []
        with _pool(jobs, init_fn, ctx) as pool:
            inflight: dict = {}
            while todo or inflight:
                while todo and len(inflight) < jobs:
                    i = todo.popleft()
                    inflight[pool.submit(row_fn, items[i])] = i
                done, _ = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    i = inflight.pop(fut)
                    try:
                        results[i] = fut.result()
                    except BrokenProcessPool:
                        broken.append(i)
                if broken:          # the pool is dead: every row still in flight fails with it
                    broken.extend(inflight.values())
                    break
                while next_out in results:
                    yield results.pop(next_out)
                    next_out += 1
        for i in sorted(broken):
            try:
                results[i] = _alone(items[i], row_fn, init_fn, ctx)
            except BrokenProcessPool:
                if error_row is None:
                    raise
                results[i] = error_row(items[i])
        while next_out in results:
            yield results.pop(next_out)
            next_out += 1
