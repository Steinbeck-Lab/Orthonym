"""Machine-wide JVM concurrency budget, shared by every session and script.

Replaces the blanket folk rule "never run two OPSIN jobs concurrently" with a
measured budget and real enforcement. The rule it replaces was never measured;
worse, it did not describe the code -- `` has always run
``mp.Pool(cpu_count() - 4)`` = 12 concurrent JPype JVMs on this host.
Measurements and citations: ``.

Why a budget at all, given memory is not the constraint
-------------------------------------------------------
Measured on the 16-vCPU / 60 GB host: one OPSIN JVM peaks at ~200-400 MB RSS
(the 15.8 GB ``MaxHeapSize`` default is *reserved address space*, not committed),
and 32 concurrent JVMs summed to 4.7 GB. Memory would allow 100+.

**CPU is the binding constraint, and oversubscription is a pure loss.** JVM
startup is 0.938 s against a marginal 0.92 ms per name -- one launch costs about
1,020 names of real work -- so splitting a fixed workload across JVMs re-pays
startup per JVM and the startups contend. Throughput falls monotonically:
712 names/s at c=1 down to 166 at c=24. The budget exists to stop unrelated jobs
from driving each other down that curve, not to prevent a correctness failure.

Why ``flock`` rather than a PID file or ``pgrep``
------------------------------------------------
The kernel releases an ``flock`` when the holding process dies, however it dies
-- including ``SIGKILL``. So a slot **cannot** go stale, which is exactly the
failure mode a PID file or sentinel has. ``pgrep`` additionally cannot express
"how many", and it self-matches (a documented hazard here: an unescaped pattern
matches the checking command itself).

This budget is **advisory and fail-open by design**. It exists to schedule work,
never to lose it: if the lock directory is unusable the call proceeds with a
warning rather than failing a caller's run. Set ``ORTHONYM_JVM_BUDGET=off`` to
disable entirely.

Usage
-----
    from orthonym.jvm_budget import jvm_slots

    # a gate: in-process, one JVM
    with jvm_slots(1, purpose="v22_gate"):
        run_conformance(...)

    # an eval harness: one JVM per pool worker
    with jvm_slots(jobs, purpose="harness:dev500"):
        pool.map(...)

    # see what holds the budget right now (replaces `pgrep -f v22_gate`)
    from orthonym.jvm_budget import status
    for slot in status()["held"]:
        print(slot["purpose"], slot["pid"])
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

#: Slots reserved for the interactive shell, RDKit, and the coordinating Python
#: process itself. `:454`` independently arrived at the same
#: ``cpu_count() - 4``; keeping the identical formula means wiring the budget in
#: does not change that tool's existing default degree of parallelism.
_RESERVED_CPUS = 4

_ENV_TOTAL = "ORTHONYM_JVM_SLOTS"
_ENV_DIR = "ORTHONYM_JVM_SLOT_DIR"
_ENV_OFF = "ORTHONYM_JVM_BUDGET"


class BudgetTimeout(RuntimeError):
    """Raised by:func:`jvm_slots` when ``on_timeout='raise'`` and the wait expired."""


def is_enabled() -> bool:
    return os.environ.get(_ENV_OFF, "").strip().lower() not in ("off", "0", "false", "no")


def total_slots() -> int:
    """Total concurrent JVM slots for this machine."""
    raw = os.environ.get(_ENV_TOTAL, "").strip()
    if raw:
        try:
            n = int(raw)
            if n > 0:
                return n
            logger.warning("%s=%r is not positive; ignoring", _ENV_TOTAL, raw)
        except ValueError:
            logger.warning("%s=%r is not an integer; ignoring", _ENV_TOTAL, raw)
    return max(1, (os.cpu_count() or 4) - _RESERVED_CPUS)


def slot_dir() -> Path:
    """Directory holding the lock files. Shared across sessions and worktrees.

    Deliberately NOT inside the repo: the budget is a property of the machine,
    and two worktrees or two sessions of the same checkout must contend over the
    same slots.
    """
    override = os.environ.get(_ENV_DIR, "").strip()
    base = Path(override) if override else Path(tempfile.gettempdir()) / "orthonym-jvm-slots"
    return base


def _ensure_slot_dir() -> Optional[Path]:
    d = slot_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        return d
    except OSError as exc:
        logger.warning("JVM budget disabled: cannot use slot dir %s (%s)", d, exc)
        return None


class _Held:
    """Open file descriptors for the acquired slots. Closing releases the locks."""

    def __init__(self, fds: List[int], purpose: str, count: int, waited_s: float):
        self._fds = fds
        self.purpose = purpose
        self.count = count
        self.waited_s = waited_s
        self.enforced = bool(fds)

    def release(self) -> None:
        for fd in self._fds:
            # Closing the descriptor releases the flock. Truncate first so a
            # later status() does not report a dead holder's metadata.
            with contextlib.suppress(OSError):
                os.ftruncate(fd, 0)
            with contextlib.suppress(OSError):
                os.close(fd)
        self._fds = []


def _try_take(path: Path, purpose: str) -> Optional[int]:
    """Non-blocking exclusive flock on one slot file. Returns an fd, or None."""
    try:
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        logger.debug("slot %s unopenable: %s", path, exc)
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            logger.debug("slot %s flock error: %s", path, exc)
        return None
    # Record who holds it, for status() / debugging. Best-effort only.
    with contextlib.suppress(OSError):
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({
            "pid": os.getpid(),
            "purpose": purpose,
            "argv0": (sys.argv[0] if sys.argv else ""),
            "since": time.time(),
        }).encode() + b"\n")
        os.fsync(fd)
    return fd


def _acquire(count: int, purpose: str, timeout: Optional[float]) -> _Held:
    """Take ``count`` slots, releasing partial holdings between attempts.

    Releasing on partial acquisition is what prevents two callers from each
    pinning half the budget forever; the jittered backoff keeps them from
    re-colliding in lockstep.
    """
    t0 = time.monotonic()
    d = _ensure_slot_dir()
    if d is None:
        return _Held([], purpose, count, 0.0)

    total = total_slots()
    if count > total:
        logger.warning(
            "JVM budget: %r asked for %d slots but the machine budget is %d; "
            "capping the reservation (the work still runs, oversubscribed)",
            purpose, count, total)
        count = total

    paths = [d / f"slot-{i:03d}" for i in range(total)]
    rng = random.Random(os.getpid())
    announced = False

    while True:
        fds: List[int] = []
        for p in paths:
            fd = _try_take(p, purpose)
            if fd is not None:
                fds.append(fd)
                if len(fds) == count:
                    return _Held(fds, purpose, count, time.monotonic() - t0)
        # Not enough free slots -- drop everything so we do not deadlock a peer.
        for fd in fds:
            with contextlib.suppress(OSError):
                os.ftruncate(fd, 0)
            with contextlib.suppress(OSError):
                os.close(fd)

        waited = time.monotonic() - t0
        if timeout is not None and waited >= timeout:
            return _Held([], purpose, count, waited)
        if not announced:
            logger.info("JVM budget: %r waiting for %d/%d slots (held by other jobs)",
                        purpose, count, total)
            announced = True
        remaining = None if timeout is None else max(0.0, timeout - waited)
        nap = rng.uniform(0.25, 1.5)
        time.sleep(nap if remaining is None else min(nap, remaining))


@contextlib.contextmanager
def jvm_slots(count: int = 1, *, purpose: str = "unnamed",
              timeout: Optional[float] = None,
              on_timeout: str = "proceed") -> Iterator[_Held]:
    """Reserve ``count`` concurrent-JVM slots for the duration of the block.

    Args:
        count: JVMs this job will have alive at once. One in-process JPype JVM
            or one ``java -jar`` subprocess is 1; an ``mp.Pool(n)`` whose workers
            each start a JVM is ``n``.
        purpose: short label recorded in the slot file and shown by
            :func:`status` -- e.g. ``"v22_gate"``, ``"harness:dev500"``.
        timeout: seconds to wait. ``None`` waits indefinitely.
        on_timeout: ``"proceed"`` (default) runs anyway with a warning --
            correct for a long job a caller must not lose; ``"raise"`` raises
            :class:`BudgetTimeout`, for callers that would rather refuse.

    Never raises on a missing/unwritable lock directory: the budget is advisory
    and must not be able to fail a naming run.
    """
    if count < 1:
        count = 1
    if not is_enabled():
        yield _Held([], purpose, count, 0.0)
        return

    held = _acquire(count, purpose, timeout)
    if not held.enforced:
        if timeout is not None and on_timeout == "raise":
            raise BudgetTimeout(
                f"{purpose!r} could not reserve {count} of {total_slots()} JVM slots "
                f"within {timeout:.0f}s")
        logger.warning(
            "JVM budget: %r proceeding WITHOUT a reservation (%d slots wanted). "
            "Throughput may degrade; correctness is unaffected.", purpose, count)
    elif held.waited_s > 1.0:
        logger.info("JVM budget: %r acquired %d slots after %.1fs",
                    purpose, count, held.waited_s)
    try:
        yield held
    finally:
        held.release()


def status() -> dict:
    """Report which slots are currently held, and by what.

    The supported replacement for ``pgrep -f "\\.py"``: it counts,
    it names the holder, and it cannot self-match the checking command.
    """
    total = total_slots()
    d = slot_dir()
    held, free = [], 0
    for i in range(total):
        p = d / f"slot-{i:03d}"
        if not p.exists():
            free += 1
            continue
        try:
            fd = os.open(str(p), os.O_RDWR)
        except OSError:
            free += 1
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Locked by someone else -> read their metadata via a separate handle.
            info = {"slot": i, "pid": None, "purpose": "<unknown>"}
            with contextlib.suppress(Exception):
                rec = json.loads(p.read_text().strip() or "{}")
                info.update({k: rec.get(k) for k in ("pid", "purpose", "argv0", "since")})
                if rec.get("since"):
                    info["age_s"] = round(time.time() - float(rec["since"]), 1)
            held.append(info)
            os.close(fd)
            continue
        # We got it, so it was free. Release immediately.
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        free += 1
    return {
        "enabled": is_enabled(),
        "total": total,
        "free": free,
        "held": held,
        "slot_dir": str(d),
    }


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m orthonym.jvm_budget`` -- show the current budget."""
    st = status()
    print(f"JVM budget: {st['total'] - st['free']}/{st['total']} slots held "
          f"({'enabled' if st['enabled'] else 'DISABLED'}) dir={st['slot_dir']}")
    for h in st["held"]:
        age = f"{h.get('age_s')}s" if h.get("age_s") is not None else "?"
        print(f" slot {h['slot']:>3} pid={h.get('pid')} {h.get('purpose')} age={age}")
    if not st["held"]:
        print(" (all slots free -- no OPSIN/JVM job is running)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
