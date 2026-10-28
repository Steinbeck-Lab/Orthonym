"""``--time-limit`` of the command line: a wall-clock limit on the naming of each structure.

A few slow structures in a large ``--batch`` file hold up the whole run, and a limit of the caller's
own (a thread that waits, a ``signal.alarm`` whose handler raises an ``Exception``) does not stop
naming: the naming path holds hundreds of ``except Exception`` handlers, and a thread cannot be
stopped. ``--time-limit SECONDS`` arms:func:`orthonym.wallclock.wall_clock_limit` around the naming
of each structure. A structure that is not named within the limit is a decline of its own, with the
reason code ``TIME_LIMIT``, and the run goes on with the next line, which is named as in a fresh
process (the limit puts the process-global naming state back, and the batch builds a new
``Orthonym`` after a timeout).

What the limit does not do. The alarm is a ``SIGALRM`` handler, and Python runs a handler only when
the main thread is back in Python. A call that has not returned yet, into the JVM (OPSIN, the
centres labeller) or into RDKit, runs to its end first, so the limit can be exceeded by the length of
the one call that was running when it expired. Measured on 168 real structures at limits of 0.05 to
1 s (840 runs, 268 of them timed out): 1.1 ms over the limit in the middle, 10 ms for 99 in 100, 57 ms
at worst; a JVM call was running at the deadline in 12 of them, for 11 ms at most. The calls are short
(the longest JVM call in a naming of 90 s was 0.42 s, the first of the process, which starts the JVM); a
call into a cold JVM is long, so:func:`prepare` starts the JVM before the first limit is armed, and
``jvm_bridge._alarm_held`` keeps an alarm out of the one call into the JVM that cannot be interrupted
(``startJVM``). OPSIN needs 2 s for a made-up name of 3,600 characters.

A call that lasts much longer is the case the alarm cannot serve, and it was met: a polyphenylene of 37
benzene rings (GitHub issue 121, the two dendrimers of C222H150 and C258H174) sent a ring-only fragment of
222 atoms to the cage analysis of ``vonbaeyer_universal``, which starts by matching the fragment onto its own
canonical re-parse (``Mol.GetSubstructMatch``, a VF2 search that grows with the symmetry of the branches and
refuses the fragment only after it); the match ran for more than ten minutes in one call, with the alarm
of ``--time-limit 850`` pending (both runs were ended by an outer ``timeout 900``). That call is no longer
made for rings that are joined by bonds (``ring_cluster.is_one_ring_cluster``), and the structures are named
in about five seconds. For a run of ONE structure,:func:`on_hard_stop` is the net for the next call of this
kind: the run ends:data:`HARD_STOP_GRACE_S` seconds after the limit with the outcome of any other timeout (a
thread cannot stop the call, and does not need to: the process is over). A ``--batch`` run has no net: the
lines after a call that does not return would wait for it.

Where no alarm can be set (a platform without ``SIGALRM``, a command line run from a thread other
than the main one),:func:`prepare` says so once on the error stream and the run goes on without a
limit.
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import threading
from typing import Callable, Optional

from .errors import OrthonymLimitError
from .wallclock import can_limit, wall_clock_limit

__all__ = ["CODE", "HARD_STOP_GRACE_S", "REFIRE_S", "WARM_UP_SMILES", "add_option", "error", "label", "limit",
           "on_hard_stop", "prepare", "row", "seconds_arg", "timed"]

#: The alarm that was lost is raised again after this long (``wall_clock_limit(refire=...)``; the default of
#: the measurement tools is 1 s). A first alarm that lands in a ``WeakKeyDictionary`` callback is dropped, and
#: the naming would run on for a second more.
REFIRE_S = 0.1

#: A run of one structure that is still in the call it was in when the limit expired this long after it (the alarm
#: cannot be handled until the call returns) ends with the outcome of a timeout (:func:`on_hard_stop`).
HARD_STOP_GRACE_S = 5.0

#: The end of a run of one structure, set by:func:`on_hard_stop`: called by a timer thread, and the process exits with
#: its return value (with ``_HARD_STOP_STATUS`` when it raises, as a closed standard output does).
_HARD_STOP: Optional[Callable[[], int]] = None
_HARD_STOP_STATUS = 0

#: The reason code of a structure that was not named within the limit. It says something about the run,
#: not about the structure, so it is not a key of ``errors.LIMIT_CATALOG`` (as ``INVALID_SMILES`` is not).
CODE = "TIME_LIMIT"

#: A small structure with a stereocentre, named once before the first limit is armed so that the one-time
#: start-up costs (the JVM, OPSIN's parse tables, the centres labeller, the naming modules imported on first
#: use) are not paid under a limit.
WARM_UP_SMILES = "C[C@H](N)C(=O)O"


def seconds_arg(value: str) -> float:
    """``--time-limit`` value: a number of seconds above 0."""
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number of seconds") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number of seconds above 0")
    return seconds


def add_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--time-limit",
        dest="time_limit",
        type=seconds_arg,
        metavar="SECONDS",
        default=None,
        help=(
            "Give each structure at most this many seconds (a number above 0, for example 30). A "
            "structure that is not named in time gets no name, as for any other structure that gets "
            "none (--on-failure), with the reason code TIME_LIMIT; with --provenance its row has the "
            "tier abstain and limit_code TIME_LIMIT. With --batch the run goes on with the next line, "
            "and --verbose counts the lines that ran out of time. Off by default. The limit is a "
            "SIGALRM that stops the naming at the next Python statement, so a call into the JVM or "
            "into RDKit that is already running finishes first (the JVM is started before the first "
            "limit begins, and its calls last milliseconds); for one structure (no --batch), a call that is "
            "still running 5 s after the limit ends the run with the same outcome. Where SIGALRM is not "
            "available (Windows, a thread other than the main one) the run says so once and has no limit."
        ),
    )


def prepare(seconds: Optional[float], style: str, emit_tier: str) -> Optional[float]:
    """The limit this run can enforce: ``seconds``, or None for no limit (not asked for, or no alarm can
    be set here, which is said once on the error stream). When there is a limit, the start-up costs of
    naming are paid first (see:data:`WARM_UP_SMILES`), outside any limit.

    A timeout while the JVM is starting is not harmless: the first structure would be charged the start-up
    (about a second), and the structure that follows would depend on where in the start-up the alarm
    landed."""
    if seconds is None:
        return None
    if not can_limit():
        print("orthonym: note: --time-limit needs SIGALRM on the main thread, which is not available "
              "here; no time limit is applied", file=sys.stderr)
        return None
    from .namer import Orthonym
    from .tiers import emit_tier_flags
    try:
        Orthonym(style=style, **emit_tier_flags(emit_tier)).name_tiered(WARM_UP_SMILES)
    except Exception:  # noqa: BLE001 - a warm-up that fails has only failed to warm up
        pass
    return seconds


def limit(seconds: Optional[float]):
    """The wall-clock limit of ``seconds`` (none when None) as a ``with`` block: raises
    :class:`orthonym.wallclock.WallClockTimeout` when the body runs out of it, with the naming state put back."""
    return wall_clock_limit(seconds, refire=REFIRE_S)


def on_hard_stop(callback: Optional[Callable[[], int]], status: int = 0) -> None:
    """Register (``None`` clears) what ends a run of ONE structure whose call has not returned:data:`HARD_STOP_GRACE_S`
    seconds after its limit: ``callback`` prints the outcome of a timeout and returns the exit status; ``status`` is the
    status of the run when the callback raises (the status of a timeout of that run), so that the process ends whatever
    the callback does."""
    global _HARD_STOP, _HARD_STOP_STATUS
    _HARD_STOP, _HARD_STOP_STATUS = callback, status


@contextlib.contextmanager
def _hard_stop_timer(seconds: Optional[float]):
    """A timer thread that runs the registered end of the run ``seconds`` +:data:`HARD_STOP_GRACE_S` from now, unless
    the body has left by then. The thread runs while the main thread is inside one call that holds no Python (RDKit,
    the JVM); the lock keeps the end and the normal output from both being written."""
    callback, fallback = _HARD_STOP, _HARD_STOP_STATUS
    if callback is None or not seconds:
        yield
        return
    lock, state = threading.Lock(), {"left": False}

    def stop() -> None:
        with lock:
            if state["left"]:
                return
            status = fallback
            try:
                status = callback()
            finally:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                except Exception:  # noqa: BLE001 - a closed output must not keep the process alive
                    pass
                os._exit(status)

    timer = threading.Timer(seconds + HARD_STOP_GRACE_S, stop)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        with lock:
            state["left"] = True
        timer.cancel()


def timed(seconds: Optional[float], fn, *args, **kwargs):
    """``fn(*args, **kwargs)`` under:func:`limit` (and the net of:func:`on_hard_stop`, when a run of one structure set it)."""
    with limit(seconds), _hard_stop_timer(seconds):
        return fn(*args, **kwargs)


def label(seconds: float) -> str:
    """The label of a structure that ran out of time: what ``--on-failure label`` prints in place of a name."""
    return f"no name within the time limit of {seconds:g} s"


def error(smiles: str, seconds: float) -> OrthonymLimitError:
    """The limit that ``--on-failure raise`` raises for a structure that ran out of time."""
    return OrthonymLimitError(CODE, label(seconds), smiles=smiles)


def row(smiles: str, seconds: float) -> dict:
    """The ``--provenance`` row of a structure that ran out of time: the keys of ``Orthonym.name_tiered``,
    tier ``abstain`` (no name), ``limit_code`` ``TIME_LIMIT`` and the final check as ``not_run``."""
    formula = None
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:  # noqa: BLE001 - the formula is a courtesy of the row
        pass
    return {"name": None, "tier": "abstain", "is_pin": False, "source": "abstain", "opsin": "n/a",
            "gates_passed": [], "gate_outcome": "not_run", "formula": formula, "limit_code": CODE,
            "stereo_unexpressed": False, "suffix_free_prefix_name": False, "prefix_order_fallback": False,
            "verified": "unverified", "spelling_failures": [], "producer": None}
