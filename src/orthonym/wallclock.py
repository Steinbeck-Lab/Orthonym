"""A wall-clock limit on one call, for the tools that name many molecules.

The engine bounds its own work with deterministic operation budgets
(``assembly.fragment_naming.PerfBudgetExceeded``) and never reads a clock; nothing
under ``src/orthonym`` arms an alarm. The measurement tools (``eval/harness.py``,
``scripts/pin_conformance_eval.py``, the naming-scale shard workers,...) add a
per-molecule wall-clock limit on top, so that one slow molecule costs one row and
not the run.

That limit must be a ``BaseException``. The naming path holds hundreds of
``except Exception`` handlers. An ``Exception`` raised from a ``SIGALRM`` handler is
caught by whichever of them is active when the alarm fires, and naming carries on
from a half-finished choice: ``VonBaeyerAnalyzer._choose_lowest_locant_numbering``
keeps its incumbent numbering when its candidate search is cut off, and a molecule
that abstains in every unloaded run was named in a loaded one. The outcome of a slow
molecule then depends on the load of the machine, and the tool's TIMEOUT row is never
written.:class:`WallClockTimeout` passes every ``except Exception`` on its way out
(like ``PerfBudgetExceeded``; the package has no bare ``except:`` and no
``except BaseException``), and the tool catches it by name.

Use::

    from orthonym.wallclock import WallClockTimeout, clear_leaked_naming_state, wall_clock_limit

    try:
        with wall_clock_limit(30):
            res = namer.name_tiered(smiles)
    except WallClockTimeout:
        leaked = clear_leaked_naming_state # empty: the limit put the state back
        row = {"tier": "TIMEOUT",...}

Put the ``with`` inside the ``try``: the alarm is then disarmed before the
``except`` branch runs, so it can never fire inside the tool's own bookkeeping.
"""
from __future__ import annotations

import contextvars
import signal
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

__all__ = [
    "WallClockTimeout",
    "wall_clock_limit",
    "can_limit",
    "leaked_naming_state",
    "reset_naming_state",
    "clear_leaked_naming_state",
    "NAMING_TIER_CTX",
    "NAMING_REQUEST_CTX",
    "NAMING_DEPTHS",
]

#: After the first alarm, the limit fires again at this interval until the body
#: has left the ``with``. The first ``WallClockTimeout`` can be lost: an alarm that
#: lands in a garbage-collector callback (a ``WeakKeyDictionary`` entry removal, a
#: ``__del__``) is printed as "Exception ignored" and dropped. Seen in this
#: module's own tests.
REFIRE_INTERVAL_S = 1.0


class WallClockTimeout(BaseException):
    """A tool's wall-clock limit on one call was exceeded.

    Derives from ``BaseException`` (NOT ``Exception``) so that no ``except
    Exception`` on the naming path can catch it and let the cascade continue from a
    half-finished choice. The tool that armed the limit catches it by name.

    ``restored`` names the process-global naming state the interrupted call had
    changed and the limit put back (see:class:`wall_clock_limit`)."""

    def __init__(self, seconds: Optional[float] = None):
        self.seconds = seconds
        self.restored: List[str] = []
        super().__init__(
            f"wall-clock limit of {seconds} s exceeded" if seconds is not None
            else "wall-clock limit exceeded")


def can_limit() -> bool:
    """True when this thread can arm a limit: a POSIX ``SIGALRM`` on the main thread
    (Python runs signal handlers on the main thread only)."""
    return (hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
            and threading.current_thread() is threading.main_thread())


class wall_clock_limit:  # noqa: N801 - used as a function-style context manager
    """``with wall_clock_limit(seconds):`` raises:class:`WallClockTimeout` in the
    body once ``seconds`` of wall-clock time have passed.

    * Arms ``ITIMER_REAL`` with a ``SIGALRM`` handler that raises the timeout, and
      fires again every:data:`REFIRE_INTERVAL_S` until the body has left.
    * If the alarm fired, the ``with`` statement raises ``WallClockTimeout``, even
      when the body went on to return normally or to raise something else (the first
      timeout was dropped or converted on the way): a limit that was exceeded always
      reads as a timeout.
    * On every exit the timer is disarmed and the previous ``SIGALRM`` handler is put
      back. A limit armed by an enclosing caller is re-armed with what is left of it,
      so limits nest.
    * On a timeout, the orthonym ContextVars and module-level thread-locals the
      interrupted call changed are put back to their values at entry (the timeout's
      ``restored`` lists them). The engine resets this state in ``finally`` blocks,
      but an alarm that lands inside such a block, or between a ``set`` and the
      ``try`` that guards it, skips the reset (seen: an alarm in ``name``'s
      ``finally`` left ``session_depth`` at 1), and every later naming in the
      process would run with it. State of a module first imported during the call
      is left as it is.

    ``seconds`` of ``None`` or ``<= 0`` means no limit. Off the main thread, or on a
    platform without ``SIGALRM``, the body runs without a limit (a signal cannot be
    delivered there), as the tools did before this helper.
    """

    def __init__(self, seconds: Optional[float]):
        self.seconds = seconds
        self.fired = False
        self.restored: List[str] = []
        self._active = False
        self._live = False

    def __enter__(self) -> "wall_clock_limit":
        if self.seconds is None or self.seconds <= 0 or not can_limit():
            return self
        self._snapshot = _snapshot_naming_state()
        self.fired = False
        self._live = True
        self._previous_handler = signal.signal(signal.SIGALRM, self._on_alarm)
        self._started = time.monotonic()
        self._previous_timer = signal.setitimer(
            signal.ITIMER_REAL, self.seconds, REFIRE_INTERVAL_S)
        self._active = True
        return self

    def _on_alarm(self, signum, frame):  # noqa: ARG002
        if not self._live:
            return
        self.fired = True
        raise WallClockTimeout(self.seconds)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self._active:
            return False
        self._active = False
        try:
            self._live = False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._previous_handler)
            previous_delay, previous_interval = self._previous_timer
            if previous_delay > 0:
                # The enclosing limit keeps its own deadline; if it has passed
                # already, it fires at once.
                remaining = previous_delay - (time.monotonic() - self._started)
                signal.setitimer(signal.ITIMER_REAL, max(remaining, 1e-6),
                                 previous_interval)
        if not self.fired:
            return False
        self.restored = _restore_naming_state(self._snapshot)
        if isinstance(exc, WallClockTimeout):
            exc.restored = self.restored
            return False
        timeout = WallClockTimeout(self.seconds)
        timeout.restored = self.restored
        raise timeout from exc


# ---------------------------------------------------------------------------
# Process-global naming state: put back after a timeout
# ---------------------------------------------------------------------------
_UNSET = object()
_REGISTRY: Dict[str, object] = {"n_modules": -1, "vars": (), "locals": ()}


def _naming_state_objects():
    """The ContextVars and ``threading.local`` objects at module level in the loaded
    orthonym modules, as ``((qualified name, object),...)`` pairs; rebuilt when the
    set of loaded modules has grown."""
    n_modules = len(sys.modules)
    if n_modules != _REGISTRY["n_modules"]:
        cvars, tlocals, seen = [], [], set()
        for mod_name, mod in list(sys.modules.items()):
            if mod is None or not (mod_name == "orthonym" or mod_name.startswith("orthonym.")):
                continue
            try:
                items = list(vars(mod).items())
            except TypeError:
                continue
            for attr, obj in items:
                if id(obj) in seen:
                    continue
                if isinstance(obj, contextvars.ContextVar):
                    seen.add(id(obj))
                    cvars.append((f"{mod_name}.{attr}", obj))
                elif isinstance(obj, threading.local):
                    seen.add(id(obj))
                    tlocals.append((f"{mod_name}.{attr}", obj))
        _REGISTRY.update(n_modules=n_modules, vars=tuple(cvars), locals=tuple(tlocals))
    return _REGISTRY["vars"], _REGISTRY["locals"]


def _copy(value):
    return value.copy() if isinstance(value, (list, set, dict)) else value


def _same(a, b) -> bool:
    if a is b:
        return True
    try:
        return bool(type(a) is type(b) and a == b)
    except Exception:  # noqa: BLE001 - an object whose == fails is "changed"
        return False


def _snapshot_naming_state():
    cvars, tlocals = _naming_state_objects()
    var_values = {}
    for _, var in cvars:
        try:
            var_values[id(var)] = var.get()
        except LookupError:
            var_values[id(var)] = _UNSET
    local_values = {id(tl): {k: _copy(v) for k, v in vars(tl).items()}
                    for _, tl in tlocals}
    return var_values, local_values


def _restore_naming_state(snapshot) -> List[str]:
    var_values, local_values = snapshot
    cvars, tlocals = _naming_state_objects()
    restored = []
    for qualname, var in cvars:
        before = var_values.get(id(var), _UNSET)
        if before is _UNSET:
            continue  # first imported during the call, or never set and no default
        try:
            now = var.get()
        except LookupError:
            now = _UNSET
        if not _same(now, before):
            var.set(before)
            restored.append(qualname)
    for qualname, tl in tlocals:
        before = local_values.get(id(tl))
        if before is None:
            continue
        now = vars(tl)
        if set(now) != set(before) or not all(_same(now[k], before[k]) for k in before):
            now.clear()
            now.update({k: _copy(v) for k, v in before.items()})
            restored.append(qualname)
    return restored


# ---------------------------------------------------------------------------
# The known naming state: what a tool checks after a timeout
# ---------------------------------------------------------------------------
# ``Orthonym.name`` publishes four tier ContextVars and the request tier for the
# length of a top-level call, and the fragment recursion keeps thread-local depth
# counters and a visited set; the test suite's isolation guard checks exactly these
# (``tests/conftest.py``, ``_leaked_naming_state``; a unit test keeps the tuples
# equal). The tools also check the rest of the scaffolding ``name`` sets up and
# tears down (the memo scope, the armed budgets, the lone-pair input scope, the
# caller's input) and the re-entry guards of the producers. Each value below is the
# one it has outside a naming call.
NAMING_TIER_CTX = ("general_fallback_ctx", "best_effort_ctx",
                   "allow_aromatic_general_ctx", "full_coverage_ctx")
NAMING_REQUEST_CTX = ("best_effort_request_ctx",)
NAMING_DEPTHS = ("session_depth", "name_call_depth")

_IDLE_CONTEXTVARS: Tuple[Tuple[str, str, object], ...] = (
    ("orthonym.assembly.memo", "_cache_var", None),
    ("orthonym.assembly.memo", "pin_promotion_var", False),
    ("orthonym.assembly.memo", "prefixes_apart_var", False),
)
_IDLE_THREAD_LOCALS: Tuple[Tuple[str, str, Dict[str, object]], ...] = (
    ("orthonym.assembly.fragment_naming", "_fragment_guard",
     {"work_budget": None, "perf_budget": None, "analysis_budget": None,
      "cache": None, "naming_pass_cap": None}),
    ("orthonym.namer", "_LP_INPUT", {"open": False, "rewrite": None}),
    ("orthonym.namer", "_CALLER_INPUT", {"smiles": None}),
    ("orthonym.namer", "_ALT_PARENT_RESCUE", {"active": False}),
    ("orthonym.assembly.substituent_enumerator", "_GATE_REENTRY", {"active": False}),
    ("orthonym.assembly.group_splitting", "_IN_SPLIT_PROBE", {"active": False}),
    ("orthonym.rules.charged_router", "_route_reentry", {"depth": 0}),
)


def leaked_naming_state() -> Dict[str, object]:
    """The known process-global naming state of the calling thread that is not at
    its value outside a naming call (empty when clean). Reads only modules already
    imported; imports nothing."""
    leaked: Dict[str, object] = {}
    pv = sys.modules.get("orthonym.metrics.provenance")
    if pv is not None:
        for attr in NAMING_TIER_CTX:
            var = getattr(pv, attr, None)
            if var is not None and var.get() not in (False, None):
                leaked[attr] = var.get()
        for attr in NAMING_REQUEST_CTX:
            var = getattr(pv, attr, None)
            if var is not None and var.get() is not None:
                leaked[attr] = var.get()
    fn = sys.modules.get("orthonym.assembly.fragment_naming")
    guard = getattr(fn, "_fragment_guard", None) if fn is not None else None
    if guard is not None:
        for attr in NAMING_DEPTHS:
            if getattr(guard, attr, 0):
                leaked[attr] = getattr(guard, attr)
        if getattr(guard, "visited", None):
            leaked["visited"] = len(guard.visited)
    for mod_name, attr, idle in _IDLE_CONTEXTVARS:
        var = getattr(sys.modules.get(mod_name), attr, None)
        if var is not None and not _same(var.get(), idle):
            value = var.get()
            leaked[attr] = len(value) if isinstance(value, dict) else value
    for mod_name, local_name, idle_attrs in _IDLE_THREAD_LOCALS:
        tl = getattr(sys.modules.get(mod_name), local_name, None)
        if tl is None:
            continue
        for attr, idle in idle_attrs.items():
            value = getattr(tl, attr, idle)
            if not _same(value, idle):
                leaked[f"{local_name}.{attr}"] = (
                    len(value) if isinstance(value, (dict, set, list)) else value)
    return leaked


def reset_naming_state() -> None:
    """Put the state:func:`leaked_naming_state` reads back to its value outside a
    naming call."""
    pv = sys.modules.get("orthonym.metrics.provenance")
    if pv is not None:
        for attr in NAMING_TIER_CTX:
            var = getattr(pv, attr, None)
            if var is not None:
                var.set(False)
        for attr in NAMING_REQUEST_CTX:
            var = getattr(pv, attr, None)
            if var is not None:
                var.set(None)
    fn = sys.modules.get("orthonym.assembly.fragment_naming")
    guard = getattr(fn, "_fragment_guard", None) if fn is not None else None
    if guard is not None:
        for attr in NAMING_DEPTHS:
            setattr(guard, attr, 0)
        guard.visited = set()
    for mod_name, attr, idle in _IDLE_CONTEXTVARS:
        var = getattr(sys.modules.get(mod_name), attr, None)
        if var is not None:
            var.set(idle)
    for mod_name, local_name, idle_attrs in _IDLE_THREAD_LOCALS:
        tl = getattr(sys.modules.get(mod_name), local_name, None)
        if tl is None:
            continue
        for attr, idle in idle_attrs.items():
            if hasattr(tl, attr):
                setattr(tl, attr, idle)


def clear_leaked_naming_state() -> Dict[str, object]:
    """After a:class:`WallClockTimeout`: return the known naming state still left
    behind (empty when clean, which it is once:class:`wall_clock_limit` has put the
    state back) and reset it, so the next molecule in this process is named as in a
    fresh one. A tool records a non-empty result on its TIMEOUT row."""
    leaked = leaked_naming_state()
    if leaked:
        reset_naming_state()
    return leaked
