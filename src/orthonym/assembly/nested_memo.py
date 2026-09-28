"""Replay-memo for NESTED naming calls (a performance pass, a lever).

A nested ``name_compound(...)`` inside an outer ``name`` has three side effects the
outer molecule's tier and abstention decisions read: (1) the provenance ContextVars
(``clear_provenance`` runs only in ``name_tiered``, and ``general_ring_prefix`` /
``suffix_free_prefix_name`` / ``stereo_unexpressed`` are set-only), (2) the shared
perf/analysis/work budgets (``_budget_scope`` shares them across nested ``name``),
(3) the return value. A plain memo hit would skip (1) and (2) and could flip the outer
tier or move a PerfBudgetExceeded trip.

This helper reproduces (1) EXACTLY via a provenance TOUCHED-log (a performance pass):
it records which vars the call WROTE — not merely which net-CHANGED — so a var written
to a value it already held, or written A->B->A, is still replayed to the value the fresh
call left (a before/after delta would record it as untouched and leave a stale value on a
later hit in a different ambient context). It records (2) the units each budget lost and
replays them. Scope-bound through ``assembly.memo`` (fail-open outside a scope; verify
mode recomputes).
"""
from typing import Any, Callable, Tuple

from . import fragment_naming as _fn
from .memo import cache_or_compute, side_get, side_put
from ..metrics import provenance as _pv

# The keys here MUST equal the keys ``_pv.get_provenance`` returns (verified at
# metrics/provenance.py:299) — the delta loop does ``_VARS[k].set(v)`` for every
# key the fresh call changed, so a key ``get_provenance`` can return that is absent
# here would KeyError on replay.
_VARS = {
    "source": _pv._SOURCE, "opsin": _pv._OPSIN, "stereo_unexpressed": _pv._STEREO_UNEXPRESSED,
    "gate_outcome": _pv._GATE_OUTCOME, "gate_outcome_name": _pv._GATE_OUTCOME_NAME,
    "general_ring_prefix": _pv._GENERAL_RING_PREFIX, "suffix_free_prefix_name": _pv._SUFFIX_FREE_PREFIX_NAME,
    "non_pin_fragments": _pv._NON_PIN_FRAGMENTS,
}
# Accumulators: a hit MERGES the fresh call's recorded entries into the current
# value instead of overwriting it -- an overwrite would drop entries recorded
# after the fresh call (in this ambient context) that the snapshot never saw.
_ACCUMULATORS = frozenset({"non_pin_fragments"})
_BUDGETS = ("perf_budget", "analysis_budget", "work_budget")


def _budgets() -> Tuple[Any, ...]:
    return tuple(getattr(_fn._fragment_guard, b, None) for b in _BUDGETS)


def _replay_budgets(units: Tuple[int, int, int]) -> None:
    perf, analysis, work = units
    # perf/analysis raise PerfBudgetExceeded when the (possibly lower) outer budget
    # can no longer absorb the recorded charge — that correctly reproduces the fresh
    # call's abstention at the current budget state, and per the brief propagates
    # uncached.
    if perf:
        _fn.spend_perf_work(perf)
    if analysis:
        _fn.spend_analysis_call(analysis)
    # spend_fragment_work takes no argument and decrements work_budget by 1 per call
    # (fragment_naming.py:507); it returns a bool and never raises, so replay the exact
    # count and ignore the return value — only the decrement trajectory must match.
    for _ in range(work):
        _fn.spend_fragment_work()


def _apply(replay: dict, units: tuple, replay_budgets: bool = True) -> None:
    """Reproduce a fresh call's provenance (+ budget, unless ``replay_budgets`` is
    False) side effects on a memo hit."""
    for k, v in replay.items():
        if k in _ACCUMULATORS:
            cur = _VARS[k].get()
            _VARS[k].set(cur + tuple(x for x in (v or ()) if x not in cur))
            # An enclosing memoised substituent-fragment naming logs the fragments
            # this hit merges, as it logs every record_non_pin_fragment call.
            for x in (v or ()):
                _pv._log_non_pin(x)
        else:
            _VARS[k].set(v)
    # These direct.set writes bypass the provenance setters, so an ENCLOSING
    # touched-log would miss them; note them explicitly so an outer memoised call
    # still records that this nested hit wrote these vars.
    _pv.note_touched(*replay.keys())
    if replay_budgets:
        _replay_budgets(units)


def cached_nested_call(namespace: str, key: tuple, fn: Callable[[], Any], *,
                       replay_budgets: bool = True) -> Any:
    """Return ``fn``'s result for ``(namespace, key)``; on a hit replay the EXACT
    provenance vars the call touched and its budget units. Exceptions propagate uncached.

    Only the RESULT is stored under ``namespace`` (so ``ORTHONYM_MEMO=verify`` compares
    names — an honest incomplete-key oracle); the context-relative replay metadata lives
    in the scope-bound side store, which verify never compares.

    ``replay_budgets=False``: a hit replays the provenance only and charges no budget
    unit. The recorded units are the fresh call's COLD cost; for a memo placed over a
    whole-pipeline sub-namer (the peptide substitutive attempts) the unmemoised repeat
    that a hit stands for runs WARM (the scope's fragment cache and plain memos already
    hold its sub-results) and costs almost nothing, so replaying the cold cost on every
    hit charges the budget for work that neither the memoised nor the unmemoised code
    does. Measured (ChEBI 12-residue peptides, fresh process): the whole-molecule attempt
    cost 145 analysis calls cold and 0, 0, 0 on its three unmemoised repeats; the replay
    charged 4 x 145 and more against the 500-call budget, which then aborted the naming
    (PerfBudgetExceeded -> abstain). A hit does no work, so it spends none; every fresh
    computation still charges its own work, so the hang budgets still bound the call."""
    state = {"hit": True, "meta": None}

    def _fresh():
        state["hit"] = False
        before_b = _budgets()
        _pv.push_touched_log()
        try:
            result = fn()
        finally:
            touched = _pv.pop_touched_log()
        after_prov = _pv.get_provenance(); after_b = _budgets()
        # EXACT replay: the value the fresh call LEFT for every var it wrote — not
        # a net-change delta (which drops an A->B->A round-trip or a same-value write).
        replay = {k: after_prov[k] for k in touched}
        units = tuple((b - a) if (b is not None and a is not None and b > a) else 0
                      for b, a in zip(before_b, after_b))
        state["meta"] = (replay, units)
        return result

    result = cache_or_compute(namespace, key, _fresh)
    if state["hit"]:
        # A true hit: fn did not run — replay the metadata recorded on the fresh miss.
        meta = side_get(namespace, key)
        if meta is not None:
            _apply(meta[0], meta[1], replay_budgets)
    else:
        # A fresh computation (miss, or a verify-mode recompute): fn's real side
        # effects already happened; persist the metadata for a future hit.
        side_put(namespace, key, state["meta"])
    return result
