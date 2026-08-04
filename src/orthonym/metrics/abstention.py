"""Typed abstention limit-codes (v25 P0 Task 0.1).

Orthonym's fail-closed paths all collapse into one descriptive fallback
string, which is right for the naming contract but blind for measurement:
the v25 coverage program needs to know *which* mechanism abstained so the
recoverable buckets can be counted before any coverage engine is scoped
(the census in ``, Task 0.2).

This module is a per-top-level-naming-session telemetry slot, deliberately
side-effect-only:

* ``record_abstention`` never raises and never touches the name string —
  the instrumentation is proven byte-identical on the naming suites.
* Two-stage precedence. GENERATION-stage codes (``NO_PARENT``,
  ``BRANCH_UNNAMEABLE``) are first-writer-wins: the site closest to the
  root cause records first. POST-GENERATION sites (``COVERAGE_DOWNGRADE``,
  ``GATE_SUPPRESSED``) record via ``record_suppression`` with the candidate
  they rejected in hand: when that candidate is a REAL name (not
  failure-marked), its existence PROVES generation completed, so any
  speculative generation-stage record from an exploratory dead path is
  overridden. When the candidate itself embeds the failure marker (e.g.
  ``'unknownacetic acid'``), the earlier branch record IS the root cause
  and is kept. Post-generation codes never override each other
  (first-wins), so a gate suppressing a downgrade's decomposition fallback
  keeps the ``COVERAGE_DOWNGRADE`` attribution.
* The slot is only meaningful for a FAILED naming: ``abstention_code_for``
  returns None unless the result is the failure sentinel
  (``errors.is_failure_name``), so speculative codes recorded during a
  naming that ultimately succeeds never surface.

The codes complement (do not replace) ``errors.OrthonymLimitError``:
limit codes classify the *molecule shape* post-hoc; abstention codes
classify the *pipeline site* that declined.
"""

import threading
from enum import Enum
from typing import NamedTuple, Optional

from ..errors import is_failure_name
from . import candidate_ledger

__all__ = [
    "AbstentionCode",
    "AbstentionRecord",
    "abstention_code_for",
    "clear_abstention",
    "peek_abstention",
    "record_abstention",
    "record_suppression",
]


class AbstentionCode(str, Enum):
    """Which pipeline mechanism abstained (str-valued for JSON artifacts)."""

    #: The parent structure itself was refused at the top level (e.g. the
    #: G0 UNSUPPORTED_RING_SYSTEM fail-closed refusal).
    NO_PARENT = "NO_PARENT"
    #: A substituent branch / fragment could not be named while the parent
    #: could (the presumptive E2 recursive-namer bucket).
    BRANCH_UNNAMEABLE = "BRANCH_UNNAMEABLE"
    #: A fully generated candidate name was suppressed by a correctness
    #: gate (SELF-01 OPSIN validity, P10 structure-conservation, the
    #: organometallic/oxoacid source vetoes).
    GATE_SUPPRESSED = "GATE_SUPPRESSED"
    #: The >15-HA GENERAL quality/atom-coverage downgrade machinery
    #: rejected the assembled name (the P2.1 re-emission lever bucket).
    COVERAGE_DOWNGRADE = "COVERAGE_DOWNGRADE"
    #: Residual: the naming failed without any instrumented site recording
    #: a more specific code.
    OTHER = "OTHER"


class AbstentionRecord(NamedTuple):
    code: AbstentionCode
    detail: Optional[str]


_slot = threading.local()


def clear_abstention() -> None:
    """Reset the slot. Called at the START of every top-level naming call
    (never by nested/fragment naming, which shares the parent's session)."""
    _slot.record = None


#: Codes recorded DURING name generation (branch/parent failures).
_GENERATION_CODES = frozenset(
    {AbstentionCode.NO_PARENT, AbstentionCode.BRANCH_UNNAMEABLE}
)


def record_abstention(code: AbstentionCode, detail: Optional[str] = None) -> None:
    """Tag the current naming session with an abstention code.

    First-writer-wins: if a code is already recorded for this session the
    call is a no-op, so the earliest (closest-to-root-cause) site keeps the
    attribution. Never raises; never alters naming output.
    """
    try:
        if getattr(_slot, "record", None) is None:
            _slot.record = AbstentionRecord(code, detail)
    except Exception:  # pragma: no cover - telemetry must never break naming
        pass


def record_suppression(code: AbstentionCode, detail: Optional[str] = None,
                       candidate: Optional[str] = None) -> None:
    """Tag from a POST-GENERATION site (downgrade gate / correctness gate)
    that is rejecting ``candidate``.

    When ``candidate`` is a real name (not failure-marked), generation
    provably completed, so a speculative generation-stage record from an
    exploratory dead path is overridden by ``code``. A failure-marked
    candidate (the branch marker got glued into the name) keeps the earlier
    generation-stage attribution. Post-generation records are never
    overridden (first-wins among themselves). Never raises.
    """
    try:
        # v30 PE-1: mirror into the candidate ledger BEFORE the first-writer-wins
        # logic below discards this event. That precedence rule is right for "which
        # site declined" and wrong for "what was thrown away" -- a suppression that
        # loses the race here still destroyed a candidate, and the ledger must see
        # it. Off unless a consumer enabled it.
        if candidate and candidate_ledger.is_enabled():
            # Scope/depth come from candidate_ledger.resolve_scope() (the one
            # place that answers it), so a gate suppressing a nested fragment
            # naming is not scored against the whole input.
            candidate_ledger.record_candidate(
                f"{code}", candidate_ledger.Stage.SUPPRESSED, candidate,
                detail=detail,
            )
        existing = getattr(_slot, "record", None)
        if existing is None:
            _slot.record = AbstentionRecord(code, detail)
        elif (existing.code in _GENERATION_CODES
              and candidate and not is_failure_name(candidate)):
            _slot.record = AbstentionRecord(code, detail)
    except Exception:  # pragma: no cover - telemetry must never break naming
        pass


def peek_abstention() -> Optional[AbstentionRecord]:
    """Raw slot contents (or None). For tests / census detail dumps; most
    consumers want ``abstention_code_for`` instead."""
    return getattr(_slot, "record", None)


def abstention_code_for(result_name: Optional[str]) -> Optional[AbstentionCode]:
    """The abstention code for the naming call that produced ``result_name``.

    Returns None when ``result_name`` is a real name (speculative codes from
    exploratory paths never surface for a successful naming). For a failure
    sentinel, returns the recorded code, defaulting to ``OTHER`` when no
    instrumented site fired.
    """
    if not is_failure_name(result_name):
        return None
    rec = peek_abstention()
    return rec.code if rec is not None else AbstentionCode.OTHER
