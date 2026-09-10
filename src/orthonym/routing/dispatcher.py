"""a phase ClassFirstRouter — dispatch substrate at _name_impl integration site.

Architecture (158-internal notes):
- ``dispatch`` — two-tier predicate evaluation (Tier-1 mol-only first;
  perception ONCE between tiers; Tier-2 features-required last; GENERAL
  catch-all per). For a phase the perception step lives INSIDE the
  GENERAL handler's caller (``_name_impl`` body) per Task 158-02-01 design
  choice (a) — the dispatcher itself never invokes ``_perceive`` directly,
  preserving ``routing/`` as decoupled from ``composer.py`` (boundary).
- ``get_dispatch_stats`` / ``reset_dispatch_stats`` — per-instance
  histogram counter accessor. Per-instance (NOT module-global) per.
- ``_invoke_audit_log`` — ``ORTHONYM_DISPATCH_AUDIT`` env-var-gated
  INFO-level log. Default-OFF preserves stdout-byte-identical canary.

Anti-pattern hygiene (internal notes-CFR.md AP-block):
-: silent fallthrough -> GENERAL @ 99999 with ``lambda *_: True``;
  impossible-state raises RuntimeError in dispatch (defensive).
-: module-global counter -> counter lives on ``self._dispatch_stats``
  per internal notes.
-: routing layer modifies handler output -> dispatcher returns
  ``ClassDispatchResult`` and lets the caller invoke ``result.handler(...)``;
  it does NOT mutate names.
-: feature-flag-controlled CFR routing path -> NO opt-out flag per
  internal notes; CFR is the only routing path post-Phase-158.

Per internal notes honest-fail-on-data: ``dispatch`` raises RuntimeError if
the GENERAL catch-all is unreachable (impossible by construction; defensive).
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Any, Dict, Optional

from rdkit import Chem  # noqa: F401 -- type annotation only

from .dispatch_table import (
    DISPATCH_TABLE,
    ClassDispatchResult,
    StoutClass,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants (internal notes +)
# ---------------------------------------------------------------------------

ORTHONYM_DISPATCH_AUDIT_ENV_VAR: str = "ORTHONYM_DISPATCH_AUDIT"
_DISPATCH_P99_BUDGET_MS: float = 1.0  # hard gate (informational; benchmark in Plan-03)


# ---------------------------------------------------------------------------
# ClassFirstRouter (internal notes + +)
# ---------------------------------------------------------------------------


class ClassFirstRouter:
    """: Class-first dispatcher; replaces the implicit cascade in _name_impl.

    Three responsibilities (158-internal notes ``<domain>``):

    1. ``dispatch`` — walks ``DISPATCH_TABLE`` in priority order; first-match-wins.
       Returns a frozen ``ClassDispatchResult`` whose ``handler`` field the
       caller invokes to produce the name string. The dispatcher does NOT
       call the handler itself per +.
    2. ``get_dispatch_stats`` — per-instance histogram counter accessor.
       Returns a defensive copy so callers cannot mutate internal state.
    3. ``reset_dispatch_stats`` — explicit reset for batch-run
       boundaries.

    Construction: ``ClassFirstRouter`` with no args. Reads
    ``ORTHONYM_DISPATCH_AUDIT`` env var by default for the audit-log gate
    ; the constructor kwarg ``_audit_log`` overrides for testing.
    """

    # Class-level constant: pre-seeded counter buckets (one per StoutClass).
    # Mirrors a phase OpsinGrammar.STAT_KEYS pattern.
    STAT_KEYS: tuple = tuple(c.value for c in StoutClass)

    def __init__(self, *, _audit_log: Optional[bool] = None) -> None:
        #: env-var-gated audit log (default-OFF per byte-identical-stdout)
        if _audit_log is None:
            _audit_log = os.environ.get(ORTHONYM_DISPATCH_AUDIT_ENV_VAR) == "1"
        self._audit_log: bool = _audit_log
        #: per-instance counter (no module-global state per)
        self._dispatch_stats: Counter = Counter()

    # -----------------------------------------------------------------------
    # Public hot-path API (dispatch contract)
    # -----------------------------------------------------------------------

    def dispatch(
        self,
        mol: "Chem.Mol",
        smiles: str,
        canonical_smiles: str,
        features: Optional[Any] = None,
        **kwargs: Any,
    ) -> ClassDispatchResult:
        """: two-tier predicate evaluation; first-match-wins.

        Walks ``DISPATCH_TABLE`` in priority order (sorted-by-priority).
        ``**kwargs`` is forwarded verbatim to predicate calls so callers can
        thread ``_skip_decomposition`` (option (b)) and ``_style``
        (RETAINED_NAME / AMINO_ACID predicates) without bloating the
        ``ClassDispatchEntry`` shape.

        Returns:
            ``ClassDispatchResult``; caller invokes ``result.handler(...)``.

        Raises:
            RuntimeError: if no entry matched (impossible by construction;
                GENERAL ``lambda *_: True`` always matches as the catch-all).
                Defensive raise per honest-fail-on-data.
        """
        #: explicit sort makes priority-ordering invariant (defense
        # against ordering drift).
        for entry in sorted(DISPATCH_TABLE.values(), key=lambda e: e.priority):
            # Predicate evaluation; signature is
            # (mol, smiles, canonical_smiles, features, **kwargs).
            # The kwargs absorb _skip_decomposition + _style threading.
            try:
                matched = entry.predicate(
                    mol, smiles, canonical_smiles, features, **kwargs
                )
            except TypeError:
                # Defensive: if a predicate doesn't accept kwargs (legacy /
                # external test injection), fall back to the kwarg-less call.
                matched = entry.predicate(mol, smiles, canonical_smiles, features)
            if matched:
                self._dispatch_stats[entry.class_id] += 1
                audit_record: Dict[str, Any] = {
                    "class_id": str(entry.class_id),
                    "smiles_canonical": canonical_smiles,
                    "handler_name": entry.handler.__name__,
                    "tier": entry.tier,
                    "iupac_section": entry.iupac_section,
                }
                if self._audit_log:
                    self._invoke_audit_log(audit_record)
                return ClassDispatchResult(
                    class_id=entry.class_id,
                    handler=entry.handler,
                    audit_record=audit_record,
                    tier=entry.tier,
                )
        # Per internal notes: this is unreachable by construction (GENERAL's
        # ``lambda *_: True`` always matches). Defensive raise per
        # honest-fail-on-data discipline.
        raise RuntimeError(
            "CFR dispatch reached impossible state: GENERAL did not match. "
            "DISPATCH_TABLE integrity violation."
        )

    # -----------------------------------------------------------------------
    # Telemetry accessors (internal notes)
    # -----------------------------------------------------------------------

    def get_dispatch_stats(self) -> Dict[StoutClass, int]:
        """: defensive copy of the (StoutClass -> int) histogram."""
        return dict(self._dispatch_stats)

    def reset_dispatch_stats(self) -> None:
        """: explicit reset for batch-run boundaries."""
        self._dispatch_stats.clear()

    # -----------------------------------------------------------------------
    # Audit-log helper (internal notes; private)
    # -----------------------------------------------------------------------

    def _invoke_audit_log(self, audit_record: dict) -> None:
        """: structured INFO log; gated by ``self._audit_log``.

        Default-OFF (env var unset). When enabled, emits one INFO line per
        dispatch with the JSON-serialized audit record. The canary harness
        MUST run with the env var unset so stdout stays byte-identical
        (stdout-byte-identical contract).
        """
        logger.info("CFR dispatch: %s", json.dumps(audit_record))


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (internal notes)
#
# Per internal notes +, per-instance counters are preferred for telemetry;
# these convenience wrappers are for one-off non-stat-tracking callers and
# tests that don't need their own router instance.
# ---------------------------------------------------------------------------


_DEFAULT_ROUTER: Optional[ClassFirstRouter] = None


def dispatch(
    mol: "Chem.Mol",
    smiles: str,
    canonical_smiles: str,
    features: Optional[Any] = None,
    **kwargs: Any,
) -> ClassDispatchResult:
    """ module-level wrapper for callers that don't carry a router instance.

    Uses a lazily-initialized module-default router. Per internal notes +,
    per-instance counters are preferred for telemetry; this convenience
    wrapper is for one-off non-stat-tracking callers.
    """
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        _DEFAULT_ROUTER = ClassFirstRouter()
    return _DEFAULT_ROUTER.dispatch(mol, smiles, canonical_smiles, features, **kwargs)


def get_dispatch_stats() -> Dict[StoutClass, int]:
    """ +: stats from the module-default router."""
    if _DEFAULT_ROUTER is None:
        return {}
    return _DEFAULT_ROUTER.get_dispatch_stats()


def reset_dispatch_stats() -> None:
    """ +: reset module-default router stats."""
    if _DEFAULT_ROUTER is not None:
        _DEFAULT_ROUTER.reset_dispatch_stats()
