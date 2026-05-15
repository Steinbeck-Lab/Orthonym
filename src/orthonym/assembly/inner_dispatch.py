"""Phase 160 inner-dispatch substrate (DECOMP-01 + CONTEXT D-10).

Mirrors Phase 158 outer CFR ``routing/dispatch_table.py`` substrate at the
INNER (post-class-routing) dispatch layer. Per CONTEXT D-10 the inner
table is structurally identical to the outer CFR table:

* ``@dataclass(frozen=True) class InnerDispatchEntry`` with 7 fields
  (handler_id, priority, predicate, handler, iupac_section, description,
  side_effect_inventory).
* ``INNER_DISPATCH_TABLE: OrderedDict[str, InnerDispatchEntry]`` populated
  at module-import time by ``_register_inner(...)`` calls in the per-handler
  atomic commits 02-01..02-29 (Plan-02) + 03-01..03-09 (Plan-03).
* ``_register_inner(...)`` private helper; raises ``RuntimeError`` if called
  after ``_INNER_REGISTRATION_FROZEN`` is set, or if a handler_id /
  priority is already registered.
* ``dispatch_inner(features) -> Optional[InnerDispatchResult]`` first-match-
  wins iteration in priority order.

Plan-02 substrate commit (02-00) ships the table EMPTY; commits 02-01..02-29
each append ONE ``_register_inner(...)`` call (Tier-1 + Tier-1.5 handlers).
Plan-03 ships the catch-all ``general_acyclic`` at priority 99999 (commit
03-09) — per CONTEXT D-08 the catch-all closes the Plan-02 fallthrough gap.

Anti-pattern hygiene:
- AP-160-08 banned: silent fallthrough in inner-dispatch without explicit
  ``general_acyclic`` catch-all entry at priority 99999. Plan-02 fallthrough
  to inline composer.py mid-tier + root branches is the deliberate Plan-02
  bridge — Plan-03 closes the gap.
- AP-160-09 banned: opt-out flag for inner dispatch (no ``_disable_inner_dispatch``
  kwarg or env-var-controlled bypass).
- AP-160-13: module-global mutable state for inner-dispatch_stats — keep
  per-Orthonym-instance counter mirror of Phase 158 D-16; stats helpers
  are stateful but per-instance-isolated.
- AP-160-15 / D-25 hard invariant: ``side_effect_inventory == ()`` for every
  entry; ``_register_inner`` raises ``ValueError`` if a non-empty tuple is
  passed.
- AP-160-10 banned: invent-as-you-go INNER_DISPATCH_TABLE entries not in
  audit § 1. Every ``_register_inner(...)`` call MUST cite the audit row
  it implements.

References:
- 160-AUDIT-DECOMP.md § 1 — inner-dispatch branch enumeration (39 rows;
  source-of-truth for the 29 Plan-02 commits + 10 Plan-03 commits).
- 160-CONTEXT.md D-10 — substrate shape matches Phase 158.
- 160-PATTERNS.md § 3 — analog: routing/dispatch_table.py:130-211, 696-720.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


# Forward-reference NamingResult by string so we can keep the import
# lazy (avoid name_tree -> inner_dispatch cycle at import time).
NamingResultLike = Optional[Any]


@dataclass(frozen=True)
class InnerDispatchEntry:
    """Phase 160 D-10: frozen dataclass row of INNER_DISPATCH_TABLE.

    Mirrors Phase 158 D-05 + D-26 (routing/dispatch_table.ClassDispatchEntry)
    at the INNER dispatch layer. The 7-field schema is locked per CONTEXT
    D-10:

        handler_id            HANDLER_POLICIES key (lowercase snake_case)
        priority              spaced int; lower = earlier; first-match-wins
        predicate             Callable[..., bool]; MolecularFeatures → bool;
                              PURE per D-25 (no mutation of features / mol /
                              module-global state)
        handler               Callable[..., Optional[NamingResult]]; the
                              handler module's name_<handler_id> entry point
        iupac_section         Blue Book P-section cite (audit metadata)
        description           one-line summary for audit logging
        side_effect_inventory MUST be () per D-25 hard invariant; non-empty
                              tuples raise ValueError in _register_inner.

    Per CONTEXT D-25 + AP-160-15: ``side_effect_inventory == ()`` is the
    HARD invariant; the integrity test
    ``test_side_effect_inventory_is_empty`` (Plan-04) asserts emptiness for
    every entry. AP-160-26 explicitly bans any predicate that mutates
    ``MolecularFeatures``, ``mol``, module-global state, or thread-local
    state. Handler invocations MAY call ``pool.add()`` (the SOLE accepted
    shared-state interaction per Phase 145.1 contract); recursive
    ``orthonym.name_compound(...)`` calls are PERMITTED in the N-oxide
    handler (audited per § 2.4 of 160-AUDIT-DECOMP.md).
    """
    handler_id: str
    priority: int
    predicate: Callable[..., bool]
    handler: Callable[..., NamingResultLike]
    iupac_section: str
    description: str
    side_effect_inventory: Tuple[str, ...] = ()


@dataclass(frozen=True)
class InnerDispatchResult:
    """Phase 160 D-10: result of a successful ``dispatch_inner(features)`` call.

    Returned by ``dispatch_inner`` when a predicate matches. The caller
    (composer.py:_assemble_name_impl) invokes ``result.handler(features, ...)``
    to obtain a ``NamingResult`` and then routes through the standard
    pool.add() / _inject_stereo_if_missing pipeline.

    Plan-02 wave: ``dispatch_inner`` returns ``None`` on no-match because
    the Plan-02 INNER_DISPATCH_TABLE has NO catch-all entry — the general_
    acyclic catch-all ships in Plan-03 commit 03-09. When ``dispatch_inner``
    returns None, the caller falls through to the inline mid-tier + root
    branches still present at composer.py:1501-1903.

    The ``audit_record`` field is a dict of audit metadata (handler_id,
    priority, iupac_section) for the Plan-04 ``--dump-tree`` CLI + the
    inner-dispatch stats counter.
    """
    handler_id: str
    handler: Callable[..., NamingResultLike]
    audit_record: Dict[str, str]
    matched_entry: InnerDispatchEntry


# Plan-02 substrate ships INNER_DISPATCH_TABLE EMPTY. Per-handler atomic
# commits 02-01..02-29 (Plan-02) + 03-01..03-09 (Plan-03) each append ONE
# _register_inner(...) call. Total at Plan-03 end: 30 entries (29
# Tier-1/Tier-1.5 + 1 general_acyclic catch-all).
INNER_DISPATCH_TABLE: "OrderedDict[str, InnerDispatchEntry]" = OrderedDict()

# Registration-freezing sentinel per CONTEXT D-10. Plan-04 may toggle this
# to True after Plan-03 commit 03-10 (composer.py thinning) to lock the
# table against runtime modification. Plan-02/03 keeps it False so each
# atomic commit can append.
_INNER_REGISTRATION_FROZEN: bool = False

# Per-instance counter for the inner-dispatch stats helper (mirror of
# Phase 158 routing/dispatcher.py:_dispatch_counter pattern, per CONTEXT
# D-18 + AP-160-13).
_INNER_DISPATCH_STATS: Dict[str, int] = {}


def _register_inner(
    handler_id: str,
    priority: int,
    predicate: Callable[..., bool],
    handler: Callable[..., NamingResultLike],
    iupac_section: str,
    description: str,
    side_effect_inventory: Tuple[str, ...] = (),
) -> None:
    """Phase 160 D-10 / DECOMP-01: register one inner-dispatch entry.

    Module-import-time-only helper; raises ``RuntimeError`` after the
    table is frozen via ``freeze_inner_table()``. Per CONTEXT D-10 +
    AP-160-15 + AP-160-26: every ``side_effect_inventory`` MUST be ``()``;
    non-empty tuples raise ``ValueError`` immediately.

    Raises:
        RuntimeError: if ``_INNER_REGISTRATION_FROZEN`` is True; or if
            the same ``handler_id`` is registered twice; or if the same
            ``priority`` is registered twice (priority uniqueness ensures
            deterministic first-match-wins iteration).
        ValueError: if ``side_effect_inventory != ()`` (D-25 hard invariant
            per AP-160-15).
    """
    global _INNER_REGISTRATION_FROZEN  # noqa: PLW0603 — module-import-time sentinel

    if _INNER_REGISTRATION_FROZEN:
        raise RuntimeError(
            f"INNER_DISPATCH_TABLE is frozen post-import; cannot register "
            f"{handler_id!r}. Per CONTEXT D-10 + AP-160-09: no runtime "
            f"table modification."
        )

    # D-25 hard invariant per AP-160-15: side_effect_inventory MUST be ().
    if side_effect_inventory != ():
        raise ValueError(
            f"Inner-dispatch registration for {handler_id!r} violates D-25 "
            f"hard invariant: side_effect_inventory={side_effect_inventory!r} "
            f"(must be ()). Per CONTEXT D-25 + AP-160-15: handler predicates "
            f"are PURE (read-only); the fix is upstream in the handler, not "
            f"relaxation of this assertion."
        )

    if handler_id in INNER_DISPATCH_TABLE:
        raise RuntimeError(
            f"Duplicate inner-dispatch registration for handler_id "
            f"{handler_id!r}. Per CONTEXT D-03 + AP-160-06: one file per "
            f"HANDLER_POLICIES handler_id."
        )

    if any(e.priority == priority for e in INNER_DISPATCH_TABLE.values()):
        existing = next(
            e.handler_id for e in INNER_DISPATCH_TABLE.values()
            if e.priority == priority
        )
        raise RuntimeError(
            f"Duplicate inner-dispatch priority {priority} for "
            f"{handler_id!r}; already assigned to {existing!r}. Per "
            f"CONTEXT D-10: priorities are spaced and unique."
        )

    INNER_DISPATCH_TABLE[handler_id] = InnerDispatchEntry(
        handler_id=handler_id,
        priority=priority,
        predicate=predicate,
        handler=handler,
        iupac_section=iupac_section,
        description=description,
        side_effect_inventory=side_effect_inventory,
    )


def freeze_inner_table() -> None:
    """Phase 160 D-10: lock INNER_DISPATCH_TABLE against further registration.

    Called by Plan-03 commit 03-10 (composer.py thinning) at the end of
    module import; after this call, ``_register_inner(...)`` raises
    ``RuntimeError`` per AP-160-09 (no opt-out / runtime modification).

    Plan-02 / Plan-03 atomic commits do NOT call this — the table stays
    open across the migration. Plan-04 is the FIRST plan that calls
    ``freeze_inner_table()`` after registering all 30 entries.
    """
    global _INNER_REGISTRATION_FROZEN  # noqa: PLW0603
    _INNER_REGISTRATION_FROZEN = True


def dispatch_inner(features: Any) -> Optional[InnerDispatchResult]:
    """Phase 160 D-10: first-match-wins inner-cascade dispatch.

    Iterates ``INNER_DISPATCH_TABLE`` in priority order (lowest first;
    OrderedDict insertion order matches priority-sorted insertion via
    ``_register_inner``); returns the first match per
    ``entry.predicate(features)``.

    Returns:
        InnerDispatchResult with the matched handler + entry, or ``None``
        on no match.

    Plan-02 wave: returns ``None`` on no-match because Plan-02 does NOT
    register the catch-all. The caller (composer.py:_assemble_name_impl)
    falls through to the inline mid-tier + root branches still present
    at composer.py:1501-1903 in Plan-02. Plan-03 commit 03-09 adds the
    catch-all entry; from that commit on, dispatch_inner ALWAYS returns
    a non-None InnerDispatchResult.

    Per CONTEXT D-27 honest-fail-on-data: defensive ``try/except TypeError``
    around the predicate call (PATTERNS § Error Handling) handles the case
    where a predicate doesn't accept the kwarg-less call shape.
    """
    # Iterate in priority order. OrderedDict preserves insertion order;
    # _register_inner inserts in priority-sorted call order (Plan-02/03
    # atomic commits register in the audit § 3 dependency-graph topology
    # order, which IS the priority order).
    for entry in sorted(
        INNER_DISPATCH_TABLE.values(),
        key=lambda e: e.priority,
    ):
        try:
            matched = entry.predicate(features)
        except TypeError:
            # Defensive per PATTERNS § Error Handling: predicate may have
            # a strict signature (no kwargs); call directly and surface
            # the error explicitly. Per CONTEXT D-27 honest-fail-on-data:
            # we do NOT silently swallow — re-raise on a second TypeError.
            matched = entry.predicate(features)

        if matched:
            # Increment per-instance stats per CONTEXT D-18 + AP-160-13.
            _INNER_DISPATCH_STATS[entry.handler_id] = (
                _INNER_DISPATCH_STATS.get(entry.handler_id, 0) + 1
            )
            return InnerDispatchResult(
                handler_id=entry.handler_id,
                handler=entry.handler,
                audit_record={
                    "handler_id": entry.handler_id,
                    "priority": str(entry.priority),
                    "iupac_section": entry.iupac_section,
                    "description": entry.description,
                },
                matched_entry=entry,
            )

    # No-match per CONTEXT D-08 Plan-02 boundary: return None and let the
    # caller fall through to inline branches. Plan-03 commit 03-09 adds
    # the general_acyclic catch-all at priority 99999; from that commit
    # on, this branch is unreachable in production.
    return None


def get_inner_dispatch_stats() -> Dict[str, int]:
    """Phase 160 D-18 + AP-160-13: read per-handler dispatch counters.

    Returns a COPY of the in-memory counter dict so callers cannot mutate
    the underlying state. Plan-04 wires this into the CLI (--audit-trace
    flag) to surface per-handler hit rates during benchmarking.
    """
    return dict(_INNER_DISPATCH_STATS)


def reset_inner_dispatch_stats() -> None:
    """Phase 160 D-18: clear the per-handler dispatch counters.

    Called by tests + Plan-04 benchmarks to isolate counter state across
    runs. Mirrors Phase 158 ``reset_dispatch_counter()`` per AP-160-13.
    """
    _INNER_DISPATCH_STATS.clear()


# ============================================================================
# Per-handler _register_inner(...) calls (appended per Plan-02 / Plan-03
# atomic commit per 160-AUDIT-DECOMP.md § 3 topological ordering).
#
# Each call MUST cite the audit row it implements; the audit § 1
# enumeration is the LOCKED spec per CONTEXT D-07 + AP-160-10.
# ============================================================================

# --- Plan-02 commit 02-01: oxime (Tier-1 LIFT; composer.py:771-782 inline
#     branch removed; composer.py:1906-2042 body stays per CONTEXT D-24;
#     audit § 1 row 'oxime' + § 2.2 predicate purity proof).
from .handlers.oxime import _is_oxime, name_oxime  # noqa: E402

_register_inner(
    handler_id="oxime",
    priority=100,
    predicate=_is_oxime,
    handler=name_oxime,
    iupac_section="P-66.6",
    description="Oxime functional class naming; e.g. propan-2-one oxime",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-02: hydrazone (Tier-1 LIFT; composer.py:785-794
#     inline branch removed; shared body with oxime at composer.py:1906-2042;
#     audit § 1 row 'hydrazone' + § 2.3 predicate purity proof).
from .handlers.hydrazone import _is_hydrazone, name_hydrazone  # noqa: E402

_register_inner(
    handler_id="hydrazone",
    priority=200,
    predicate=_is_hydrazone,
    handler=name_hydrazone,
    iupac_section="P-66.6",
    description="Hydrazone functional class naming; e.g. propan-2-one hydrazone",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-03: n_oxide (Tier-1 LIFT; composer.py:813-820 inline
#     branch removed; composer.py:2044-2174 body stays; audit § 1 row 'n_oxide'
#     + § 2.4 predicate purity proof (recursive name_compound permitted).
from .handlers.n_oxide import _is_n_oxide, name_n_oxide  # noqa: E402

_register_inner(
    handler_id="n_oxide",
    priority=300,
    predicate=_is_n_oxide,
    handler=name_n_oxide,
    iupac_section="P-62.5",
    description="N-oxide functional class naming; e.g. pyridine 1-oxide",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-04: isocyanate (Tier-1 LIFT; composer.py:828-836
#     inline branch removed; audit § 1 row 'isocyanate' + § 2.5 purity proof).
from .handlers.isocyanate import _is_isocyanate, name_isocyanate  # noqa: E402

_register_inner(
    handler_id="isocyanate",
    priority=400,
    predicate=_is_isocyanate,
    handler=name_isocyanate,
    iupac_section="P-66.5.4.3",
    description="Isocyanate functional class naming; e.g. methyl isocyanate",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-05: isothiocyanate (Tier-1 LIFT; composer.py:842-850
#     inline branch removed; audit § 1 row 'isothiocyanate' + § 2.6 purity).
from .handlers.isothiocyanate import _is_isothiocyanate, name_isothiocyanate  # noqa: E402

_register_inner(
    handler_id="isothiocyanate",
    priority=500,
    predicate=_is_isothiocyanate,
    handler=name_isothiocyanate,
    iupac_section="P-66.5.4.3",
    description="Isothiocyanate functional class naming; e.g. methyl isothiocyanate",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-06: carbamic_acid (Tier-1 LIFT; audit § 1 row + § 2.7).
from .handlers.carbamic_acid import _is_carbamic_acid, name_carbamic_acid  # noqa: E402

_register_inner(
    handler_id="carbamic_acid",
    priority=600,
    predicate=_is_carbamic_acid,
    handler=name_carbamic_acid,
    iupac_section="P-66.5.5",
    description="Carbamic acid retained name with N-substitution",
    side_effect_inventory=(),
)


__all__ = [
    "InnerDispatchEntry",
    "InnerDispatchResult",
    "INNER_DISPATCH_TABLE",
    "_register_inner",
    "freeze_inner_table",
    "dispatch_inner",
    "get_inner_dispatch_stats",
    "reset_inner_dispatch_stats",
]
