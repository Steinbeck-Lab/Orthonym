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
        except TypeError as exc:
            # Per CONTEXT D-27 honest-fail-on-data + CR-01: do NOT silently
            # re-execute or swallow. A TypeError from a predicate is almost
            # always a real bug (e.g., NoneType attribute access), not a
            # signature mismatch. Surface the bug at the PREDICATE source
            # line with context, chaining the original exception via __cause__.
            raise RuntimeError(
                f"dispatch_inner: predicate for handler_id "
                f"{entry.handler_id!r} raised TypeError: {exc}. "
                f"Fix the predicate (CONTEXT D-25: predicates are pure "
                f"read-only; AP-160-26)."
            ) from exc

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

# --- Plan-02 commit 02-07: carbamate (Tier-1 LIFT; audit § 1 row + § 2.8).
from .handlers.carbamate import _is_carbamate, name_carbamate  # noqa: E402

_register_inner(
    handler_id="carbamate",
    priority=700,
    predicate=_is_carbamate,
    handler=name_carbamate,
    iupac_section="P-66.5.5.1",
    description="Carbamate functional class naming",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-08: urea (Tier-1 LIFT; audit § 1 row + § 2.9).
from .handlers.urea import _is_urea, name_urea  # noqa: E402

_register_inner(
    handler_id="urea",
    priority=800,
    predicate=_is_urea,
    handler=name_urea,
    iupac_section="P-66.6",
    description="Urea retained name with N-substitution",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-09: guanidine (Tier-1 LIFT; audit § 1 row + § 2.10).
from .handlers.guanidine import _is_guanidine, name_guanidine  # noqa: E402

_register_inner(
    handler_id="guanidine",
    priority=900,
    predicate=_is_guanidine,
    handler=name_guanidine,
    iupac_section="P-66.6",
    description="Guanidine retained name with N-substitution",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-10: boronic_acid (Tier-1 LIFT; audit § 1 row + § 2.23).
#     Note: source inline branch is at composer.py:1205-1214 (far below the
#     other Tier-1 leaves). Cross-predicate mutex via principal_group string
#     guarantees byte-identical preservation regardless of priority position.
from .handlers.boronic_acid import _is_boronic_acid, name_boronic_acid  # noqa: E402

_register_inner(
    handler_id="boronic_acid",
    priority=1000,
    predicate=_is_boronic_acid,
    handler=name_boronic_acid,
    iupac_section="P-66.6.4",
    description="Boronic acid retained name (Tier B; e.g. methylboronic acid)",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-11: acid_halide (Tier-1.5 SHIM; audit § 1 + § 2.11).
from .handlers.acid_halide import _is_acid_halide, name_acid_halide  # noqa: E402

_register_inner(
    handler_id="acid_halide",
    priority=1100,
    predicate=_is_acid_halide,
    handler=name_acid_halide,
    iupac_section="P-66.5",
    description="Acyl halide functional class naming (ethanoyl chloride etc.)",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-12: anhydride (Tier-1.5 SHIM; audit § 1 + § 2.12).
from .handlers.anhydride import _is_anhydride, name_anhydride  # noqa: E402

_register_inner(
    handler_id="anhydride",
    priority=1200,
    predicate=_is_anhydride,
    handler=name_anhydride,
    iupac_section="P-66.6.3",
    description="Acid anhydride functional class naming (ethanoic anhydride)",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-13: lactone (Tier-1.5 SHIM with coverage gate; audit
#     § 1 + § 2.13). Coverage gate (ring_size > 8 OR total_heavy <= ring_size + 8)
#     preserved verbatim from composer.py:828-834.
from .handlers.lactone import _is_lactone, name_lactone  # noqa: E402

_register_inner(
    handler_id="lactone",
    priority=1300,
    predicate=_is_lactone,
    handler=name_lactone,
    iupac_section="P-66.6.3",
    description="Monocyclic lactone (cyclic ester) naming",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-14: lactam (Tier-1.5 SHIM with coverage gate;
#     parallel to lactone; audit § 1 + § 2.14).
from .handlers.lactam import _is_lactam, name_lactam  # noqa: E402

_register_inner(
    handler_id="lactam",
    priority=1400,
    predicate=_is_lactam,
    handler=name_lactam,
    iupac_section="P-66.6.3",
    description="Monocyclic lactam (cyclic amide) naming",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-15 (renumbered; original plan's 02-15 was
#     polyfunctional, deferred to Plan-03 per deferred-items.md): sulfoxide
#     (Tier-1.5 SHIM; audit § 1 + § 2.19; predicate mutex with ester via
#     principal_group string; safe to extract before ring_ester).
from .handlers.sulfoxide import _is_sulfoxide, name_sulfoxide  # noqa: E402

_register_inner(
    handler_id="sulfoxide",
    priority=1800,
    predicate=_is_sulfoxide,
    handler=name_sulfoxide,
    iupac_section="P-66.5.2.4",
    description="Sulfoxide functional class naming (Tier B; e.g. dimethyl sulfoxide)",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-16: sulfone (Tier-1.5 SHIM; audit § 1 + § 2.20;
#     parallel to sulfoxide via principal_group string mutex).
from .handlers.sulfone import _is_sulfone, name_sulfone  # noqa: E402

_register_inner(
    handler_id="sulfone",
    priority=1900,
    predicate=_is_sulfone,
    handler=name_sulfone,
    iupac_section="P-66.5.2.4",
    description="Sulfone functional class naming (Tier B; e.g. dimethyl sulfone)",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-17: thioether (Tier-1.5 SHIM; audit § 1 + § 2.21;
#     cyclic + fused-heterocycle skip guards mirrored from inline branch).
from .handlers.thioether import _is_thioether, name_thioether  # noqa: E402

_register_inner(
    handler_id="thioether",
    priority=2000,
    predicate=_is_thioether,
    handler=name_thioether,
    iupac_section="P-66.5.2.4",
    description="Sulfide / thioether functional class naming (Tier B)",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-18: phosphine_oxide (Tier-1.5 SHIM; audit § 1 + § 2.22;
#     direct-return; pool.add() + _inject_stereo_if_missing).
from .handlers.phosphine_oxide import (  # noqa: E402
    _is_phosphine_oxide, name_phosphine_oxide,
)

_register_inner(
    handler_id="phosphine_oxide",
    priority=2100,
    predicate=_is_phosphine_oxide,
    handler=name_phosphine_oxide,
    iupac_section="P-68.3",
    description="Phosphine oxide functional class naming (direct-return)",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-19: phosphate_ester (Tier-1.5 SHIM; audit § 1 + § 2.23;
#     direct-return; principal_group in {phosphate_triester/diester/monoester}).
from .handlers.phosphate_ester import (  # noqa: E402
    _is_phosphate_ester, name_phosphate_ester,
)

_register_inner(
    handler_id="phosphate_ester",
    priority=2200,
    predicate=_is_phosphate_ester,
    handler=name_phosphate_ester,
    iupac_section="P-68.3.1",
    description="Phosphate ester (mono / di / triester) naming",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-20: phosphine (Tier-1.5 SHIM; audit § 1 + § 2.24;
#     covers tertiary/secondary/primary phosphine variants with benzene-parent
#     skip guard).
from .handlers.phosphine import _is_phosphine, name_phosphine  # noqa: E402

_register_inner(
    handler_id="phosphine",
    priority=2300,
    predicate=_is_phosphine,
    handler=name_phosphine,
    iupac_section="P-68.3.1.2",
    description="Phosphine (tertiary / secondary / primary) functional naming",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-21: phosphinic_acid (Tier-1.5 SHIM; audit § 1 + § 2.25;
#     direct-return; pool.add() + _inject_stereo_if_missing).
from .handlers.phosphinic_acid import (  # noqa: E402
    _is_phosphinic_acid, name_phosphinic_acid,
)

_register_inner(
    handler_id="phosphinic_acid",
    priority=2400,
    predicate=_is_phosphinic_acid,
    handler=name_phosphinic_acid,
    iupac_section="P-68.3.1.2.2",
    description="Phosphinic acid functional class naming (direct-return)",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-22: ring_assembly (Tier-1.5 SHIM; audit § 1 + § 2.24;
#     direct-return; predicate = ring_assembly_info AND not chain_is_parent;
#     fires BEFORE complex_ring per composer.py:979 dispatch ordering).
from .handlers.ring_assembly import (  # noqa: E402
    _is_ring_assembly, name_ring_assembly,
)

_register_inner(
    handler_id="ring_assembly",
    priority=2500,
    predicate=_is_ring_assembly,
    handler=name_ring_assembly,
    iupac_section="P-28",
    description="Ring assembly / biaryl naming (direct-return)",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-23: polycyclic (Tier-1.5 SHIM with mutex; audit § 1 +
#     § 2.26; predicate gates on (polycyclic_name AND not chain_is_parent AND
#     not _is_complex_ring_system) to mirror inline gate semantics. Cases that
#     fall through complex_ring rejection are captured by the inline fallback
#     block still present at composer.py:1116+ in Plan-02 (closed in Plan-03).
from .handlers.polycyclic import _is_polycyclic, name_polycyclic  # noqa: E402

_register_inner(
    handler_id="polycyclic",
    priority=2600,
    predicate=_is_polycyclic,
    handler=name_polycyclic,
    iupac_section="P-25",
    description="Polycyclic aromatic (retained name) naming (direct-return)",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-24: partial_sat (Tier-1.5 SHIM with same complex_ring
#     mutex pattern as polycyclic; audit § 1 + § 2.27. Predicate gates on
#     (is_cyclic AND not chain_is_parent AND not _is_complex_ring_system);
#     post-complex_ring rejection fallback retained in inline block.
from .handlers.partial_sat import _is_partial_sat, name_partial_sat  # noqa: E402

_register_inner(
    handler_id="partial_sat",
    priority=2700,
    predicate=_is_partial_sat,
    handler=name_partial_sat,
    iupac_section="P-25.3",
    description="Partially saturated carbocycle naming (Tier B; tetralin-shape)",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-25: simple_molecule (LIFT; audit § 1 + § 2.34).
#     Predicate `not principal_chain and not ring_systems` is mutually
#     exclusive with all other Plan-02 handlers (atom-only molecules); safe
#     to fire at any priority. Priority 2800 per audit.
from .handlers.simple_molecule import (  # noqa: E402
    _is_simple_molecule, name_simple_molecule,
)

_register_inner(
    handler_id="simple_molecule",
    priority=2800,
    predicate=_is_simple_molecule,
    handler=name_simple_molecule,
    iupac_section="P-14",
    description="Single-atom / noble-gas / simple-molecule naming",
    side_effect_inventory=(),
)

# --- Plan-02 commit 02-26: ion_dispatch (Phase 160 addition; audit § 1 + § 2.1).
#     Per CONTEXT D-09, ion / salt / zwitterion / radical species use a
#     PRE-POOL inline bypass at composer.py:751-768 — that call site STAYS
#     unchanged. This inner-dispatch entry exists for architectural
#     uniformity (so all HANDLER_POLICIES + Phase 160 additions appear in
#     the table) but is structurally unreachable: the inline bypass at
#     composer.py:751-768 returns BEFORE control reaches dispatch_inner.
#     Priority 50 places ion_dispatch first in priority order per the
#     audit (lowest priority = first iteration).
from .handlers.ion_dispatch import (  # noqa: E402
    _is_ion_dispatch, name_ion_dispatch,
)

_register_inner(
    handler_id="ion_dispatch",
    priority=50,
    predicate=_is_ion_dispatch,
    handler=name_ion_dispatch,
    iupac_section="P-15.6/P-15.7/P-72/P-73/P-74",
    description="Ion / salt / zwitterion / radical pre-pool bypass (D-09)",
    side_effect_inventory=(),
)

# --- Plan-03 commit 03-01: ring_nitrile (Tier-2 mid-tier; audit § 1 + § 3).
#     Predicate: principal_group == 'nitrile' AND is_cyclic AND not chain_is_parent.
#     Body lift: composer.py:4537-4605 (_assemble_ring_nitrile_name, 69 LOC).
#     Inline branch composer.py:1337-1348 REMOVED at this commit.
from .handlers.ring_nitrile import _is_ring_nitrile, name_ring_nitrile  # noqa: E402

_register_inner(
    handler_id="ring_nitrile",
    priority=5100,
    predicate=_is_ring_nitrile,
    handler=name_ring_nitrile,
    iupac_section="P-66.5.1",
    description="Ring-nitrile (cyclic carbonitrile) direct-return handler",
    side_effect_inventory=(),
)

# --- Plan-03 commit 03-02: amide (Tier-2 mid-tier with polyfunctional + Tier-A
#     mutex; audit § 1 + § 3). Predicate:
#       principal_group in {primary_amide, secondary_amide, tertiary_amide}
#       AND pg_count == 1
#       AND NOT is_polyfunctional (mutex with polyfunctional inline branch at
#           composer.py:870, which fires BEFORE amide in inline cascade order)
#       AND (NOT is_cyclic OR chain_is_parent) (Tier-A mutex — same pattern
#           as partial_sat handler in Plan-02 commit 02-24)
#     Body lift: composer.py:4154-4295 (_assemble_amide_name, 142 LOC).
#     Inline branch composer.py:1343-1356 RETAINED as Tier-A-rejection fallback
#     (composer.py thinning in commit 03-10 consolidates this).
from .handlers.amide import _is_amide, name_amide  # noqa: E402

_register_inner(
    handler_id="amide",
    priority=5200,
    predicate=_is_amide,
    handler=name_amide,
    iupac_section="P-66.5.3",
    description="Amide (primary/secondary/tertiary, single-group; non-cyclic non-polyfunctional fast-path)",
    side_effect_inventory=(),
)

# --- Plan-03 commit 03-03: amine (Tier-2 mid-tier with polyfunctional + Tier-A
#     mutex; audit § 1 + § 3). Predicate:
#       principal_group in {secondary_amine, tertiary_amine}
#       AND NOT is_polyfunctional
#       AND (NOT is_cyclic OR chain_is_parent) (Tier-A mutex)
#     Body lift: composer.py:4296-4536 (_assemble_amine_name, 241 LOC).
#     Inline branch composer.py:1374-1386 RETAINED as Tier-A-rejection fallback.
from .handlers.amine import _is_amine, name_amine  # noqa: E402

_register_inner(
    handler_id="amine",
    priority=5300,
    predicate=_is_amine,
    handler=name_amine,
    iupac_section="P-66.6.1",
    description="Amine (secondary/tertiary; non-cyclic non-polyfunctional fast-path)",
    side_effect_inventory=(),
)

# --- Plan-03 commit 03-04: ring_ester (Tier-2 mid-tier direct-return; audit
#     § 1 + § 3). Predicate:
#       principal_group == 'ester'
#       AND exocyclic_esters from rules.esters.detect_exocyclic_esters(mol)
#       AND NOT _is_complex_ring_system(mol)
#     ring_ester fires BEFORE polyfunctional + ester-family + Tier-A in the
#     inline cascade order (composer.py:850), AND BEFORE partial_sat (which
#     can match cyclic-ester molecules and return None, blocking ring_ester).
#     Priority 1450 places ring_ester between lactam (1400) and the reserved
#     polyfunctional slot (1500), preserving inline cascade order.
#     Body lift: composer.py:3204-3455 (_assemble_ring_with_ester_prefixes, 254 LOC).
#     Inline branch composer.py:850-865 REMOVED at this commit.
from .handlers.ring_ester import _is_ring_ester, name_ring_ester  # noqa: E402

_register_inner(
    handler_id="ring_ester",
    priority=1450,
    predicate=_is_ring_ester,
    handler=name_ring_ester,
    iupac_section="P-66.6.3",
    description="Ring-attached ester with acyloxy prefix on ring parent",
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
