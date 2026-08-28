"""v33 Phase 0 Task L2.1: whole-molecule Offer + rank_offers/select_offer.

A NEW, small, PURE module -- deliberately NOT built on `assembly.candidate_pool`
or wired through `assembly.inner_dispatch`. The L2 SPY (task-L2-brief.md,
invariant 17) proved those are the wrong vehicle: `candidate_pool.best()` ranks
parent SKELETONS by the P-44 seniority criteria and `CandidateName` carries no
`is_pin`/`tier`/coverage field at all, and restructuring `dispatch_inner` would
hit `tier_a_ring.py:540-542`, which scrubs its own rejected candidates from the
shared pool.

This module instead ranks WHOLE finished NAMES (one per naming strategy that
offered a result), consumed additively at `namer.py::_finish`. With exactly one
offer (today, L2) selection is a pure identity; L3 adds a second (systematic
floor) offer and L4 adds RT-gated retry offers -- neither of those layers
changes this module, only what gets appended to the pool before `select_offer`
runs.

Pure by design: imports only `dataclasses`/`typing`. No OPSIN, no rdkit, no
`namer` import -- keeping it import-cycle-free and trivially unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


#: Confidence-band names for the result ``tier`` field (renamed from the
#: T1..T5 codes; single source of truth for the spelling -- `namer.py`
#: imports these rather than re-spelling the literals).
PIN_VERIFIED = "pin_verified"            # was T1
PIN_UNVERIFIED = "pin_unverified"        # was T2 (reserved, unused)
SYSTEMATIC_VERIFIED = "systematic_verified"  # was T3
BEST_EFFORT = "best_effort"              # was T4
ABSTAIN = "abstain"                      # was T5

#: Lower rank sorts first (preferred). Unknown/future tier strings fall back
#: to 9 in `rank_offers` below -- deny-by-default, never crashes, never sorts
#: an unrecognised tier ahead of a known one.
_TIER_RANK = {
    PIN_VERIFIED: 1,
    PIN_UNVERIFIED: 2,
    SYSTEMATIC_VERIFIED: 3,
    BEST_EFFORT: 4,
    ABSTAIN: 5,
}


@dataclass(frozen=True)
class Offer:
    """One candidate finished name a naming strategy is offering as the
    result for the whole molecule.

    ``result_obj`` is the ``GeneralEngineResult`` backing ``name`` when one
    exists (``None`` for a bare-str PIN-path winner) -- carried through so a
    later gate (L4) can re-run OPSIN/E1 checks against the SAME bindings that
    produced ``name``, rather than re-deriving them from the string.
    """
    name: str
    result_obj: Any
    is_pin: bool
    tier: str
    source: str
    complete: bool


def _rank_key(offer: Offer):
    """The deterministic, TOTAL sort key `rank_offers` sorts ascending by.

    ``(0 if complete else 1, 0 if is_pin else 1, tier_rank, source, name)`` --
    a complete winner always beats an incomplete one; among complete offers a
    PIN beats a non-PIN; then lower tier number wins; ``source`` and finally
    ``name`` are pure tiebreaks that make the key TOTAL (no two distinct
    offers with all four other fields equal can tie -- unless `name` itself
    is equal too, which `sorted`'s stability then resolves by original list
    order, matching the plan's `test_deterministic_and_empty` expectation).
    """
    return (
        0 if offer.complete else 1,
        0 if offer.is_pin else 1,
        _TIER_RANK.get(offer.tier, 9),
        offer.source,
        offer.name,
    )


def rank_offers(offers: List[Offer]) -> List[Offer]:
    """Sort ``offers`` ascending by `_rank_key` -- the most-preferred offer
    (complete, PIN, lowest tier, then alphabetically-first source/name) is
    first; an incomplete offer always sorts after every complete one."""
    return sorted(offers, key=_rank_key)


def select_offer(offers: List[Offer]) -> Optional[Offer]:
    """The single most-preferred offer, or ``None`` for an empty pool."""
    ranked = rank_offers(offers)
    return ranked[0] if ranked else None


def select_rt_passing(offers: List[Offer], rt_ok) -> Optional[Offer]:
    """v33 Phase 0 L4-core: the RT/PIN-gate-over-offers SELECTION PRIMITIVE.

    Walks ``rank_offers(offers)`` in order and returns the FIRST offer that is
    both ``.complete`` and passes the caller-injected ``rt_ok(offer) -> bool``
    predicate; ``None`` if no offer qualifies (including an empty pool).

    This is the SAFETY NET every later offer (a systematic floor, a capability
    producer) relies on: an added offer can never win purely on rank -- it
    must also pass whichever OPSIN round-trip / self-consistency check
    ``rt_ok`` encodes. With today's single-offer pool this is a proven
    IDENTITY whenever that offer's own ``rt_ok`` is True (the common case,
    since the primary winner was already gated upstream); the caller's
    contract (not this function's) is to fall back to the name already
    shipping when this returns ``None``, rather than newly abstaining.

    ``rt_ok`` is injected (never imported) so this module stays pure -- no
    OPSIN, no rdkit, no `namer` import, matching the rest of the module.
    """
    for offer in rank_offers(offers):
        if offer.complete and rt_ok(offer):
            return offer
    return None
