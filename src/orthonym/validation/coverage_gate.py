"""Phase 1 Part A: the SINGLE certification gate for a general-engine result,
shared by every best-effort emission lane.

Root-cause of the per-lane drift Fable named
(`` follow-on #1): three lanes ran
``name_general`` and gated its ``GeneralEngineResult`` DIFFERENTLY --
``assembly/t4_coverage.py`` ran E1 + ``verify_spine`` (escalated), while the
inline G1 lane (``namer.py:3900``) and the multifragment/recovery lane
(``namer.py:3161``) ran E1 ONLY, with the ``_stereo_emit_decision`` cardinality
check and the stereo-insensitive BBR-GATE downstream. A name whose bindings
partition the atoms correctly but silently re-fragment a ring (cyclohexane
spelled as two disjoint propyl halves) passes E1 outright and was shippable via
the two unwired lanes. This gate is the ONE place that answers "is this
``GeneralEngineResult`` a faithful spelling of the graph?", so the lanes cannot
drift again -- competition-analysis P2 ("always-on blocking coverage audit on
the default path").

The gate is E1 (the flat atom partition: coverage + disjointness) AND
``verify_spine(mode="audit", escalate=STRICT_STEREO_CHARGE_AXES)`` (bond
totality P2, token spans P4/P5, token arity P6, plus the atom-indexed stereo
axis P8 and P3 charge promoted to error). It is VOID-ONLY: a failure degrades
the caller to its next rung / abstain, NEVER to a wrong name. It is the
hardening layer on top of -- not a replacement for -- the load-bearing 0-wrong
net, which remains ``_rt_match``'s isomeric OPSIN round-trip (SELF-01).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .binding_spine import (BindingSpine, STRICT_STEREO_CHARGE_AXES,
                            verify_spine)
from .e1_certificate import verify_certificate

if TYPE_CHECKING:  # type-only; no runtime dependency on general_engine
    from ..assembly.general_engine import GeneralEngineResult

logger = logging.getLogger(__name__)


def certify_general_result(mol, result: "GeneralEngineResult", *,
                           allow_charged: bool = False) -> bool:
    """True iff ``result`` passes BOTH E1 and the binding-spine proof.

    ``allow_charged`` is threaded to BOTH proofs so they never disagree about
    scope: ``t4_coverage`` passes ``False`` (``NET_CHARGE_OUT_OF_SCOPE`` voids
    any net charge there); the namer complete-tier lanes pass
    ``self._allow_aromatic_general`` so a legitimately-charged complete-tier
    name (``-ylium``/``-ide`` suffix on a bound atom) is NOT voided.

    Mirrors ``t4_coverage``'s two existing call sites exactly
    (``verify_certificate`` then ``verify_spine(mode="audit", escalate=
    STRICT_STEREO_CHARGE_AXES)``), so refactoring those to call this is
    behaviour-preserving; the two namer lanes gain the ``verify_spine`` half
    they were missing.

    Fail-closed: a certification that raises returns ``False`` (the candidate is
    not certified) rather than propagating -- a proof bug must never turn a
    naming into a crash, and a non-certification only ever degrades to the next
    rung / abstain.
    """
    if result is None:
        return False
    try:
        if not verify_certificate(mol, result, allow_charged=allow_charged).ok:
            return False
        spine = BindingSpine.from_token_bindings(
            result.bindings,
            stereo_atom_to_locant=getattr(result, 'stereo_atom_to_locant', None))
        proof = verify_spine(mol, spine, result.name, mode="audit",
                             allow_charged=allow_charged,
                             escalate=STRICT_STEREO_CHARGE_AXES)
        if not proof.ok:
            logger.info("coverage_gate: verify_spine voided %r: %s",
                        result.name, proof.codes())
        return proof.ok
    except Exception as exc:  # fail-closed: never crash the naming path
        logger.info("coverage_gate: certification raised (voided): %s", exc)
        return False
