""" Part A: the SINGLE certification gate for a general-engine result,
shared by every best-effort emission lane.

Root-cause of the per-lane drift named
(`` follow-on #1): three lanes ran
``name_general`` and gated its ``GeneralEngineResult`` DIFFERENTLY --
``assembly/t4_coverage.py`` ran E1 + ``verify_spine`` (escalated), while the
inline G1 lane (``namer.py:3900``) and the multifragment/recovery lane
(``namer.py:3161``) ran E1 ONLY, with the ``_stereo_emit_decision`` cardinality
check and the stereo-insensitive OPSIN-validity stereo carve-out downstream. A name whose bindings
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

from . import binding_spine as _bs
from .binding_spine import (BindingSpine, STRICT_STEREO_CHARGE_AXES,
                            verify_spine)
from .e1_certificate import verify_certificate

if TYPE_CHECKING:  # type-only; no runtime dependency on general_engine
    from ..assembly.general_engine import GeneralEngineResult

logger = logging.getLogger(__name__)


#: The STRUCTURAL binding-spine axes -- the ones that decide whether the name
#: denotes the right GRAPH: P1 atom partition, P2 bond totality, P3 charge
#: totality, P8 stereo. An error in any of these means the name is about a
#: different molecule (or drops/swaps a stereo feature), which is exactly the
#: swap-witness class the broad-lane wiring exists to catch.
#:
#: The NAME-SPELLING axes (P4 token spans, P5 residue, P6 token arity) are
#: DELIBERATELY EXCLUDED for the ``structural_only`` broad-lane gate: they are
#: about whether the name STRING spells the bindings, which on the best-effort
#: lanes the downstream SELF-01 OPSIN round-trip already proves (an ill-formed
#: name does not parse / does not round-trip). They also carry two measured
#: false-positive classes on legitimate general-engine output -- P6
#: MULTIPLICITY/ARITY_MISMATCH on the engine's one-binding-per-multiplied-group
#: convention (``carbonitrile`` claiming both C#N of a ``dicarbonitrile``), and
#: the P5 residue on the same -- so blocking on them regressed correct names to
#: uglier T4 fallbacks. t4_coverage keeps the FULL proof (``structural_only``
#: False) -- its certified Phase-0c behaviour is unchanged.
#: Only PROVABLE structural disagreements block -- the name demonstrably denotes
#: a different graph. The "unproven, not disproven" codes (``CHARGE_UNVERIFIED``,
#: the ``*_UNVERIFIED`` arity/valence/stereo codes, ``PROOF_UNSUBSTANTIATED``)
#: are EXCLUDED: they fire because a legacy binding producer never threads the
#: evidence (e.g. the charge-suffix path names a zwitterion's ``-ium``/``-olate``
#: correctly but does not populate ``charge_atom_ids``), and SELF-01's isomeric
#: round-trip already verifies charge/stereo on these lanes, so blocking on them
#: false-voids correct names (measured: a mesoionic zwitterion) with no 0-wrong
#: benefit.
_STRUCTURAL_BLOCKING_CODES = frozenset({
    _bs.ATOM_DOUBLE_BOUND, _bs.ATOM_UNBOUND, _bs.ATOM_PHANTOM,          # P1
    _bs.BOND_UNCLAIMED, _bs.BOND_DOUBLE_CLAIMED, _bs.BOND_AMBIGUOUS_LINKAGE,  # P2
    _bs.CHARGE_UNCLAIMED, _bs.CHARGE_DOUBLE_CLAIMED,                    # P3 (provable)
    _bs.NET_CHARGE_OUT_OF_SCOPE,
    _bs.STEREO_DESCRIPTOR_MISSING, _bs.STEREO_DESCRIPTOR_MISMATCH,
    _bs.STEREO_PARENT_BLOCK_AMBIGUOUS, _bs.SUBSTITUENT_STEREO_MISSING,
    _bs.SUBSTITUENT_STEREO_MISMATCH,                                    # P8 (provable)
})

# : ST.3: the stereo PROOF-GAP codes -- a rejection here means the
#: binding-spine could NOT decide (it could not positionally anchor which
#: leading descriptor block is the parent's, because ``stereo_atom_to_locant``
#: is int-locant-only and a compound/primed spiro locant is unmappable), NOT
#: that the stereo is disproved. It is the deny-by-default "never guess" arm of
#: P8, exactly parallel to ``STEREO_UNVERIFIED``. The DISPROOF codes
#: (``STEREO_DESCRIPTOR_MISSING``/``_MISMATCH``, the substituent pair) are
#: DELIBERATELY EXCLUDED: those assert the emitted stereo is demonstrably wrong,
#: and are never relaxed. A gap in this set is overridden IFF the full-InChIKey
#: OPSIN round-trip -- a strictly stronger oracle than the parent-block proof --
#: positively verifies the whole structure incl. every stereodescriptor.
_STEREO_PROOF_GAP_CODES = frozenset({_bs.STEREO_PARENT_BLOCK_AMBIGUOUS})


def _full_inchikey_roundtrips(mol, name: str) -> bool:
    """True iff *name* OPSIN-round-trips to *mol*'s FULL isomeric InChIKey.

    The ground-truth 0-wrong oracle (SELF-01's own round-trip), computed here so
    a stereo PROOF-GAP void (``STEREO_PARENT_BLOCK_AMBIGUOUS``) can be overridden
    ONLY when the whole name -- every atom, bond, charge AND stereodescriptor --
    is positively verified. Strictly stronger than the binding-spine
    parent-stereo-block proof it overrides.

    Fail-CLOSED by construction: ``opsin_roundtrip_check`` returns
    ``passed=False`` for a missing OPSIN jar, an absent JVM, an unparseable name
    (e.g. pseudoasymmetric ``r``/``s`` OPSIN 2.9.0 cannot read), OR any
    constitution/stereo mismatch, and any exception here also returns False. So
    without a working OPSIN this never relaxes the gate, and a wrong stereoisomer
    (InChI mismatch) is never accepted.
    """
    if mol is None or not name:
        return False
    try:
        from rdkit import Chem
        from .opsin_roundtrip import opsin_roundtrip_check
        smiles = Chem.MolToSmiles(mol)  # canonical, isomeric (stereo retained)
        if not smiles:
            return False
        return opsin_roundtrip_check(smiles, name).get("passed") is True
    except Exception:  # fail-closed: a probe bug must never relax the gate
        return False


def certify_general_result(mol, result: "GeneralEngineResult", *,
                           allow_charged: bool = False,
                           structural_only: bool = False) -> bool:
    """True iff ``result`` passes BOTH E1 and the binding-spine proof.

    ``allow_charged`` is threaded to BOTH proofs so they never disagree about
    scope: ``t4_coverage`` passes ``False`` (``NET_CHARGE_OUT_OF_SCOPE`` voids
    any net charge there); the namer complete-tier lanes pass
    ``self._allow_aromatic_general`` so a legitimately-charged complete-tier
    name (``-ylium``/``-ide`` suffix on a bound atom) is NOT voided.

    ``structural_only`` (B4, the broad-lane wiring) blocks ONLY on the
    structural axes (``_STRUCTURAL_BLOCKING_CODES``: atom partition / bond
    totality / charge / stereo -- the swap-witness class) and treats the
    name-spelling axes (P4/P5/P6) as advisory, because on the best-effort lanes
    the downstream SELF-01 round-trip already proves the name is well-formed,
    and P5/P6 carry measured false positives on the engine's multiplied-group
    binding convention. E1 (the flat atom partition) is ALWAYS required either
    way -- it is itself structural. Default ``False`` reproduces the full-proof
    behaviour t4_coverage was certified on.

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
        if proof.ok:
            return True
        if structural_only:
            blocking = [f.code for f in proof.findings
                        if f.severity == "error"
                        and f.code in _STRUCTURAL_BLOCKING_CODES]
            if not blocking:
                logger.info("coverage_gate: name-spelling-only findings on %r "
                            "(structural gate passes): %s",
                            result.name, proof.codes())
                return True
        else:
            blocking = [f.code for f in proof.findings if f.severity == "error"]
        # ST.3: RT-gated stereo PROOF-GAP rescue. When the ONLY thing
        # blocking certification is a stereo proof gap (STEREO_PARENT_BLOCK_
        # AMBIGUOUS -- the spine could not anchor which leading block is the
        # parent's, never a disproof), accept IFF the full name round-trips to
        # the input's isomeric InChIKey. That oracle is strictly stronger than
        # the binding-spine parent-stereo-block proof: it verifies every atom,
        # bond, charge AND stereodescriptor, so a name it passes cannot be a
        # wrong molecule OR a wrong stereoisomer. A wrong stereoisomer reaches
        # this same gap (its descriptors are equally unanchorable) and is
        # rejected by the RT check (InChI mismatch); a DISPROVED descriptor is
        # STEREO_DESCRIPTOR_MISMATCH, not in _STEREO_PROOF_GAP_CODES, and is
        # never relaxed. Fail-closed with no OPSIN (passed=False -> stay strict).
        if (blocking
                and all(c in _STEREO_PROOF_GAP_CODES for c in blocking)
                and _full_inchikey_roundtrips(mol, result.name)):
            logger.info("coverage_gate: stereo proof-gap %s on %r overridden by "
                        "full-InChIKey OPSIN round-trip", blocking, result.name)
            return True
        if structural_only:
            logger.info("coverage_gate: structural void %r: %s",
                        result.name, blocking)
            return False
        logger.info("coverage_gate: verify_spine voided %r: %s",
                    result.name, proof.codes())
        return proof.ok
    except Exception as exc:  # fail-closed: never crash the naming path
        logger.info("coverage_gate: certification raised (voided): %s", exc)
        return False
