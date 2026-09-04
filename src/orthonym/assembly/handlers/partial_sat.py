"""Phase 160 partial_sat handler — Tier B shim (gate 0.40).

Verbatim move of composer.py:1116-1131 dispatch logic. Wraps the inline
``_try_partially_saturated_carbocycle(features.mol)`` check + enrichment +
pool.add gate.

Per CONTEXT incremental migration: the underlying
``_try_partially_saturated_carbocycle`` body STAYS in composer.py during
Plan-02 and moves to this module in Plan-03 commit 03-10.

Byte-identical contract: same mutex as polycyclic — the inline gate is
nested under ``if not _complex_ring_accepted and not chain_is_parent`` so
the predicate here uses ``not _is_complex_ring_system`` for the fast-path.
The fallback for complex-ring-rejection cases lives in the inline
partial_sat block at composer.py:1112+ (unchanged in Plan-02).

IUPAC cite: P-25.3 (partially saturated carbocycles; tetrahydronaphthalene).

References:
- composer.py:1116-1131 (inline partial_sat branch; TRIMMED at this commit).
- composer.py:_try_partially_saturated_carbocycle (STAYS until 03-10).
- composer.py:_enrich_handler_name (STAYS until 03-10).
- 160-AUDIT-DECOMP.md § 1 row 'partial_sat' + § 2.27 purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_partial_sat(features: Any) -> bool:
    """Predicate: is_cyclic AND not chain_is_parent AND not complex ring system.

    WR-02: the ``_is_complex_ring_system`` SMARTS check is memoized on the
    features object via ``cached_is_complex_ring_system`` so partial_sat,
    polycyclic, and ring_ester predicates share one evaluation per dispatch
    instead of three. CONTEXT predicate purity is preserved — the cache
    is per-features-instance state owned by features itself.
    """
    from rdkit import Chem

    from ._handler_shared import cached_is_complex_ring_system

    if not getattr(features, 'is_cyclic', False):
        return False
    if getattr(features, 'chain_is_parent', False):
        return False
    mol = getattr(features, 'mol', None)
    if mol is None:
        return False
    # A ring ketone (ring C=O) makes this a cyclic-oxo compound, not a bare
    # partially-saturated carbocycle — _try_partially_saturated_carbocycle would
    # DROP the C=O (2,3-dihydronaphthalene-1,4-dione -> 2,3-dihydronaphthalene,
    # a wrong structure suppressed by SELF-01). Decline so tier_a_ring@4500
    # (name_cyclic_oxo_compound) names it correctly (P-58.2.5 / P-58.2.3.1.2).
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.IsInRing():
            for b in atom.GetBonds():
                o = b.GetOtherAtom(atom)
                if (b.GetBondType() == Chem.BondType.DOUBLE
                        and o.GetSymbol() == 'O' and o.GetDegree() == 1
                        and not o.IsInRing()):
                    return False
    if cached_is_complex_ring_system(features):
        # WSD-02 (RING-04): an ortho-fused tetralin is ALWAYS "complex", which
        # used to veto the (correct) partially-saturated-carbocycle namer and let
        # the generic path re-emit the saturated bridge as a phantom alkyl
        # (`4-butyl-1,2,3,4-tetrahydronaphthalene`). Lift the veto ONLY for the
        # hydro-PAH class: a fused system that retains >= 1 fully-aromatic ring
        # AND has >= 1 fully-saturated carbocyclic ring. Decalin (no aromatic
        # ring) keeps the veto and its PIN-correct `decahydronaphthalene` path;
        # indane is safe because `name_partially_saturated_carbocycle` returns
        # None for it (the handler then defers to the retained-name path).
        rings = mol.GetRingInfo().AtomRings()
        ring_atom_idxs = set().union(*rings) if rings else set()
        # Atom-level mix test (a fused tetralin has NO fully-saturated SSSR ring —
        # its sp3 ring shares the two aromatic fusion carbons), so require an
        # aromatic ring ATOM AND an sp3 ring CARBON anywhere in the ring system.
        has_aromatic_ring_atom = any(
            mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_atom_idxs
        )
        has_sp3_ring_carbon = any(
            (not mol.GetAtomWithIdx(i).GetIsAromatic())
            and mol.GetAtomWithIdx(i).GetSymbol() == 'C'
            for i in ring_atom_idxs
        )
        # WSD-02 (code-review HI-01): restrict to the 2-ring tetralin topology the
        # partial-saturation namer numbers correctly. A 3+-ring acene (e.g.
        # 9,10-dihydroanthracene) gets a WRONG hydro-locant from this path
        # (`1,2-dihydroanthracene`), a confidently-wrong-different-molecule name —
        # so 3+-ring fused systems keep the veto and defer to the existing path.
        only_two_rings = len(rings) == 2
        if not (has_aromatic_ring_atom and has_sp3_ring_carbon and only_two_rings):
            return False
    return True


def name_partial_sat(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Tier-B partial_sat handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _enrich_handler_name,
        _try_partially_saturated_carbocycle_with_locants,
    )

    produced = _try_partially_saturated_carbocycle_with_locants(features.mol)
    if not produced:
        return None
    partial_sat_name = produced.name
    atom_to_locant = produced.atom_to_locant

    # Inherit BOTH halves of the producer's answer:
    #
    #  * its numbering -- the parent name, its hydro locants and any
    #    principal-characteristic-group suffix were all spelled from THIS map,
    #    so enrichment must place its substituent prefixes on the same one or it
    #    spells a different molecule (`1-methyl-` for a 2-substituted tetralin);
    #  * what it already spelled -- this producer names AROMATIC-ring
    #    substituents itself and leaves only the sp3-ring ones to enrichment,
    #    so re-citing them yields `6-methyl-6-methyl-...`.
    #
    # Both were suppressed by SELF-01 rather than shipped, i.e. each cost a
    # correct name. P-58.2.5 / P-15.1.5.3.
    partial_sat_name = _enrich_handler_name(
        features, partial_sat_name, "partial_sat",
        atom_to_locant=atom_to_locant or None,
        already_spelled_atoms=produced.spelled_offring_atoms or None,
    )

    # Phase 145.1: route through pool.add() — Tier B gate-fall-through.
    pool = get_current_pool()
    cand = pool.add(partial_sat_name, "partial_sat", features)
    if cand is None:
        # Low confidence: pool.add returned None, fall through.
        return None

    # composer.py:1130 inline: return cand.name (NO _inject_stereo wrap)
    return NamingResult(
        name=cand.name,
        tree=NameTreeNode(parent_stem=cand.name, class_id="partial_sat", iupac_section_cite="P-25.3", fragment_legacy=cand.name),
        atom_to_locant_hint=atom_to_locant or None,
    )


__all__ = ["name_partial_sat", "_is_partial_sat"]
