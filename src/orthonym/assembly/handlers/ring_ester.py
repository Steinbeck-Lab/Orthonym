"""Phase 160 ring_ester handler — Tier-2 mid-tier direct-return.

Verbatim lift of composer.py:850-865 (inline dispatch branch). Body
``_assemble_ring_with_ester_prefixes`` at composer.py:3204-3455 (254 LOC)
STAYS until Plan-03 commit 03-10 (composer.py thinning) per CONTEXT D-24.

Predicate gates on:
1. ``principal_group == 'ester'``
2. ``exocyclic_esters`` from ``rules.esters.detect_exocyclic_esters(features.mol)``
3. ``not _is_complex_ring_system(features.mol)``

ring_ester fires BEFORE polyfunctional + ester-family + Tier-A in the inline
cascade order (composer.py:850), so no additional mutex is needed.

Byte-identical contract per CONTEXT D-21 (DECOMP-03): handler's behavior
on every canary fixture MUST equal the inline branch's behavior bit-for-bit;
verified by `python scripts/verify_decomp_byte_identical.py --mode delta`
at the atomic commit gate.

IUPAC cite: P-66.6.3 (cyclic ester with exocyclic substituents).

References:
- composer.py:850-865 (inline dispatch branch; REMOVED at this commit).
- composer.py:3204-3455 (``_assemble_ring_with_ester_prefixes`` body; STAYS
  until 03-10 thinning).
- composer.py:_is_complex_ring_system (mutex helper).
- rules.esters.detect_exocyclic_esters (predicate helper).
- 160-AUDIT-DECOMP.md § 1 row 'ring_ester' + § 3 Tier-2 row.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_ring_ester(features: Any) -> bool:
    """Predicate: principal_group == 'ester' AND exocyclic_esters AND not complex.

    Mirrors composer.py:850-853 inline guard. The exocyclic_esters check + the
    complex_ring mutex are NOT pure boolean attribute reads — they call
    ``rules.esters.detect_exocyclic_esters(mol)`` and
    ``composer._is_complex_ring_system(mol)`` respectively. Both helpers are
    pure (read-only) per CONTEXT D-25 / AP-160-26.
    """
    if getattr(features, 'principal_group', None) != 'ester':
        return False
    mol = getattr(features, 'mol', None)
    if mol is None:
        return False
    from ...rules.esters import detect_exocyclic_esters
    from ._handler_shared import cached_is_complex_ring_system

    exocyclic = detect_exocyclic_esters(mol)
    if not exocyclic:
        return False
    # WR-02: shared memoization with partial_sat / polycyclic predicates.
    if cached_is_complex_ring_system(features):
        return False
    return True


def _alcohol_is_ring(mol: Any) -> Optional[str]:
    """``'ring'`` when the ester's alkoxy carbon is a ring atom, else ``None``.

    ``find_ester_match`` returns the ``[CX3](=O)[OX2][#6]`` match, so
    ``match[3]`` is the alcohol-side carbon -- the atom the ``<R>yl`` word of a
    P-65.6.3.2.1 functional-class ester name is numbered from. ``None`` means
    "not declarable here", never "chain": the acid-side parent of such a name is
    not ``features.principal_chain`` in general, so claiming ``'chain'`` would be
    the same unproven inference this whole fix removes.
    """
    from ...rules.esters import find_ester_match
    try:
        match = find_ester_match(mol)
        if match is None or len(match) < 4:
            return None
        return 'ring' if mol.GetAtomWithIdx(int(match[3])).IsInRing() else None
    except Exception:  # noqa: BLE001 -- declaration is advisory; None = fail closed
        return None


def _functional_class_name(features: Any) -> Optional[str]:
    """P-65.6.3.2.1 functional-class name for a mono-ester, or None.

    P-65.6.3.2.1 "General methodology": *"All preferred IUPAC names for esters
    are named by functional class nomenclature."* So `cyclohexyl acetate`, not
    the substitutive `acetyloxycyclohexane` this handler otherwise builds.

    The acyloxy-prefix form is licensed by P-65.6.3.2.3 "Esters cited as
    prefixes" in two situations, and NEITHER can hold here:

      * *"another group is present that has priority for citation as the
        principal group"* — impossible: `_is_ring_ester` requires
        ``principal_group == 'ester'``, and every group that outranks an ester
        in the P-41 seniority order would have been chosen as the PG instead.
        (This is exactly the case in P-65.6.3.3.6's example
        ``CH3-CO-O-C6H4-COOH`` -> `4-(acetyloxy)benzoic acid` (PIN): there the
        carboxylic acid is senior, so the PG is not the ester and this handler
        does not fire.)
      * *"when all ester groups cannot be described by the methods prescribed
        for naming esters"* — that is precisely the fall-back below, taken when
        this function returns None.

    Restricted to the single-ester case; a ring carrying several ester groups
    is left to the prefix assembler, which can express all of them.
    """
    from ...rules.esters import find_ester_match, name_ester
    try:
        match = find_ester_match(features.mol)
        if match is None:
            return None
        return name_ester(features.mol, match) or None
    except Exception:
        return None


def name_ring_ester(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return ring_ester handler.

    Verbatim semantics of composer.py:850-865. Returns
    ``NamingResult(name=<stereo-injected name>, tree=None, atom_to_locant_hint=None)``
    on success. Wraps the result in ``_inject_stereo_if_missing`` per the
    inline branch behavior at composer.py:865.
    """
    from ...rules.esters import detect_exocyclic_esters
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _assemble_ring_with_ester_prefixes,
        _inject_stereo_if_missing,
    )

    exocyclic = detect_exocyclic_esters(features.mol)
    if not exocyclic:
        return None

    # v29 P7 C1: record WHICH parent this handler numbered the name in, so the
    # stereo injector does not have to guess it (P-91.3, BB:44639 "NAMING OF
    # STEREOISOMERS", :44643 -- a front-of-name block is read in the parent's
    # numbering). Neither features.principal_chain nor features.chain_is_parent
    # recovers it here: both say "chain" on the ring-parented rows this handler
    # emits (see composer._inject_stereo_if_missing).
    parent_scope = None
    ring_ester_name = _functional_class_name(features) if len(exocyclic) == 1 else None
    if ring_ester_name is not None:
        # P-65.6.3.2.1 functional class `<R>yl <acyl>ate`. The front-of-name
        # descriptor is numbered in the ALCOHOL component, so the ring is the
        # scope only when the alkoxy carbon is the ring atom. When it is not
        # (`methyl ...oate`, acid-side parent) the scope stays undeclared and the
        # injector fails closed rather than cite the ring's numbering.
        parent_scope = _alcohol_is_ring(features.mol)
    else:
        ring_ester_name = _assemble_ring_with_ester_prefixes(features, exocyclic)
        # Substitutive ring parent by construction -- the assembler numbers the
        # ring and cites every ester as an acyloxy prefix on it.
        parent_scope = 'ring'
    if not ring_ester_name:
        return None

    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "ring_ester", _ha, ring_ester_name[:60],
        )

    pool = get_current_pool()
    pool.add(ring_ester_name, "ring_ester", features)
    final_name = _inject_stereo_if_missing(
        features, pool.best().name, parent_scope=parent_scope,
    )
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="ring_ester", iupac_section_cite="P-65.6", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_ring_ester", "_is_ring_ester"]
