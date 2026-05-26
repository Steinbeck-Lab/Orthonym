"""Phase 160 Plan-06 composite handler: ester family (polyfunctional → multi_ester → ester).

Per CONTEXT D-28 (gap-closure) + ADR-19-02 §3.1 Option A: encodes the
multi-try-on-None cascade for the 3 ester-family handlers as a SINGLE
composite handler so the dispatch_inner first-match-wins interface
(CONTEXT D-22) stays untouched.

LIFT SOURCE: composer.py:859-949 (verbatim, with ``return pool.best().name``
replaced by ``return NamingResult(name=pool.best().name,
tree=<coarse NameTreeNode>, atom_to_locant_hint=None)`` — Phase 165 SCORE-01
attaches a coarse tree at every return site).

Internal cascade order (preserved from composer.py inline body):
1. polyfunctional (if features.is_polyfunctional)
2. multi_ester (if features.principal_group == 'ester' AND >= 2 ester matches)
   - dicarboxylic_diester / polyol_polyester / independent sub-classifications
3. ester (if features.principal_group == 'ester' AND ester_match)

Returns None if ALL three sub-paths fall through (the surrounding
dispatch_inner loop then continues to the next handler — typically the
Tier-A ring cascade composite).

IUPAC cite: P-65.6.3 (ester family); P-66.6 (polyfunctional); P-65.6.3.4 (diester).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NamingResult, NameTreeNode

logger = logging.getLogger(__name__)


def _is_ester_family(features: Any) -> bool:
    """Predicate: matches ONLY when one of the three sub-paths can actually
    produce a name (mirrors the success-condition gates inside
    rules.polyfunctional.name_polyfunctional and rules.esters.name_ester).

    Per CONTEXT D-25 + AP-160-26 predicate-purity. Pure read-only on
    features (RingInfo + atom traversal are OK — D-25 bans mutation, not
    expensive reads).

    Why tight, not broad: dispatch_inner is first-match-wins (D-22). If
    the predicate matched ``is_polyfunctional`` unconditionally, ring-
    assembly-dominant molecules would be preempted from ring_assembly
    @2500 by this handler @1500 and fall back to a wrong name. IUPAC
    P-44.1 hierarchical seniority requires that ring_assembly handle
    these. Tight predicate preserves byte-identical canary by mirroring
    ``name_polyfunctional``'s internal success conditions
    (rules/polyfunctional.py:1083-1129 ring-as-parent path).
    """
    # Sub-path 2 / 3: ester principal group with concrete match data.
    pg = getattr(features, 'principal_group', None)
    if pg == 'ester':
        all_esters = getattr(features, 'all_ester_matches', None)
        if all_esters and len(all_esters) >= 2:
            return True
        if getattr(features, 'ester_match', None):
            return True

    # Sub-path 1: polyfunctional. Only match conditions name_polyfunctional
    # actually handles successfully.
    if getattr(features, 'is_polyfunctional', False):
        # Chain-parent polyfunctional (rules/polyfunctional.py:1077-1082):
        # name_polyfunctional handles this when principal_chain + atom_to_locant
        # are populated AND the molecule has a principal characteristic group.
        if (
            getattr(features, 'principal_chain', None)
            and getattr(features, 'atom_to_locant', None)
            and pg
        ):
            return True

        # Ring-as-parent polyfunctional (rules/polyfunctional.py:1083-1129):
        # name_polyfunctional handles this when cyclic + ring-parent + has
        # principal_group + single saturated non-aromatic ring + ring is
        # ≥35% of heavy atoms. Mirror these gates verbatim so the
        # predicate fires only when the handler will succeed.
        if (
            getattr(features, 'is_cyclic', False)
            and not getattr(features, 'chain_is_parent', False)
            and pg
        ):
            mol = getattr(features, 'mol', None)
            if mol is None:
                return False
            ring_info = mol.GetRingInfo()
            if ring_info.NumRings() != 1:
                return False
            ring_atoms = list(ring_info.AtomRings()[0])
            total_heavy = mol.GetNumHeavyAtoms()
            if total_heavy <= 0:
                return False
            if len(ring_atoms) / total_heavy < 0.35:
                return False
            ring_set = set(ring_atoms)
            for idx in ring_atoms:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetIsAromatic():
                    return False
                for bond in atom.GetBonds():
                    other = bond.GetOtherAtomIdx(idx)
                    if other in ring_set and bond.GetBondTypeAsDouble() == 2.0:
                        return False
            return True

    return False


def name_ester_family(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Composite handler: tries polyfunctional → multi_ester → ester in order.

    Returns the FIRST sub-path's NamingResult; returns None if all fall through.

    Byte-identical preservation per DECOMP-03: each sub-path's pool.add call
    is preserved verbatim from composer.py:859-949 (same handler_id, same
    features argument, same return type semantics).

    Per Phase 160.1 D-18 / ADR-19-04 handler contract: when pool.add()
    accepts a candidate but pool.best() returns None (pool rejected the
    candidate per quality threshold / wildcard / ratio-floor), the handler
    raises AttributeError verbatim mirroring the pre-Plan-03-01 inline
    cascade behavior at composer.py:875 (`return pool.best().name`).

    The AttributeError propagates UN-WRAPPED through dispatch_inner per
    the Phase 160.1 D-16 exception list in dispatch_inner; namer's broad
    `except (TypeError, KeyError, IndexError, AttributeError)` at
    namer.py:1888 catches it and falls through to _descriptive_fallback
    (wildcard-bearing molecules) OR — when called recursively from the
    decomposition engine — signals "this fragment is unnameable; try
    another decomposition strategy."

    This is the SAME exception flow as the pre-amendment inline cascade
    at composer.py:875 (`return pool.best().name`); it is NOT a new
    behavior. The cascade-removal sweep (Plan-03-01) keeps the exception
    propagation path that decomposition engine + namer.name_compound's
    broad except both depend on.
    """
    from ..candidate_pool import get_current_pool

    # ============================================================
    # Sub-path 1: polyfunctional (composer.py:859-877 verbatim lift)
    # ============================================================
    if getattr(features, 'is_polyfunctional', False):
        from ...rules.polyfunctional import name_polyfunctional
        poly_name = name_polyfunctional(features)
        if poly_name:
            if logger.isEnabledFor(logging.DEBUG):
                _ha = features.mol.GetNumHeavyAtoms()
                logger.debug(
                    "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                    "polyfunctional", _ha, poly_name[:60],
                )
            pool = get_current_pool()
            pool.add(poly_name, "polyfunctional", features)
            # Phase 160.1 Plan-03-01 D-16: pool.best().name raises
            # AttributeError when pool.best() is None — mirrors the
            # pre-amendment inline cascade at composer.py:875 verbatim.
            # The AttributeError is propagated un-wrapped through
            # dispatch_inner per the D-16 exception list, then caught by
            # namer.name_compound's broad `except (TypeError, KeyError,
            # IndexError, AttributeError)` at namer.py:1888, falling
            # through to _descriptive_fallback (or the decomposition
            # engine's mixed-decomp fragment rejection per
            # decomposition/engine.py "Mixed-decomp fragment name X
            # rejected: poor coverage"). DO NOT defensively return None
            # — that changes the byte-identical canary baseline and
            # breaks the decomposition engine's coverage-rejection path.
            _nm = pool.best().name
            return NamingResult(
                name=_nm,
                tree=NameTreeNode(parent_stem=_nm, class_id="ester_family", iupac_section_cite="P-65.6", fragment_legacy=_nm),
                atom_to_locant_hint=None,
            )
        logger.debug(
            "DROP-22 substituent_skip: reason=polyfunctional_returned_none",
        )

    # ============================================================
    # Sub-path 2: multi_ester (composer.py:881-926 verbatim lift)
    # ============================================================
    if getattr(features, 'principal_group', None) == "ester":
        all_esters = getattr(features, 'all_ester_matches', None)
        if (
            all_esters
            and len(all_esters) >= 2
            and not getattr(features, 'is_polyfunctional', False)
        ):
            from ...rules.esters import (
                classify_multi_ester,
                name_dicarboxylic_diester,
                name_polyol_polyester,
                name_independent_esters,
            )
            ester_type = classify_multi_ester(features.mol, all_esters)
            # Phase 160.1 Plan-03-01 D-16: all multi_ester sub-paths mirror
            # the pre-amendment inline cascade verbatim — pool.best().name
            # raises AttributeError when pool.best() is None, propagated
            # un-wrapped through dispatch_inner per D-16 to namer's broad
            # except (TypeError, KeyError, IndexError, AttributeError).
            if ester_type == "dicarboxylic_diester":
                diester_name = name_dicarboxylic_diester(features.mol, all_esters)
                if diester_name:
                    if logger.isEnabledFor(logging.DEBUG):
                        _ha = features.mol.GetNumHeavyAtoms()
                        logger.debug(
                            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                            "multi_ester", _ha, diester_name[:60],
                        )
                    pool = get_current_pool()
                    pool.add(diester_name, "multi_ester", features)
                    _nm = pool.best().name
                    return NamingResult(
                        name=_nm,
                        tree=NameTreeNode(parent_stem=_nm, class_id="ester_family", iupac_section_cite="P-65.6", fragment_legacy=_nm),
                        atom_to_locant_hint=None,
                    )
            elif ester_type == "polyol_polyester":
                polyol_name = name_polyol_polyester(features.mol, all_esters)
                if polyol_name:
                    if logger.isEnabledFor(logging.DEBUG):
                        _ha = features.mol.GetNumHeavyAtoms()
                        logger.debug(
                            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                            "multi_ester", _ha, polyol_name[:60],
                        )
                    pool = get_current_pool()
                    pool.add(polyol_name, "multi_ester", features)
                    _nm = pool.best().name
                    return NamingResult(
                        name=_nm,
                        tree=NameTreeNode(parent_stem=_nm, class_id="ester_family", iupac_section_cite="P-65.6", fragment_legacy=_nm),
                        atom_to_locant_hint=None,
                    )
            elif ester_type == "independent":
                indep_name = name_independent_esters(features.mol, all_esters)
                if indep_name:
                    if logger.isEnabledFor(logging.DEBUG):
                        _ha = features.mol.GetNumHeavyAtoms()
                        logger.debug(
                            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                            "multi_ester", _ha, indep_name[:60],
                        )
                    pool = get_current_pool()
                    pool.add(indep_name, "multi_ester", features)
                    _nm = pool.best().name
                    return NamingResult(
                        name=_nm,
                        tree=NameTreeNode(parent_stem=_nm, class_id="ester_family", iupac_section_cite="P-65.6", fragment_legacy=_nm),
                        atom_to_locant_hint=None,
                    )

    # ============================================================
    # Sub-path 3: ester (composer.py:931-949 verbatim lift)
    # ============================================================
    # Phase 160.1 Plan-03-01 D-16: mirrors the pre-amendment inline cascade
    # verbatim — pool.best().name raises AttributeError when pool.best()
    # is None, propagated un-wrapped through dispatch_inner per D-16.
    if getattr(features, 'principal_group', None) == "ester":
        ester_match = getattr(features, 'ester_match', None)
        if ester_match:
            from ...rules.esters import name_ester
            ester_name = name_ester(features.mol, ester_match)
            if ester_name:
                if logger.isEnabledFor(logging.DEBUG):
                    _ha = features.mol.GetNumHeavyAtoms()
                    logger.debug(
                        "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                        "ester", _ha, ester_name[:60],
                    )
                pool = get_current_pool()
                pool.add(ester_name, "ester", features)
                _nm = pool.best().name
                return NamingResult(
                    name=_nm,
                    tree=NameTreeNode(parent_stem=_nm, class_id="ester_family", iupac_section_cite="P-65.6", fragment_legacy=_nm),
                    atom_to_locant_hint=None,
                )

    # All three sub-paths fell through.
    return None


__all__ = ["name_ester_family", "_is_ester_family"]
