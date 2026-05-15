"""Phase 160 Plan-06 composite handler: ester family (polyfunctional → multi_ester → ester).

Per CONTEXT D-28 (gap-closure) + ADR-19-02 §3.1 Option A: encodes the
multi-try-on-None cascade for the 3 ester-family handlers as a SINGLE
composite handler so the dispatch_inner first-match-wins interface
(CONTEXT D-22) stays untouched.

LIFT SOURCE: composer.py:859-949 (verbatim, with ``return pool.best().name``
replaced by ``return NamingResult(name=pool.best().name, tree=None,
atom_to_locant_hint=None)``).

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

from ..name_tree import NamingResult

logger = logging.getLogger(__name__)


def _is_ester_family(features: Any) -> bool:
    """Predicate: matches if any of the three sub-paths could fire.

    Per CONTEXT D-25 + AP-160-26 predicate-purity. Pure read-only on features.
    """
    if getattr(features, 'is_polyfunctional', False):
        return True
    if getattr(features, 'principal_group', None) == 'ester':
        all_esters = getattr(features, 'all_ester_matches', None)
        if all_esters and len(all_esters) >= 2:
            return True
        if getattr(features, 'ester_match', None):
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
            return NamingResult(
                name=pool.best().name, tree=None, atom_to_locant_hint=None,
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
                    return NamingResult(
                        name=pool.best().name, tree=None, atom_to_locant_hint=None,
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
                    return NamingResult(
                        name=pool.best().name, tree=None, atom_to_locant_hint=None,
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
                    return NamingResult(
                        name=pool.best().name, tree=None, atom_to_locant_hint=None,
                    )

    # ============================================================
    # Sub-path 3: ester (composer.py:931-949 verbatim lift)
    # ============================================================
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
                return NamingResult(
                    name=pool.best().name, tree=None, atom_to_locant_hint=None,
                )

    # All three sub-paths fell through.
    return None


__all__ = ["name_ester_family", "_is_ester_family"]
