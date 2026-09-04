"""D3 chain_diamide handler — acyclic diamide with N-substituents.

Intercepts the mixed primary + N-substituted (and symmetric) acyclic diamide
class BEFORE ester_family(@1500) / the polyfunctional path, which otherwise
double-count the terminal secondary amide (SELF-01 -> 'unknown') or sweep the
N-alkyl carbons into pg_atom_set and drop them (principal_group_branch_overlap -> wrong molecule).

Dispatched at inner_dispatch priority 1490 (just before ester_family@1500).
The predicate is TIGHT (exactly 2 amide groups, both carbonyl carbons at chain
ends) so it never intercepts mono-amides, diacids, esters, or triamides. When
the diamide has no N-substituents (symmetric primary, e.g. butanediamide) the
handler returns the same plain parent name the general_acyclic path produced.

IUPAC cite: P-66.1.1.1.1 (acyclic diamide parent) / P-66.1.1.3.1.1
(N{locant} substituent prefixes).

References:
- rules/amides.py::name_chain_diamide (naming body).
- handlers/ring_ester.py (direct-return handler template).
- dspec-D3-mixed-diamide.json (spec).
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)

_DIAMIDE_PRINCIPAL_GROUPS = ('primary_amide', 'secondary_amide')


def _collect_amide_matches(features: Any) -> List[tuple]:
    """Union of primary_amide + secondary_amide SMARTS matches."""
    fgs = getattr(features, 'functional_groups', None) or {}
    matches: List[tuple] = []
    matches.extend(fgs.get('primary_amide', []) or [])
    matches.extend(fgs.get('secondary_amide', []) or [])
    return matches


def _is_chain_diamide(features: Any) -> bool:
    """Predicate: exactly-2 chain-end acyclic amides (primary and/or secondary).

    Pure read-only. Gates on:
    1. principal_group in {primary_amide, secondary_amide}
    2. exactly 2 total amide matches (primary + secondary union)
    3. acyclic chain parent present (principal_chain + atom_to_locant)
    4. not is_cyclic OR chain_is_parent (Tier-A mutex, same as amide handler)
    5. both amide carbonyl carbons (tuple index 0) at chain-END locants {1, N}
    """
    if getattr(features, 'principal_group', None) not in _DIAMIDE_PRINCIPAL_GROUPS:
        return False

    matches = _collect_amide_matches(features)
    if len(matches) != 2:
        return False

    chain = getattr(features, 'principal_chain', None)
    atom_to_locant = getattr(features, 'atom_to_locant', None)
    if not chain or not atom_to_locant:
        return False

    # Tier-A mutex: same pattern as the amide handler.
    is_cyclic = getattr(features, 'is_cyclic', False)
    chain_is_parent = getattr(features, 'chain_is_parent', False)
    if is_cyclic and not chain_is_parent:
        return False

    chain_len = len(chain)
    if chain_len < 2:
        return False
    end_locants = {1, chain_len}

    seen = set()
    for match in matches:
        carbonyl_c = match[0]
        locant = atom_to_locant.get(carbonyl_c)
        if locant is None or locant not in end_locants:
            return False
        seen.add(locant)
    # Both amides must be at DISTINCT chain ends.
    if seen != end_locants:
        return False

    return True


def name_chain_diamide_handler(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return chain_diamide handler.

    Delegates to ``rules.amides.name_chain_diamide``. Returns ``None`` (falls
    through to the next handler) when the naming body fails closed.
    """
    from ...rules.amides import name_chain_diamide
    from ..candidate_pool import get_current_pool

    matches = _collect_amide_matches(features)
    result = name_chain_diamide(
        features.mol, matches, features.principal_chain, features.atom_to_locant,
    )
    if not result:
        return None

    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "chain_diamide", _ha, result[:60],
        )

    pool = get_current_pool()
    pool.add(result, "chain_diamide", features)
    best = pool.best()
    return NamingResult(
        name=best.name,
        tree=NameTreeNode(
            parent_stem=best.name, class_id="chain_diamide",
            iupac_section_cite="P-66.1.1.1.1", fragment_legacy=best.name,
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_chain_diamide_handler", "_is_chain_diamide"]
