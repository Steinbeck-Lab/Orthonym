"""Phase 160 phosphine handler — direct-return shim with benzene skip guard.

Verbatim move of composer.py:1016-1061 dispatch logic. Covers tertiary,
secondary, and primary phosphine principal-group variants and preserves
the benzene-parent skip guard that the inline branch enforced (multi-P
or ring-with-other-substituents cases defer to benzene-as-parent naming).

IUPAC cite: P-68.3.1.2 (phosphines).

References:
- composer.py:1016-1061 (inline phosphine branch; REMOVED at this commit).
- rules.phosphorus.name_phosphine — chemical-logic body.
- 160-AUDIT-DECOMP.md § 1 row 'phosphine' + § 2.24 purity proof.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)

_PHOSPHINE_GROUPS = (
    'tertiary_phosphine', 'secondary_phosphine', 'primary_phosphine',
)


def _is_phosphine(features: Any) -> bool:
    """Mirrors composer.py:1016 (principal_group in the three phosphine variants).

    Tier-A ring mutex (D4 / P-22.2.2.1): if the molecule is cyclic and the
    chain is NOT the parent, decline so tier_a_ring@4500 handles it via the
    Hantzsch-Widman builder.  Mirrors amine.py lines 55-58 exactly.
    """
    if getattr(features, 'principal_group', None) not in _PHOSPHINE_GROUPS:
        return False
    is_cyclic = getattr(features, 'is_cyclic', False)
    chain_is_parent = getattr(features, 'chain_is_parent', False)
    if is_cyclic and not chain_is_parent:
        return False
    return True


def name_phosphine(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return phosphine handler with benzene-parent skip guard."""
    from ...rules.phosphorus import name_phosphine as _name_phosphine
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    fg_key = features.principal_group
    matches = features.functional_groups.get(fg_key, [])
    if not matches:
        return None

    # When molecule is a benzene derivative with P on the ring alongside
    # other substituents (or multiple P atoms), prefer benzene-as-parent naming.
    # The P will be expressed as a phosphanyl prefix on benzene.
    # Verbatim from composer.py:1024-1042.
    _skip_for_benzene = False
    if getattr(features, 'is_benzene', False) and matches:
        if len(matches) > 1:
            # Multiple phosphine groups -> ring should be parent
            _skip_for_benzene = True
        else:
            # Single phosphine -- check if any benzene ring atom has
            # non-P, non-ring substituent neighbors
            _bz_ring = getattr(features, 'benzene_ring', None)
            if _bz_ring:
                _bz_set = set(_bz_ring)
                for _ra in _bz_ring:
                    _ra_atom = features.mol.GetAtomWithIdx(_ra)
                    for _nbr in _ra_atom.GetNeighbors():
                        if _nbr.GetIdx() not in _bz_set and _nbr.GetSymbol() != 'P':
                            _skip_for_benzene = True
                            break
                    if _skip_for_benzene:
                        break

    if _skip_for_benzene:
        return None

    # Find phosphorus atom index (verbatim composer.py:1046-1060).
    for idx in matches[0]:
        atom = features.mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'P':
            name = _name_phosphine(features.mol, idx)
            if name:
                if logger.isEnabledFor(logging.DEBUG):
                    _ha = features.mol.GetNumHeavyAtoms()
                    logger.debug(
                        "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                        "phosphine", _ha, name[:60],
                    )
                pool = get_current_pool()
                pool.add(name, "phosphine", features)
                final_name = _inject_stereo_if_missing(features, pool.best().name)
                return NamingResult(
                    name=final_name,
                    tree=NameTreeNode(parent_stem=final_name, class_id="phosphine", iupac_section_cite="P-68.3", fragment_legacy=final_name),
                    atom_to_locant_hint=None,
                )
            break

    return None


__all__ = ["name_phosphine", "_is_phosphine"]
