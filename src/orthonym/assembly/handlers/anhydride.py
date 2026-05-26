"""Phase 160 anhydride handler — direct-return shim.

1-line wrapper around ``rules.anhydrides.name_anhydride``. Verbatim move
of composer.py:921-934 dispatch logic.

IUPAC cite: P-66.6.3 (anhydrides; functional class naming).

References:
- composer.py:921-934 (inline dispatch branch; REMOVED at this commit).
- rules.anhydrides.name_anhydride — chemical-logic body (unchanged).
- 160-AUDIT-DECOMP.md § 1 row 'anhydride' + § 2.12 purity proof.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_anhydride(features: Any) -> bool:
    """Mirrors composer.py:921 (``features.principal_group == 'anhydride'``)."""
    return getattr(features, 'principal_group', None) == 'anhydride'


def name_anhydride(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return anhydride handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing
    from ...rules.anhydrides import name_anhydride as _name_anhydride

    anhydride_name = _name_anhydride(features)
    if not anhydride_name:
        return None

    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "anhydride", _ha, anhydride_name[:60],
        )

    pool = get_current_pool()
    pool.add(anhydride_name, "anhydride", features)
    # composer.py:943 inline: _inject_stereo_if_missing(features, pool.best().name, atom_to_locant=None)
    final_name = _inject_stereo_if_missing(
        features, pool.best().name, atom_to_locant=None,
    )
    _nm = final_name
    return NamingResult(
        name=_nm,
        tree=NameTreeNode(parent_stem=_nm, class_id="anhydride", iupac_section_cite="P-65.7", fragment_legacy=_nm),
        atom_to_locant_hint=None,
    )


__all__ = ["name_anhydride", "_is_anhydride"]
