"""a phase phosphine_oxide handler — direct-return shim.

1-line wrapper around ``rules.phosphorus.name_phosphine_oxide``. Verbatim
move of composer.py:974-989 dispatch logic.

IUPAC cite: (phosphine oxides).

References:
- composer.py:974-989 (inline phosphine_oxide branch; REMOVED at this commit).
- rules.phosphorus.name_phosphine_oxide — chemical-logic body.
- internal notes-DECOMP.md row 'phosphine_oxide' + purity proof.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_phosphine_oxide(features: Any) -> bool:
    """Mirrors composer.py:974 (``features.principal_group == 'phosphine_oxide'``)."""
    return getattr(features, 'principal_group', None) == 'phosphine_oxide'


def name_phosphine_oxide(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return phosphine_oxide handler."""
    from ...rules.phosphorus import name_phosphine_oxide as _name_phosphine_oxide
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    matches = features.functional_groups.get('phosphine_oxide', [])
    if not matches:
        return None

    name = _name_phosphine_oxide(features.mol, matches[0])
    if not name:
        return None

    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "phosphine_oxide", _ha, name[:60],
        )

    pool = get_current_pool()
    pool.add(name, "phosphine_oxide", features)
    # composer.py:989 inline: _inject_stereo_if_missing(features, pool.best.name)
    final_name = _inject_stereo_if_missing(features, pool.best().name)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="phosphine_oxide", iupac_section_cite="P-68.3", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_phosphine_oxide", "_is_phosphine_oxide"]
