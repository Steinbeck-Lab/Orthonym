"""a phase phosphinic_acid handler — direct-return shim.

1-line wrapper around ``rules.phosphorus.name_phosphinic_acid``. Verbatim
move of composer.py:1064-1079 dispatch logic.

IUPAC cite: (phosphinic acids).

References:
- composer.py:1064-1079 (inline phosphinic_acid branch; REMOVED at this commit).
- rules.phosphorus.name_phosphinic_acid — chemical-logic body.
- internal notes-DECOMP.md row 'phosphinic_acid' + purity proof.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_phosphinic_acid(features: Any) -> bool:
    """Mirrors composer.py:1064 (``features.principal_group == 'phosphinic_acid'``)."""
    return getattr(features, 'principal_group', None) == 'phosphinic_acid'


def name_phosphinic_acid(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return phosphinic_acid handler."""
    from ...rules.phosphorus import name_phosphinic_acid as _name_phosphinic_acid
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    matches = features.functional_groups.get('phosphinic_acid', [])
    if not matches:
        return None

    name = _name_phosphinic_acid(features.mol, matches[0])
    if not name:
        return None

    if logger.isEnabledFor(logging.DEBUG):
        _ha = features.mol.GetNumHeavyAtoms()
        logger.debug(
            "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
            "phosphinic_acid", _ha, name[:60],
        )

    pool = get_current_pool()
    pool.add(name, "phosphinic_acid", features)
    # composer.py:1079 inline: _inject_stereo_if_missing(features, pool.best.name)
    # The substituent-prefix name has no numbered skeleton: the organyl carries its own
    # stereodescriptors ('[(2E)-but-2-en-1-yl]phosphonic acid'), so a front-of-name block never
    # resolves, the Blue Book;,:2869): the caller declares the scope.
    final_name = _inject_stereo_if_missing(
        features, pool.best().name, parent_scope='retained_no_locants')
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="phosphinic_acid", iupac_section_cite="P-67.1", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_phosphinic_acid", "_is_phosphinic_acid"]
