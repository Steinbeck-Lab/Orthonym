"""Phase 160 phosphine_oxide handler — direct-return shim.

1-line wrapper around ``rules.phosphorus.name_phosphine_oxide``. Verbatim
move of composer.py:974-989 dispatch logic.

IUPAC cite: P-68.3 (phosphine oxides).

References:
- composer.py:974-989 (inline phosphine_oxide branch; REMOVED at this commit).
- rules.phosphorus.name_phosphine_oxide — chemical-logic body.
- 160-AUDIT-DECOMP.md § 1 row 'phosphine_oxide' + § 2.22 purity proof.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NamingResult

logger = logging.getLogger(__name__)


def _is_phosphine_oxide(features: Any) -> bool:
    """Mirrors composer.py:974 (``features.principal_group == 'phosphine_oxide'``)."""
    return getattr(features, 'principal_group', None) == 'phosphine_oxide'


def name_phosphine_oxide(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return phosphine_oxide handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing
    from ...rules.phosphorus import name_phosphine_oxide as _name_phosphine_oxide

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
    # composer.py:989 inline: _inject_stereo_if_missing(features, pool.best().name)
    final_name = _inject_stereo_if_missing(features, pool.best().name)
    return NamingResult(name=final_name, tree=None, atom_to_locant_hint=None)


__all__ = ["name_phosphine_oxide", "_is_phosphine_oxide"]
