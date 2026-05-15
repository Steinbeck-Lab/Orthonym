"""Phase 160 phosphinic_acid handler — direct-return shim.

1-line wrapper around ``rules.phosphorus.name_phosphinic_acid``. Verbatim
move of composer.py:1064-1079 dispatch logic.

IUPAC cite: P-68.3.1.2.2 (phosphinic acids).

References:
- composer.py:1064-1079 (inline phosphinic_acid branch; REMOVED at this commit).
- rules.phosphorus.name_phosphinic_acid — chemical-logic body.
- 160-AUDIT-DECOMP.md § 1 row 'phosphinic_acid' + § 2.25 purity proof.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NamingResult

logger = logging.getLogger(__name__)


def _is_phosphinic_acid(features: Any) -> bool:
    """Mirrors composer.py:1064 (``features.principal_group == 'phosphinic_acid'``)."""
    return getattr(features, 'principal_group', None) == 'phosphinic_acid'


def name_phosphinic_acid(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return phosphinic_acid handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing
    from ...rules.phosphorus import name_phosphinic_acid as _name_phosphinic_acid

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
    # composer.py:1079 inline: _inject_stereo_if_missing(features, pool.best().name)
    final_name = _inject_stereo_if_missing(features, pool.best().name)
    return NamingResult(name=final_name, tree=None, atom_to_locant_hint=None)


__all__ = ["name_phosphinic_acid", "_is_phosphinic_acid"]
