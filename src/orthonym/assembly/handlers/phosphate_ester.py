"""Phase 160 phosphate_ester handler — direct-return shim.

1-line wrapper around ``rules.phosphorus.name_phosphate_ester``. Verbatim
move of composer.py:992-1013 dispatch logic; handles the three phosphate
variants (triester, diester, monoester) via principal_group membership.

IUPAC cite: P-68.3.1 (phosphoric acid esters).

References:
- composer.py:992-1013 (inline phosphate_ester branch; REMOVED at this commit).
- rules.phosphorus.name_phosphate_ester — chemical-logic body.
- 160-AUDIT-DECOMP.md § 1 row 'phosphate_ester' + § 2.23 purity proof.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NamingResult

logger = logging.getLogger(__name__)

_PHOSPHATE_ESTER_GROUPS = (
    'phosphate_triester', 'phosphate_diester', 'phosphate_monoester',
)


def _is_phosphate_ester(features: Any) -> bool:
    """Mirrors composer.py:992 (principal_group in the three phosphate variants)."""
    return getattr(features, 'principal_group', None) in _PHOSPHATE_ESTER_GROUPS


def name_phosphate_ester(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return phosphate-ester handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing
    from ...rules.phosphorus import name_phosphate_ester as _name_phosphate_ester

    fg_key = features.principal_group
    matches = features.functional_groups.get(fg_key, [])
    if not matches:
        return None

    # Find phosphorus atom index from SMARTS match (verbatim from composer.py).
    for idx in matches[0]:
        atom = features.mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'P':
            name = _name_phosphate_ester(features.mol, idx)
            if name:
                if logger.isEnabledFor(logging.DEBUG):
                    _ha = features.mol.GetNumHeavyAtoms()
                    logger.debug(
                        "HANDLER_COVERAGE: handler=%s coverage=NA accounted=NA/%d name=%s",
                        "phosphate_ester", _ha, name[:60],
                    )
                pool = get_current_pool()
                pool.add(name, "phosphate_ester", features)
                final_name = _inject_stereo_if_missing(features, pool.best().name)
                from ..name_tree import NameTreeNode  # Phase 165 SCORE-01 Path-B coarse node
                _nm = final_name
                return NamingResult(
                    name=_nm,
                    tree=NameTreeNode(parent_stem=_nm, class_id="phosphate_ester", iupac_section_cite="P-65.6", fragment_legacy=_nm),
                    atom_to_locant_hint=None,
                )
            break

    return None


__all__ = ["name_phosphate_ester", "_is_phosphate_ester"]
