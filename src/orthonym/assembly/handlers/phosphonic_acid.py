"""v23 Phase 9 phosphonic_acid handler — substituent-prefix PIN (P-67.1.1.2).

Parallel to ``handlers.phosphinic_acid``. Without this handler an organyl
phosphonic acid R-P(=O)(OH)2 falls through to the generic suffix assembler and
is emitted in the explicitly-rejected parent-hydride-stem form
(``ethanephosphonic acid``) instead of the PIN substituent-prefix form
(``ethylphosphonic acid``). The handler intercepts ``principal_group ==
'phosphonic_acid'`` and delegates to ``rules.phosphorus.name_phosphonic_acid``,
which names the single organyl substituent; it fail-closes (returns ``None``,
cascade-continuation) for a complex substituent so the generic path is preserved
for parents the simple namer cannot handle.

IUPAC cite: P-67.1.1.2 (substitution of the central-atom H of phosphonic acid).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_phosphonic_acid(features: Any) -> bool:
    """Mirrors the phosphinic predicate (``principal_group == 'phosphonic_acid'``)."""
    return getattr(features, 'principal_group', None) == 'phosphonic_acid'


def name_phosphonic_acid(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return phosphonic_acid handler (substituent-prefix PIN)."""
    from ...rules.phosphorus import name_phosphonic_acid as _name_phosphonic_acid
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    matches = features.functional_groups.get('phosphonic_acid', [])
    if not matches:
        return None

    name = _name_phosphonic_acid(features.mol, matches[0])
    if not name:
        return None  # complex organyl -> defer to the generic path (cascade-continuation)

    pool = get_current_pool()
    pool.add(name, "phosphonic_acid", features)
    final_name = _inject_stereo_if_missing(features, pool.best().name)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="phosphonic_acid", iupac_section_cite="P-67.1", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_phosphonic_acid", "_is_phosphonic_acid"]
