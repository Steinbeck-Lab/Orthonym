"""W3-P11 sulfonyl/sulfinyl halide handler — direct-return shim.

Wrapper around ``rules.sulfur.name_sulfonyl_halide`` (mirrors the
acid_halide shim discipline). The acid halide of a sulfonic / sulfinic
acid is a two-word functional-class name '{stem}sulfonyl {halide}' /
'{stem}sulfinyl {halide}'.

IUPAC cite: P-67.1.4.4.1 / P-68.5.0 / P-65.3.1 (acyl halides of
sulfonic/sulfinic acids; BB 'ethanesulfonyl chloride' @39650,
'propane-1-sulfonyl chloride', '4-isocyanatobenzene-1-sulfonyl chloride
(PIN)' @26014).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_sulfonyl_halide(features: Any) -> bool:
    """True iff the principal group is a sulfonyl/sulfinyl acid halide, or a
    sulfonyl cyanide (P4-3, P-66.5.1.3.2 — same cap-and-rename body)."""
    return getattr(features, 'principal_group', None) in (
        'sulfonyl_halide', 'sulfinyl_halide', 'sulfonyl_cyanide',
    )


def name_sulfonyl_halide(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return sulfonyl/sulfinyl-halide handler (cap-and-rename body in
    rules.sulfur.name_sulfonyl_halide). Lazy import keeps the
    handlers -> rules.sulfur -> namer chain off the import-time graph."""
    from ...rules.sulfur import name_sulfonyl_halide as _name_sulfonyl_halide
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    halide_name = _name_sulfonyl_halide(features, style=style)
    if not halide_name:
        return None

    pool = get_current_pool()
    pool.add(halide_name, "sulfonyl_halide", features)
    final_name = _inject_stereo_if_missing(
        features, pool.best().name, atom_to_locant=None,
    )
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=final_name, class_id="sulfonyl_halide",
            iupac_section_cite="P-67.1.4.4.1", fragment_legacy=final_name,
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_sulfonyl_halide", "_is_sulfonyl_halide"]
