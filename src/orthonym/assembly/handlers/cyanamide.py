"""AM-1 cyanamide handler — retained-name parent with N-substitution.

IUPAC cite: P-66.1.6.2 (cyanamide; retained name H2N-C#N with N-substitution,
no N-locants). Mirrors the urea/guanidine Tier-B handler shape.

References:
- composer.py::_try_name_cyanamide (body).
- handlers/urea.py (template).
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_cyanamide(features: Any) -> bool:
    """Cyanamide FG present AND no principal characteristic group (the FG
    collision resolver suppresses the nitrile/amine reads so principal_group
    becomes None, exactly as for urea)."""
    fg = getattr(features, 'functional_groups', None) or {}
    return (bool(fg.get('cyanamide'))
            and getattr(features, 'principal_group', None) is None)


def name_cyanamide(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """AM-1 Tier-B cyanamide handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _try_name_cyanamide, _enrich_handler_name, _inject_stereo_if_missing,
    )

    cyanamide_name = _try_name_cyanamide(features)
    if not cyanamide_name:
        return None

    cyanamide_name = _enrich_handler_name(features, cyanamide_name, "cyanamide")

    pool = get_current_pool()
    cand = pool.add(cyanamide_name, "cyanamide", features)
    if cand is None:
        return None

    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=final_name, class_id="cyanamide",
            iupac_section_cite="P-66.1.6.2", fragment_legacy=final_name,
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_cyanamide", "_is_cyanamide"]
