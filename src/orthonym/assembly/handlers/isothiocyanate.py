"""a phase isothiocyanate handler — Tier B retained-name (gate 0.40).

Verbatim lift of composer.py:842-850 (inline branch) +
composer.py:2192-2202 (_name_isothiocyanate body) + composer.py:2204-2230
(shared _name_iso_x_cyanate helper). Per internal notes, bodies stay in
composer.py until Plan-03 commit 03-10.

IUPAC cite: (isothiocyanates; the functional-class form is non-PIN, so
this handler runs only for the non-PIN style).

References:
- composer.py:842-850 (inline dispatch branch; REMOVED at this commit).
- composer.py:2192-2202 (_name_isothiocyanate body).
- composer.py:2204-2230 (shared _name_iso_x_cyanate helper; identical to isocyanate).
- internal notes-DECOMP.md row 'isothiocyanate' + predicate purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_isothiocyanate(features: Any) -> bool:
    """Mirrors composer.py:842 (isothiocyanate FG present AND principal_group is None)."""
    fg = getattr(features, 'functional_groups', None) or {}
    return bool(fg.get('isothiocyanate')) and getattr(features, 'principal_group', None) is None


def name_isothiocyanate(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """a phase Tier-B isothiocyanate handler.

    Verbatim semantics of composer.py:842-850 (inline branch).
    """
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _enrich_handler_name,
        _inject_stereo_if_missing,
        _name_isothiocyanate,
    )

    # 'ISOCYANATES' (the Blue Book),:26001: "Preferred IUPAC names
    # are generated substitutively using the prefix 'isocyanato' attached
    # directly to a parent hydride." The chalcogen analogue follows it:
    # 'isothiocyanatobenzene (PIN) phenyl isothiocyanate' (:26009). Functional-
    # class 'R isothiocyanate' is general nomenclature only, so this handler
    # declines under PIN style for EVERY attachment (aromatic included; see
    # isocyanate.name_isocyanate) and the form stays available under --trivial.
    if style == "pin":
        return None

    isothio_name = _name_isothiocyanate(features)
    if not isothio_name:
        return None

    isothio_name = _enrich_handler_name(features, isothio_name, "isothiocyanate")

    pool = get_current_pool()
    cand = pool.add(isothio_name, "isothiocyanate", features)
    if cand is None:
        return None

    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="isothiocyanate", iupac_section_cite="P-66.5", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_isothiocyanate", "_is_isothiocyanate"]
