"""a phase isocyanate handler — Tier B retained-name (gate 0.40).

Verbatim lift of composer.py:828-836 (inline branch) +
composer.py:2180-2190 (_name_isocyanate body) + composer.py:2204-2230
(shared _name_iso_x_cyanate helper). Per internal notes incremental-
migration discipline, bodies stay in composer.py until Plan-03 commit
03-10 thinning.

IUPAC cite: (isocyanates; the functional-class form is non-PIN, so this
handler runs only for the non-PIN style).

References:
- composer.py:828-836 (inline dispatch branch; REMOVED at this commit).
- composer.py:2180-2190 (_name_isocyanate body).
- composer.py:2204-2230 (shared _name_iso_x_cyanate helper).
- internal notes-DECOMP.md row 'isocyanate' + predicate purity proof.
- internal notes (Tier-B lift handler pattern).
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_isocyanate(features: Any) -> bool:
    """Mirrors composer.py:828 (isocyanate FG present AND principal_group is None).

    Pure read-only per internal notes / -26.
    """
    fg = getattr(features, 'functional_groups', None) or {}
    return bool(fg.get('isocyanate')) and getattr(features, 'principal_group', None) is None


def name_isocyanate(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """a phase Tier-B isocyanate handler.

    Verbatim semantics of composer.py:828-836 (inline branch).
    """
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _enrich_handler_name,
        _inject_stereo_if_missing,
        _name_isocyanate,
    )

    # 'ISOCYANATES' (the Blue Book),:26001: "Preferred IUPAC names
    # are generated substitutively using the prefix 'isocyanato' attached
    # directly to a parent hydride. Previously, functional class names were
    # recommended for this class." (BB VERBATIM 'isocyanatocyclohexane (PIN)
    # cyclohexyl isocyanate':26007). The functional-class 'R isocyanate' form is
    # general nomenclature only, so this handler declines under PIN style for
    # EVERY attachment and the generic prefix path (NO_SENIORITY_GROUPS) emits
    # 'isocyanatoethane', 'isocyanatobenzene', '1-chloro-4-isocyanatobenzene';
    # the functional-class name stays available under --trivial. (An earlier
    # carve-out kept the functional-class form for an AROMATIC attachment on the
    # premise that the benzene ring path could not emit 'isocyanatobenzene'; it
    # can, and every aryl isocyanate it kept was a non-PIN name labelled PIN.)
    if style == "pin":
        return None

    iso_name = _name_isocyanate(features)
    if not iso_name:
        return None

    iso_name = _enrich_handler_name(features, iso_name, "isocyanate")

    pool = get_current_pool()
    cand = pool.add(iso_name, "isocyanate", features)
    if cand is None:
        return None

    # composer.py:836 inline branch wraps in _inject_stereo_if_missing (atom_to_locant=None).
    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="isocyanate", iupac_section_cite="P-66.5", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_isocyanate", "_is_isocyanate"]
