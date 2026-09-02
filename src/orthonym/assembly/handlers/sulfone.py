""" sulfone handler — Tier B shim (gate 0.40).

1-line wrapper around ``rules.sulfur.name_sulfone``. Verbatim move
of composer.py:966-978 dispatch logic (sulfone branch parallel to
sulfoxide).

IUPAC cite: P-66.5.2.4 (sulfones; functional class naming).

References:
- composer.py:966-978 (inline sulfone branch; REMOVED at this commit).
- rules.sulfur.name_sulfone — chemical-logic body.
- 160-AUDIT-DECOMP.md § 1 row 'sulfone' + § 2.20 purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_sulfone(features: Any) -> bool:
    """Wave2 T3b: 'sulfone' joined _PREFIX_ONLY_PRINCIPAL (no suffix form);
    fire on FG presence with no PCG (parallel to _is_sulfoxide)."""
    return (
        getattr(features, 'principal_group', None) is None
        and bool(getattr(features, 'functional_groups', {}).get('sulfone'))
    )


def name_sulfone(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Tier-B sulfone handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import _enrich_handler_name, _inject_stereo_if_missing
    from ...rules.sulfur import (
        chalcogen_oxide_fc_covers_molecule,
        name_chalcogen_oxide_substitutive,
        name_sulfone as _name_sulfone,
    )

    matches = features.functional_groups.get('sulfone', [])
    if not matches:
        return None

    # Wave2 T3b conservation: decline when the R-SO2-R' unit does not cover
    # the whole molecule (parallel to the sulfoxide handler; the functional-
    # class alkyl walk silently dropped atoms beyond a heteroatom).
    if not chalcogen_oxide_fc_covers_molecule(features.mol, matches[0]):
        return None

    # Wave2 T3b (P-63.6): substitutive PIN — '(ethanesulfonyl)ethane'
    # (BB 28115), "1,1'-sulfonyldibenzene". Functional class stays for
    # --trivial / builder-declined shapes.
    name = None
    if style == "pin":
        name = name_chalcogen_oxide_substitutive(
            features.mol, matches[0], 'sulfonyl'
        )
    if not name:
        name = _name_sulfone(features.mol, matches[0])
    if not name:
        return None

    name = _enrich_handler_name(features, name, "sulfone")

    pool = get_current_pool()
    cand = pool.add(name, "sulfone", features)
    if cand is None:
        return None

    # composer.py:977 inline: _inject_stereo_if_missing(features, cand.name)
    final_name = _inject_stereo_if_missing(features, cand.name)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="sulfone", iupac_section_cite="P-63.6", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_sulfone", "_is_sulfone"]
