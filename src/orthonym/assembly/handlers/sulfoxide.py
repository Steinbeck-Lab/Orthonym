"""Phase 160 sulfoxide handler — Tier B shim (gate 0.40).

1-line wrapper around ``rules.sulfur.name_sulfoxide``. Verbatim move
of composer.py:966-979 dispatch logic (sulfoxide branch of the shared
if-else with sulfone).

IUPAC cite: P-66.5.2.4 (sulfoxides; functional class naming).

References:
- composer.py:966-979 (inline sulfoxide branch; REMOVED at this commit).
- rules.sulfur.name_sulfoxide — chemical-logic body.
- 160-AUDIT-DECOMP.md § 1 row 'sulfoxide' + § 2.19 purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_sulfoxide(features: Any) -> bool:
    """Wave2 T3b: 'sulfoxide' joined _PREFIX_ONLY_PRINCIPAL (it has no
    suffix form), so principal_group is never 'sulfoxide' anymore. The
    handler covers the molecule-IS-the-sulfoxide case: FG present and no
    senior suffix-capable group claimed the PCG (else the polyfunctional
    path expresses the sulfinyl prefix)."""
    return (
        getattr(features, 'principal_group', None) is None
        and bool(getattr(features, 'functional_groups', {}).get('sulfoxide'))
    )


def name_sulfoxide(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Tier-B sulfoxide handler."""
    from ...rules.sulfur import (
        chalcogen_oxide_fc_covers_molecule,
        name_chalcogen_oxide_substitutive,
    )
    from ...rules.sulfur import (
        name_sulfoxide as _name_sulfoxide,
    )
    from ..candidate_pool import get_current_pool
    from ..composer import _enrich_handler_name, _inject_stereo_if_missing

    matches = features.functional_groups.get('sulfoxide', [])
    if not matches:
        return None

    # Wave2 T3b conservation: both names below describe EXACTLY R-SO-R'.
    # When the molecule has atoms beyond that unit, the functional-class
    # walk silently dropped them ('CSCCS(=O)C' -> 'ethyl methyl sulfoxide',
    # -S-CH3 lost). Decline so the polyfunctional path names the whole
    # structure (1-(methanesulfinyl)-2-(methylsulfanyl)ethane, BB 18284).
    if not chalcogen_oxide_fc_covers_molecule(features.mol, matches[0]):
        return None

    # Wave2 T3b (P-63.6): substitutive is the PIN — '(methanesulfinyl)methane'
    # (BB 46154), '1-(ethanesulfinyl)butane' (28094), "1,1'-sulfinyldibenzene"
    # (28110). Functional class stays for --trivial and as the fallback for
    # shapes the substitutive builder declines (fail-open to a valid name).
    name = None
    if style == "pin":
        name = name_chalcogen_oxide_substitutive(
            features.mol, matches[0], 'sulfinyl'
        )
    if not name:
        name = _name_sulfoxide(features.mol, matches[0])
    if not name:
        return None

    name = _enrich_handler_name(features, name, "sulfoxide")

    pool = get_current_pool()
    cand = pool.add(name, "sulfoxide", features)
    if cand is None:
        return None

    # composer.py:978 inline: _inject_stereo_if_missing(features, cand.name)
    # — no explicit atom_to_locant kwarg (defaults to None).
    final_name = _inject_stereo_if_missing(features, cand.name)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="sulfoxide", iupac_section_cite="P-63.6", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_sulfoxide", "_is_sulfoxide"]
