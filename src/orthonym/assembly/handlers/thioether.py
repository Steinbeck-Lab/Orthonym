"""Phase 160 thioether handler — Tier B shim (gate 0.40).

Verbatim move of composer.py:970-992 dispatch logic. Wraps
``rules.sulfur.name_sulfide`` with the cyclic-thioether + fused-heterocycle
skip guards that the inline branch enforced.

IUPAC cite: P-66.5.2.4 (sulfides; functional class naming).

References:
- composer.py:970-992 (inline thioether branch; REMOVED at this commit).
- rules.sulfur.name_sulfide — chemical-logic body.
- data.fused_heterocycles.match_fused_heterocycle_core — fused-ring guard.
- 160-AUDIT-DECOMP.md § 1 row 'thioether' + § 2.21 purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_thioether(features: Any) -> bool:
    """Wave2 D7c: 'thioether' joined _PREFIX_ONLY_PRINCIPAL (it has no suffix
    form, P-63.2.2), so principal_group is never 'thioether' anymore. Mirrors
    _is_sulfoxide/_is_sulfone: the handler covers the molecule-IS-the-sulfide
    case — the thioether FG is present and no senior suffix-capable group
    claimed the PCG (else the senior parent expresses the sulfanyl prefix).
    Was composer.py:970 (``features.principal_group == 'thioether'``)."""
    return (
        getattr(features, 'principal_group', None) is None
        and bool(getattr(features, 'functional_groups', {}).get('thioether'))
    )


def name_thioether(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Tier-B thioether handler with cyclic + fused-heterocycle skip guards."""
    from ...data.fused_heterocycles import match_fused_heterocycle_core
    from ...rules.sulfur import name_sulfide
    from ..candidate_pool import get_current_pool
    from ..composer import _enrich_handler_name, _inject_stereo_if_missing

    # Skip cyclic thioethers (named as heterocycles).
    ring_type = getattr(features, 'ring_type', None)
    if ring_type and ring_type.startswith('heterocyclic'):
        return None

    # Skip known fused heterocycles (phenothiazine, thianthrene, etc.).
    if match_fused_heterocycle_core(features.mol) is not None:
        return None

    matches = features.functional_groups.get('thioether', [])
    if not matches:
        return None

    # DD5 SEN-02 (P-41 cls 40 > 41/42): the functional-class `R R' sulfide` silently
    # DROPS any co-substituent the substituent characteriser can't express — e.g. the
    # ether of COCSC -> "dimethyl sulfide". DECLINE for such LOSSY cases so the
    # substitutive carbon-parent path names it (COCSC -> methoxy(methylsulfanyl)methane,
    # via the carbon-over-ether skeletal-replacement guard + the (R)sulfanyl producer).
    #
    # SCOPED (to keep this surgical and avoid the broad sulfide -> substitutive PIN
    # migration, which is a separate follow-on with a large test/corpus surface):
    #   - NEUTRAL only — a charged species (dithiocarbamate ammonium) would route to a
    #     wrong partial substitutive name; keep the legacy path (byte-identical to HEAD);
    #   - only when the molecule carries a non-C/H/S heteroatom (an ether O, etc.) that
    #     the functional-class sulfide name would DROP. A pure C/H/S sulfide
    #     (dimethyl sulfide, methyl phenyl sulfide) keeps its established functional-class
    #     name — unchanged from HEAD.
    _neutral = all(a.GetFormalCharge() == 0 for a in features.mol.GetAtoms())
    _has_dropped_heteroatom = any(
        a.GetSymbol() not in ('C', 'H', 'S') for a in features.mol.GetAtoms()
    )
    if _neutral and _has_dropped_heteroatom:
        return None

    sulfur_idx = matches[0][0]
    name = name_sulfide(features.mol, sulfur_idx)
    if not name:
        return None

    name = _enrich_handler_name(features, name, "thioether")

    pool = get_current_pool()
    cand = pool.add(name, "thioether", features)
    if cand is None:
        return None

    # composer.py:991 inline: _inject_stereo_if_missing(features, cand.name)
    final_name = _inject_stereo_if_missing(features, cand.name)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="thioether", iupac_section_cite="P-63.2", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_thioether", "_is_thioether"]
