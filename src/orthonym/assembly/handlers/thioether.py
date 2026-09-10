"""a phase thioether handler — Tier B shim (gate 0.40).

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

    sulfur_idx = matches[0][0]
    name = name_sulfide(features.mol, sulfur_idx)
    if not name:
        return None

    # P-63.2.5 (the Blue Book verbatim, section heading "P-63.2.5 Names of chalcogen
    # analogues of ethers, i.e., sulfides, selenides and tellurides": "Method (1),
    # substitutive nomenclature, gives preferred IUPAC names"): the functional-class
    # "R R' sulfide" (method 2) is NOT the PIN for a chalcogen analogue of an ether.
    # The PIN is substitutive — (R'-sulfanyl)RH, on the senior parent hydride RH
    # (the Blue Book "(methylsulfanyl)methane (PIN)... dimethyl sulfide"; the Blue Book
    # "(methylsulfanyl)benzene (PIN) (not thioanisole)"). DECLINE for the neutral
    # acyclic case so the substitutive (R)sulfanyl producer names it — the same path
    # selenium/tellurium already use (they have no functional-class handler, so
    # C[Se]C -> (methylselanyl)methane today).
    #
    # This SUBSUMES the DD5 SEN-02 lossy-heteroatom decline (the old
    # `_neutral and _has_dropped_heteroatom` guard, e.g. COCSC -> methoxy-
    # (methylsulfanyl)methane): those molecules are neutral, so they decline here too.
    #
    # SCOPE / degrade (0-wrong ABSOLUTE, P-63.2.5 degrade policy):
    # - NEUTRAL only. A charged species (a dithiocarbamate ammonium, a nitrile-
    # sulfide ylide) would route to a wrong partial substitutive name; it keeps
    # the legacy functional-class path (byte-identical to HEAD). name_sulfide
    # already returns None for a ring S, so on reaching here the S is acyclic.
    # - The symmetric-diaryl multiplicative PIN (P-63.2.5 method 3,
    # 1,1'-sulfanediyldibenzene) and the skeletal-replacement chains are produced
    # by OTHER paths that win before this handler; declining does not disturb them.
    _neutral = all(a.GetFormalCharge() == 0 for a in features.mol.GetAtoms())
    if _neutral:
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
