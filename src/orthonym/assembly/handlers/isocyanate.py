""" isocyanate handler — Tier B retained-name (gate 0.40).

Verbatim lift of composer.py:828-836 (inline branch) +
composer.py:2180-2190 (_name_isocyanate body) + composer.py:2204-2230
(shared _name_iso_x_cyanate helper). Per CONTEXT incremental-
migration discipline, bodies stay in composer.py until commit
03-10 thinning.

IUPAC cite: P-66.5.4.3 (isocyanates; functional class naming).

References:
- composer.py:828-836 (inline dispatch branch; REMOVED at this commit).
- composer.py:2180-2190 (_name_isocyanate body).
- composer.py:2204-2230 (shared _name_iso_x_cyanate helper).
- 160-AUDIT-DECOMP.md § 1 row 'isocyanate' + § 2.5 predicate purity proof.
- 160-PATTERNS.md § 6 (Tier-B lift handler pattern).
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_isocyanate(features: Any) -> bool:
    """Mirrors composer.py:828 (isocyanate FG present AND principal_group is None).

    Pure read-only per CONTEXT /.
    """
    fg = getattr(features, 'functional_groups', None) or {}
    return bool(fg.get('isocyanate')) and getattr(features, 'principal_group', None) is None


def name_isocyanate(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """ Tier-B isocyanate handler.

    Verbatim semantics of composer.py:828-836 (inline branch).
    """
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _name_isocyanate, _enrich_handler_name, _inject_stereo_if_missing,
    )

    # Wave2 T2a (P-61.8): the functional-class 'R isocyanate' form is
    # general nomenclature only — the PIN is the substitutive isocyanato
    # prefix on the parent hydride (BB VERBATIM 'isocyanatocyclohexane
    # (PIN) cyclohexyl isocyanate'). Decline under PIN style so the
    # generic prefix path (NO_SENIORITY_GROUPS) emits 'isocyanatoethane';
    # the functional-class name stays available under --trivial. SCOPED to
    # NON-AROMATIC attachment: the benzene ring path cannot emit
    # 'isocyanatobenzene' yet, so aryl isocyanates keep the RT-valid
    # functional-class form rather than regressing to unknown (deferred
    # with the benzene FG-prefix table).
    if style == "pin" and not _aromatic_attachment(features, 'isocyanate'):
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


def _aromatic_attachment(features: Any, fg_key: str) -> bool:
    """True when the iso(thio)cyanate N is attached to an AROMATIC atom.

    SMARTS match layout: (R_atom, N, C, O/S) — R_atom is match[0].
    """
    mol = getattr(features, 'mol', None)
    if mol is None:
        return False
    for match in (getattr(features, 'functional_groups', None) or {}).get(fg_key, []):
        if match and mol.GetAtomWithIdx(match[0]).GetIsAromatic():
            return True
    return False


__all__ = ["name_isocyanate", "_is_isocyanate", "_aromatic_attachment"]
