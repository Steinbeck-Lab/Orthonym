"""R8c hydroxamic acid handler — PIN = 'N-hydroxy<stem>amide' (P-66.1.1.3.2 / P-65.1.3.4).

A hydroxamic acid (-C(=O)-NH-OH) is an amide with an N-hydroxy substituent.
The PIN is 'N-hydroxy<stem>amide', NOT the retained 'hydroxamic acid' suffix.

Examples:
- CC(=O)NO  -> 'N-hydroxyacetamide'
- CCC(=O)NO -> 'N-hydroxypropanamide'
- O=C(NO)C1CCCCC1 -> 'N-hydroxycyclohexanecarboxamide'

IUPAC cite: P-66.1.1.3.2 / P-65.1.3.4.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_hydroxamic_acid(features: Any) -> bool:
    """Gates on principal_group == 'hydroxamic_acid'."""
    return getattr(features, 'principal_group', None) == 'hydroxamic_acid'


def name_hydroxamic_acid(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return hydroxamic acid handler (R8c).

    Delegates chain/ring detection to name_amide() — the same function used
    by the primary-amide path — then prepends 'N-hydroxy'.  Returns None on
    any failure so the cascade continues (fail-safe per ADR-19-04).
    """
    from ...rules.amides import name_amide as _rules_name_amide
    from ..candidate_pool import get_current_pool

    _mol = mol if mol is not None else getattr(features, 'mol', None)
    if _mol is None:
        return None

    pg_atoms = getattr(features, 'principal_group_atoms', None)
    if not pg_atoms:
        return None

    amide_atoms = pg_atoms[0]  # SMARTS match: (C_carbonyl, O=, N, O-H)

    # name_amide walks amide_atoms for C and N symbols; the O-H atom is
    # inert for chain/ring detection, so passing the full tuple is safe.
    base_name = _rules_name_amide(_mol, amide_atoms, suffix_form="amide")
    if not base_name:
        return None

    result_name = f"N-hydroxy{base_name}"

    pool = get_current_pool()
    pool.add(result_name, "hydroxamic_acid", features)
    best = pool.best()
    return NamingResult(
        name=best.name,
        tree=NameTreeNode(
            parent_stem=result_name,
            class_id="hydroxamic_acid",
            iupac_section_cite="P-66.1.1.3.2",
            fragment_legacy=result_name,
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_hydroxamic_acid", "_is_hydroxamic_acid"]
