"""a phase carbamic_acid handler — Tier B retained-name (gate 0.40).

Verbatim lift of composer.py:855-863 (inline branch) +
composer.py:2498-2560 (_name_carbamic_acid body). Per internal notes,
body stays in composer.py until Plan-03 commit 03-10.

IUPAC cite: (carbamic acids; retained name with N-substitution).

References:
- composer.py:855-863 (inline dispatch branch; REMOVED at this commit).
- composer.py:2498-2560 (_name_carbamic_acid body).
- internal notes-DECOMP.md row 'carbamic_acid' + purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_carbamic_acid(features: Any) -> bool:
    """Mirrors composer.py:855 (``features.principal_group == 'carbamic_acid'``); also a principal thio acid on a carbamic
    nitrogen, 'diethylcarbamodithioic acid' (``assembly.carbamothioic_acid_names``,:30768)."""
    if getattr(features, 'principal_group', None) == 'carbamic_acid':
        return True
    from ..carbamothioic_acid_names import carbamothioic_unit
    return carbamothioic_unit(features) is not None


def name_carbamic_acid(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """a phase Tier-B carbamic acid handler.

    Verbatim semantics of composer.py:855-863.
    """
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _enrich_handler_name,
        _inject_stereo_if_missing,
        _name_carbamic_acid,
    )

    spelled = None
    if getattr(features, 'principal_group', None) == 'carbamic_acid':
        carbamic_name = _name_carbamic_acid(features)
    else:
        # the thio acid name spells every atom (an unnamed N-substituent declines it): no enrichment may cite one again
        from ..carbamothioic_acid_names import carbamothioic_acid_name
        carbamic_name = carbamothioic_acid_name(features)
        spelled = set(range(features.mol.GetNumAtoms()))
    if not carbamic_name:
        return None

    carbamic_name = _enrich_handler_name(features, carbamic_name, "carbamic_acid", already_spelled_atoms=spelled)

    pool = get_current_pool()
    cand = pool.add(carbamic_name, "carbamic_acid", features)
    if cand is None:
        return None

    final_name = _inject_stereo_if_missing(features, cand.name, atom_to_locant=None)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="carbamic_acid", iupac_section_cite="P-66.4", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_carbamic_acid", "_is_carbamic_acid"]
